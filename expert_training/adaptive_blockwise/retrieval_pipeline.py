"""Train-only dense/BM25 retrieval and text-only candidate interaction scoring."""
import hashlib
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DataCollatorWithPadding

from .io_utils import atomic_torch_save, read_json, sha256_file, write_json


def case_document(text):
    """Deterministic evidence view; selection never sees a DRG label."""
    text = str(text)
    blocks = re.split(r'\n(?=[A-Za-z][A-Za-z /_-]{2,65}:)', text)
    relevant = [b for b in blocks if re.search(r'(?im)^(?:discharge diagnos|diagnos|major surgical|procedures?|hospital course|complications?|discharge condition|discharge instructions)', b)]
    sections = '\n'.join(b[:2500] for b in relevant)
    return (sections + '\n' + text[:5000] + '\n' + text[-3000:]).strip()[:16000]


def descriptions(config):
    source = read_json(Path(config['source_run']) / 'pipeline_state/config.json')
    labels = pd.read_csv(source['id2label_csv']).sort_values('label')
    table = pd.read_csv(config['drg_descriptions_csv'], sep='\t')
    lookup = table.set_index('DRG')['Description'].to_dict()
    result = []
    for row in labels.itertuples(index=False):
        code = int(row.drg_34_code)
        if code not in lookup:
            raise ValueError(f'No fixed DRG-34 description for code {code}')
        result.append(f'MS-DRG {code}: {lookup[code]}')
    if len(result) != config['num_labels']:
        raise ValueError('Description order does not match classifier labels')
    return result


def document_groups(texts):
    return np.array([hashlib.sha256(str(t).encode()).hexdigest() for t in texts])


def bm25_matrix(counts, average_length, idf, k1=1.2, b=.75):
    counts = counts.tocsr().astype(np.float32)
    lengths = np.asarray(counts.sum(1)).ravel()
    denom = k1 * (1 - b + b * lengths / max(float(average_length), 1.))
    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    counts.data = idf[counts.indices] * counts.data * (k1 + 1.) / (counts.data + denom[row_ids])
    return counts


def build_lexical_index(config, train_frame):
    root = Path(config['run_dir']) / 'retrieval'
    root.mkdir(exist_ok=True)
    manifest_path = root / 'bm25_manifest.json'
    if manifest_path.exists():
        return
    docs = [case_document(t) for t in train_frame.text]
    vectorizer = CountVectorizer(lowercase=True, strip_accents='unicode', stop_words='english',
                                 token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b',
                                 min_df=3, max_df=.95, max_features=100000, dtype=np.float32)
    counts = vectorizer.fit_transform(docs).tocsr()
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
    n = counts.shape[0]
    idf = np.log1p((n - df + .5) / (df + .5))
    average_length = float(np.asarray(counts.sum(1)).mean())
    weighted = bm25_matrix(counts, average_length, idf)
    sparse.save_npz(root / 'bm25_documents.npz', weighted)
    with (root / 'bm25_vectorizer.pkl').open('wb') as handle:
        pickle.dump(vectorizer, handle)
    np.save(root / 'train_document_groups.npy', document_groups(train_frame.text))
    write_json(manifest_path, {'num_documents': n, 'vocabulary_size': counts.shape[1],
                              'average_length': average_length, 'k1': 1.2, 'b': .75,
                              'index_split': 'train', 'uses_validation_or_test_documents': False,
                              'rule': 'Fixed clinical sections and original head/tail, then BM25',
                              'description_csv_sha256': sha256_file(config['drg_descriptions_csv'])})


def reciprocal_rank_union(dense_ids, lexical_ids, limit=64, constant=60):
    scores = {}
    for ids in (dense_ids, lexical_ids):
        for rank, idx in enumerate(ids):
            if int(idx) >= 0:
                scores[int(idx)] = scores.get(int(idx), 0.) + 1. / (constant + rank + 1)
    return np.array(sorted(scores, key=lambda idx: (-scores[idx], idx))[:limit], dtype=np.int64)


def generate_candidates(config, split, frame, rank, world):
    root = Path(config['run_dir']) / 'retrieval'
    output = root / f'{split}_candidates'
    output.mkdir(exist_ok=True)
    if (output / f'rank_{rank}.done.json').exists():
        return
    train = np.load(root / 'train_embeddings.npy', mmap_mode='r')
    query = np.load(root / f'{split}_embeddings.npy', mmap_mode='r')
    groups = np.load(root / 'train_document_groups.npy')
    query_groups = document_groups(frame.text)
    with (root / 'bm25_vectorizer.pkl').open('rb') as handle:
        vectorizer = pickle.load(handle)
    docs = sparse.load_npz(root / 'bm25_documents.npz').tocsc()
    positions = np.arange(rank, len(frame), world)
    cfg = config['retrieval']
    limit = cfg['candidate_limit']
    target = output / f'rank_{rank}.npy'
    resume = read_json(output / f'rank_{rank}.progress.json')['completed'] if (output / f'rank_{rank}.progress.json').exists() else 0
    neighbors = np.lib.format.open_memmap(target, mode='r+' if target.exists() else 'w+', dtype=np.int64,
                                         shape=(len(positions), limit))
    keys = torch.tensor(np.asarray(train), device='cuda', dtype=torch.float32)
    batch_size = cfg['query_batch']
    for offset in range(resume, len(positions), batch_size):
        indices = positions[offset:offset + batch_size]
        queries = torch.tensor(np.array(query[indices]), device='cuda', dtype=torch.float32)
        similarities = queries @ keys.T
        forbidden = query_groups[indices, None] == groups[None, :]
        similarities.masked_fill_(torch.from_numpy(forbidden).to('cuda'), -torch.inf)
        dense = similarities.topk(cfg['dense_top_k'], dim=1).indices.cpu().numpy()
        query_counts = vectorizer.transform([case_document(t) for t in frame.text.iloc[indices]])
        query_counts.data[:] = 1.
        lexical = (query_counts @ docs.T).toarray()
        lexical[forbidden] = -np.inf
        k = cfg['bm25_top_k']
        lexical_ids = np.argpartition(-lexical, kth=k - 1, axis=1)[:, :k]
        for row in range(len(indices)):
            ids = lexical_ids[row]
            ids = ids[np.lexsort((ids, -lexical[row, ids]))]
            union = reciprocal_rank_union(dense[row], ids, limit, cfg['rrf_constant'])
            if len(union) != limit:
                raise RuntimeError('Not enough distinct retrieval candidates')
            if np.any(groups[union] == query_groups[indices[row]]):
                raise RuntimeError('Self or duplicate text leaked into retrieval candidates')
            neighbors[offset + row] = union
        neighbors.flush()
        completed = min(offset + batch_size, len(positions))
        write_json(output / f'rank_{rank}.progress.json', {'completed': completed, 'total': len(positions), 'split': split})
        if rank == 0 and (offset == resume or completed % (batch_size * 10) == 0 or completed == len(positions)):
            print(f'candidates {split}: rank0 {completed}/{len(positions)} queries', flush=True)
    write_json(output / f'rank_{rank}.done.json', {'num_queries': len(positions), 'world_size': world, 'rank': rank})


def combine_candidate_shards(config, split, count, world):
    root = Path(config['run_dir']) / 'retrieval'
    source = root / f'{split}_candidates'
    path = root / f'{split}_neighbors.npy'
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.int64, shape=(count, config['retrieval']['candidate_limit']))
    for rank in range(world):
        output[rank::world] = np.load(source / f'rank_{rank}.npy', mmap_mode='r')
    output.flush()


class RerankTokenizer:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.prefix = '<|im_start|>system\nJudge whether the Document supports assigning the Query hospital case to the stated MS-DRG. Answer only "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.prefix_ids = tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(self.suffix, add_special_tokens=False)

    def row(self, query, evidence, description):
        token = self.tokenizer
        q = token.encode('<Query>: ' + str(query), add_special_tokens=False)[:416]
        doc = token.encode('\n<Document>: ' + case_document(evidence), add_special_tokens=False)[:384]
        desc = token.encode('\n<MS-DRG>: ' + description, add_special_tokens=False)[:96]
        body = q + doc + desc
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        if len(body) > budget:
            # Preserve the DRG description and the mandatory answer suffix.
            body = (q + doc)[:budget - len(desc)] + desc
        ids = self.prefix_ids + body + self.suffix_ids
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


class RerankTrainingDataset:
    def __init__(self, frame):
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {'position': max(index, 0), 'sample_weight': float(index >= 0)}


class RerankTrainingCollator:
    def __init__(self, config, frame, tokenizer):
        self.config, self.frame = config, frame
        self.groups = {int(label): np.asarray(indices) for label, indices in frame.groupby('label').groups.items()}
        self.neighbors = np.load(Path(config['run_dir']) / 'retrieval/train_neighbors.npy', mmap_mode='r')
        self.descriptions = descriptions(config)
        self.builder = RerankTokenizer(tokenizer, config['reranker']['max_length'])
        self.pad = DataCollatorWithPadding(tokenizer)
        self.epoch = 0

    def __call__(self, rows):
        encoded, labels, weights, positions = [], [], [], []
        for item in rows:
            position = item['position']
            row = self.frame.iloc[position]
            same = self.groups[int(row.label)]
            same = same[same != position]
            # Supervised positive pairs are a training-only target. Inference
            # candidates are NEVER supplemented with the correct class.
            positive = int(same[(position + self.epoch) % len(same)]) if len(same) else position
            negative = next((int(i) for i in self.neighbors[position] if int(self.frame.label.iloc[i]) != int(row.label)), None)
            if negative is None:
                raise RuntimeError('Hard-negative mining found no different-label case')
            for idx, label in ((positive, 1), (negative, 0)):
                candidate = self.frame.iloc[idx]
                encoded.append(self.builder.row(row.text, candidate.text, self.descriptions[int(candidate.label)]))
                labels.append(label)
                weights.append(item['sample_weight'] * float(label == 0 or len(same) > 0))
                positions.append(position)
        result = self.pad(encoded)
        result.update(labels=torch.tensor(labels), sample_weight=torch.tensor(weights), position=torch.tensor(positions))
        return result


@torch.no_grad()
def score_reranker(config, model, query_frame, train_frame, split, rank, world, tag):
    root = Path(config['run_dir'])
    folder = root / 'reranker_scores' / tag
    folder.mkdir(parents=True, exist_ok=True)
    neighbors = np.load(root / f'retrieval/{split}_neighbors.npy', mmap_mode='r')
    count = config['retrieval']['rerank_candidates']
    positions = np.arange(rank, len(query_frame), world)
    scores = np.lib.format.open_memmap(folder / f'rank_{rank}.npy', mode='w+', dtype=np.float32, shape=(len(positions), count))
    builder = RerankTokenizer(model.tokenizer, config['reranker']['max_length'])
    pad = DataCollatorWithPadding(model.tokenizer)
    desc = descriptions(config)
    model.eval()
    for offset, position in enumerate(positions):
        row = query_frame.iloc[position]
        candidates = train_frame.iloc[np.asarray(neighbors[position, :count])]
        encoded = [builder.row(row.text, candidate.text, desc[int(candidate.label)]) for candidate in candidates.itertuples()]
        outputs = []
        for start in range(0, len(encoded), config['eval_local_batch_size']):
            batch = {k: v.cuda() for k, v in pad(encoded[start:start + config['eval_local_batch_size']]).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(**batch)
            outputs.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
        scores[offset] = np.concatenate(outputs)
        if rank == 0 and (offset == 0 or (offset + 1) % 100 == 0 or offset + 1 == len(positions)):
            print(f'rerank {tag}: rank0 {offset + 1}/{len(positions)} queries', flush=True)
    scores.flush()
    dist.barrier()
    if rank == 0:
        output = np.lib.format.open_memmap(folder / 'scores.npy', mode='w+', dtype=np.float32, shape=(len(query_frame), count))
        for r in range(world):
            output[r::world] = np.load(folder / f'rank_{r}.npy', mmap_mode='r')
        output.flush()
    dist.barrier()
    model.train()
    return folder / 'scores.npy'


def case_probabilities(scores, candidate_labels, num_labels=738, temperature=1.):
    scores = np.asarray(scores, dtype=np.float32) / temperature
    scores = scores - scores.max(1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(1, keepdims=True)
    output = np.zeros((len(scores), num_labels), dtype=np.float32)
    np.add.at(output, (np.arange(len(scores))[:, None], np.asarray(candidate_labels)), weights)
    return output
