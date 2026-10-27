"""Frozen mature512-trained weights;1536 input formed from prefix1024+tail512.

The tail is within the original common2048-token preprocessing cap. For inputs
at most1536 tokens, preserve the complete common-token sequence. No training
provenance is rewritten, and no new separator token is inserted.
"""
import torch
from .frozen_late_context_expert import FrozenLateContextExpert
from .io_utils import sha256_file
from pathlib import Path


class FrozenHeadTailClinicalExpert(FrozenLateContextExpert):
    def __init__(self, root, checkpoint, device='cuda:0', *, parent_step=18000, inference_length=1536):
        assert parent_step == 18000 and inference_length == 1536
        super().__init__(root, checkpoint, device, parent_step=parent_step, inference_length=inference_length)
        self.head_tail = True
        self.inference_config.update(operation='Within common2048tokenizer cap, preserve<=1536 else prefix1024+tail512; no new separator',
                                     prefix_tokens=1024, tail_tokens=512)
        self.artifact_hashes[str(Path(__file__))] = sha256_file(__file__)

    def tokenize(self, texts):
        if not self.head_tail:
            return super().tokenize(texts)
        assert self.max_length == 1536
        result = self.tokenizer(texts, padding=True, truncation=True, max_length=self.tokenized_length, return_tensors='pt')
        sequences = []
        for ids, attention in zip(result['input_ids'], result['attention_mask']):
            count = int(attention.sum())
            assert count > 0
            sequence = ids[:count] if count <= 1536 else torch.cat((ids[:1024], ids[count-512:count]))
            assert len(sequence) == min(count, 1536)
            sequences.append(sequence)
        width = max(len(row) for row in sequences)
        ids = torch.full((len(texts), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(sequences):
            ids[index, :len(row)] = row
            mask[index, :len(row)] = 1
        return {'input_ids': ids.to(self.device), 'attention_mask': mask.to(self.device)}
