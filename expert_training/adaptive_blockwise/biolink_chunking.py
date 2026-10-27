"""Label-independent contiguous BERT chunks with separate CLS/SEP tokens."""
import numpy as np


def pack_biolink_chunks(tokenizer, texts):
    assert tokenizer.padding_side == tokenizer.truncation_side == 'right'
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True,
                        max_length=2040, return_attention_mask=False)['input_ids']
    ids = np.full((len(texts), 4, 512), tokenizer.pad_token_id, dtype=np.int32)
    attention = np.zeros_like(ids, dtype=np.uint8)
    valid = np.zeros((len(texts), 4), dtype=np.uint8)
    for row, tokens in enumerate(encoded):
        chunks = max(1, (len(tokens) + 509) // 510)
        assert chunks <= 4
        for chunk in range(chunks):
            content = tokens[chunk * 510:(chunk + 1) * 510]
            sequence = [tokenizer.cls_token_id, *content, tokenizer.sep_token_id]
            ids[row, chunk, :len(sequence)] = sequence
            attention[row, chunk, :len(sequence)] = 1
            valid[row, chunk] = 1
    return ids, attention, valid

