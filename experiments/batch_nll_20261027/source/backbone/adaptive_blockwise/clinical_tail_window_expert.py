"""Frozen prefix-trained clinical expert reading a label-free trailing window.

Short notes retain the original prefix-token contract exactly. Long notes use
the final510 content tokens with pinned CLS/SEP. This changes the input view,
including special-token placement; it does not assert out-of-distribution
invariance or an improvement before complete evaluation.
"""
import torch
from .warm_prefix10000_expert import WarmPrefix10000Expert


class ClinicalTailWindowExpert(WarmPrefix10000Expert):
    def tokenize(self, texts):
        special = self.tokenizer('', add_special_tokens=True)['input_ids']
        assert len(special) == 2 and self.max_length == 512
        raw = self.tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)['input_ids']
        ids = []
        for text, content in zip(texts, raw):
            if len(content) <= self.max_length - 2:
                ids.append(self.tokenizer(text, add_special_tokens=True, truncation=True,
                                          max_length=self.tokenized_length)['input_ids'][:self.max_length])
            else:
                ids.append([special[0], *content[-(self.max_length - 2):], special[1]])
        batch = self.tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                                   padding=True, return_tensors='pt')
        assert batch['input_ids'].shape[1] <= self.max_length
        return {key: value.to(self.device) for key, value in batch.items() if key in ('input_ids', 'attention_mask')}


def blend_scores(prefix, tail, selection):
    from scipy.special import softmax
    branch = (1 - selection['tail_weight']) * softmax(prefix / selection['temperature'], axis=1)
    return branch + selection['tail_weight'] * softmax(tail / selection['temperature'], axis=1)


def fuse(prior, prefix, tail, selection):
    import numpy as np
    branch = blend_scores(prefix, tail, selection)
    weight = selection['weight']
    if selection['gate'] == 'ambiguous':
        top = np.partition(prior, -2, axis=1)[:, -2:]
        weight = weight * (1 - np.abs(top[:, 1] - top[:, 0]) / top.sum(1).clip(1e-8))[:, None]
    elif selection['gate'] != 'constant':
        raise ValueError('Unsupported fixed gate')
    return (1 - weight) * prior + weight * branch
