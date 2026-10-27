"""Label-free token budget shared across the prefix and explicit problem sections."""
import numpy as np
from .clinical_summary import ANCHOR


def encode_balanced_sections(tokenizer, text, max_length=512, prefix_budget=192):
    if max_length < 16 or prefix_budget < 1:
        raise ValueError('Positive prefix and sufficient sequence budget required')
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    special = tokenizer('', add_special_tokens=True)['input_ids']
    if len(special) != 2 or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError('Balanced clinical view requires the pinned single-sequence CLS/SEP tokenizer')
    prefix = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=budget)
    matches = list(ANCHOR.finditer(text))
    if not matches or len(prefix) < budget:
        return [special[0], *prefix, special[1]]
    # Avoid concentrating the entire remaining budget in the first long problem.
    # Extremely long problem lists are sampled uniformly by textual position.
    if len(matches) > 48:
        matches = [matches[i] for i in np.linspace(0, len(matches)-1, 48, dtype=int)]
    sections = []
    for i, match in enumerate(matches):
        end = min(len(text), match.start() + 320)
        if i + 1 < len(matches):
            end = min(end, matches[i + 1].start())
        tokens = tokenizer.encode(text[match.start():end], add_special_tokens=False,
                                  truncation=True, max_length=128)
        sections.append(tokens)
    prefix = prefix[:min(prefix_budget, budget)]
    remaining = budget - len(prefix)
    allocation = [0] * len(sections)
    while remaining:
        progressed = False
        for i, tokens in enumerate(sections):
            if remaining and allocation[i] < len(tokens):
                allocation[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    content = prefix + [token for tokens, count in zip(sections, allocation) for token in tokens[:count]]
    return [special[0], *content, special[1]]


def batch_balanced_sections(tokenizer, texts, max_length=512, prefix_budget=192, return_tensors='np'):
    ids = [encode_balanced_sections(tokenizer, text, max_length, prefix_budget) for text in texts]
    return tokenizer.pad({'input_ids': ids, 'attention_mask': [[1] * len(row) for row in ids]},
                         padding='max_length', max_length=max_length, return_tensors=return_tensors)
