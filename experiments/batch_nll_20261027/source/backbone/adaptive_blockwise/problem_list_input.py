"""Fixed text-only candidate: prefix plus otherwise omitted problem headings.

This module is intentionally separate from the active model/input pipeline.
"""
import re

HEADING = re.compile(r"(?m)^[ \t]*#{1,3}[ \t]*[^\s#][^\n]{2,119}")
SEPARATOR = "\n\nAdditional problem headings:\n"


def encode_problem_list(tokenizer, text, max_length=512, heading_budget=192):
    """Return token IDs and counts; never reads labels or any other records.

    Reserve at most heading_budget tokens for headings that start beyond the
    original right-truncation boundary. Keep the original prefix in the rest.
    Short notes and notes without omitted headings remain exactly unchanged.
    """
    if not tokenizer.is_fast or tokenizer.truncation_side != "right":
        raise ValueError("Fast tokenizer with right truncation is required")
    if not 0 < heading_budget < max_length:
        raise ValueError("Heading budget must leave room for the prefix")
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    original = encoded["input_ids"]
    boundary = max((end for _, end in encoded["offset_mapping"]), default=0)
    headings = [m.group().strip() for m in HEADING.finditer(text) if m.start() >= boundary]
    metadata = {"changed": False, "omitted_heading_count": len(headings), "added_tokens": 0}
    if not headings:
        return original, metadata
    extra = tokenizer(SEPARATOR + "\n".join(headings), add_special_tokens=False)["input_ids"][:heading_budget]
    prefix = original[:max_length - len(extra)]
    result = prefix + extra
    metadata.update(changed=result != original, added_tokens=len(extra))
    return result, metadata
