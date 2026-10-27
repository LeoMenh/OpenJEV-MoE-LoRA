"""Label-free batched inference for the differentiable blockwise LoRA MoE."""
import torch


def validate_records(records):
    seen = set()
    clean = []
    for position, record in enumerate(records):
        if set(record) - {"sample_id", "text"}:
            raise ValueError("Inference accepts only sample_id and text, never labels or group assignments")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Missing or empty text at input position {position}")
        sample_id = str(record.get("sample_id", position))
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id at position {position}")
        seen.add(sample_id)
        clean.append({"sample_id": sample_id, "text": record["text"]})
    return clean


@torch.inference_mode()
def predict_records(model, tokenizer, records, device, max_length=512, batch_size=8, top_k=10):
    if min(max_length, batch_size, top_k) < 1:
        raise ValueError("max_length, batch_size and top_k must be positive")
    records = validate_records(records)
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        encoded = tokenizer([row["text"] for row in chunk], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        logits, _ = model(input_ids=encoded["input_ids"].to(device),
                          attention_mask=encoded["attention_mask"].to(device), mode="soft")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite inference logits")
        values, indices = logits.softmax(-1).topk(min(top_k, logits.shape[-1]), dim=-1)
        values, indices = values.cpu().tolist(), indices.cpu().tolist()
        routes = {str(block): weights.detach().cpu().tolist()
                  for block, weights in model.context.probabilities.items()}
        for i, row in enumerate(chunk):
            yield {"sample_id": row["sample_id"], "pred_class_id": indices[i][0],
                   "top_class_ids": indices[i], "top_probabilities": values[i],
                   "block_mixture_weights": {block: weights[i] for block, weights in routes.items()}}
