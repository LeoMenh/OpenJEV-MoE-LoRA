#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import prepare_model_for_kbit_training
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.io_utils import atomic_torch_save, dataframe_identity, load_split_csv, parse_str_list, read_json, seed_everything, write_json
from adaptive_blockwise.experiment_contract import validate_partition_groups
from adaptive_blockwise.modeling import (
    load_adapter_state,
    load_router_states,
    replace_with_blockwise_adapters,
    LinearSoftmaxTop1Router,
)
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer, make_dataset, resolve_model_compute_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate learned routing and both test-set oracle definitions on one manifest.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--allow_large_oracle", action="store_true")
    return parser.parse_args()


def num_labels(config):
    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


def input_device(model):
    return model.model.device if hasattr(model.model, "device") and model.model.device.type != "meta" else next(model.parameters()).device


def topk_rows(logits, labels, position_offset=0):
    order = np.argsort(-logits, axis=1)[:, :10]
    rows = []
    for index, (label, values) in enumerate(zip(labels, order)):
        rows.append(
            {
                "position": int(position_offset) + index,
                "true_label": int(label),
                "pred_label": int(values[0]),
                "top5_labels": " ".join(str(int(value)) for value in values[:5]),
                "top10_labels": " ".join(str(int(value)) for value in values[:10]),
            }
        )
    return pd.DataFrame(rows)


def metrics(predictions):
    if len(predictions) == 0:
        return {"num_samples": 0, "acc1": 0.0, "acc5": 0.0, "acc10": 0.0}
    true = predictions["true_label"].astype(int)
    return {
        "num_samples": int(len(predictions)),
        "acc1": float((true == predictions["pred_label"].astype(int)).mean()),
        "acc5": float(np.mean([int(str(label) in set(row.split())) for label, row in zip(true, predictions["top5_labels"])])),
        "acc10": float(np.mean([int(str(label) in set(row.split())) for label, row in zip(true, predictions["top10_labels"])])),
    }


class EvalCollator:
    def __init__(self, tokenizer, num_blocks):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)
        self.num_blocks = num_blocks

    def __call__(self, features):
        groups = {}
        stripped = []
        for feature in features:
            item = dict(feature)
            for block in range(self.num_blocks):
                groups.setdefault(block, []).append(int(item.pop(f"block_{block}_group")))
                item.pop(f"block_{block}_confidence", None)
                item.pop(f"block_{block}_probabilities", None)
            stripped.append(item)
        batch = self.base(stripped)
        for block, values in groups.items():
            batch[f"block_{block}_group"] = torch.tensor(values, dtype=torch.long)
        return batch


def block_value(value, block_id):
    if isinstance(value, dict):
        return int(value.get(str(block_id), value.get(block_id)))
    return int(value)


def load_model_and_routers(config, partition, run_dir, tokenizer):
    base = load_base_classifier(
        config["model_name_or_path"],
        num_labels(config),
        tokenizer,
        no_4bit=bool(config.get("no_4bit", False)),
        compute_dtype=resolve_model_compute_dtype(config.get("expert_compute_dtype", "bfloat16")),
    )
    if not bool(config.get("no_4bit", False)):
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    model, _ = replace_with_blockwise_adapters(
        base,
        partition["blocks"],
        parse_str_list(config["target_modules"]),
        config["num_groups"],
        int(config["r_shared"]),
        config["r_private"],
        int(config["expert_lora_alpha"]),
        float(config["expert_lora_dropout"]),
    )
    load_adapter_state(model, run_dir / "05_experts" / "blockwise_adapters.pt")
    for parameter in model.parameters():
        parameter.requires_grad = False
    routers = torch.nn.ModuleDict()
    hidden_size = int(model.model.config.hidden_size)
    for block in partition["blocks"]:
        block_id = int(block["block_id"])
        device = next(model.decoder_layers[int(block["start"])].parameters()).device
        routers[str(block_id)] = LinearSoftmaxTop1Router(hidden_size, block_value(config["num_groups"], block_id)).to(device)
    state = torch.load(run_dir / "06_routers" / "routers.pt", map_location="cpu")
    load_router_states(routers, state)
    model.eval()
    routers.eval()
    return model, routers


def extract_routes(raw_batch, num_blocks, device):
    groups = {block: raw_batch.pop(f"block_{block}_group").to(device) for block in range(num_blocks)}
    batch = {key: value.to(device) for key, value in raw_batch.items()}
    return batch, groups


def run_single_route(model, routers, batch, forced_routes=None):
    model.set_online(routers, batch["attention_mask"], forced_routes=forced_routes)
    with torch.no_grad():
        output = model(**batch)
    routes = {block: value.detach().cpu().clone() for block, value in model.route_context.routes.items()}
    probabilities = {block: value.detach().cpu().clone() for block, value in model.route_context.router_probabilities.items()}
    model.clear_routes()
    return output.logits.detach().float().cpu(), routes, probabilities


def repeat_batch(batch, repeats):
    return {
        key: value.repeat_interleave(int(repeats), dim=0)
        if torch.is_tensor(value) and value.ndim > 0
        else value
        for key, value in batch.items()
    }


def evaluate_all(config, partition, test_df, model, routers, tokenizer, output_dir, allow_large_oracle):
    num_blocks = len(partition["blocks"])
    dataset = make_dataset(
        test_df,
        tokenizer,
        int(config["max_length"]),
        extra_columns=[f"block_{block}_{suffix}" for block in range(num_blocks) for suffix in ("group", "confidence", "probabilities")],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation_batch_size", 4)),
        shuffle=False,
        collate_fn=EvalCollator(tokenizer, num_blocks),
    )
    device = input_device(model)
    learned_rows, teacher_rows, best_rows, route_rows = [], [], [], []
    learned_losses, teacher_losses, replacement_losses = [], [], {block: [] for block in range(num_blocks)}
    usage = {block: {group: 0 for group in range(block_value(config["num_groups"], block))} for block in range(num_blocks)}
    labels_all = []
    route_vector_correct = 0
    all_candidates = list(itertools.product(*[range(block_value(config["num_groups"], block)) for block in range(num_blocks)]))
    if len(all_candidates) > 729 and not allow_large_oracle:
        raise ValueError(f"Exact label-informed oracle would require {len(all_candidates)} route vectors; pass --allow_large_oracle")
    learned_elapsed = 0.0
    checkpoint_path = output_dir / "evaluation_checkpoint.pt"
    completed = 0
    identity = dataframe_identity(test_df)
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["batch_size"] != int(config["evaluation_batch_size"]) or saved["num_groups"] != config["num_groups"]:
            raise ValueError("Evaluation checkpoint contract changed")
        completed = saved["completed"]
        learned_rows, teacher_rows, best_rows, route_rows = saved["rows"]
        learned_losses, teacher_losses, replacement_losses = saved["losses"]
        usage, route_vector_correct, learned_elapsed = saved["usage"], saved["route_vector_correct"], saved["learned_elapsed"]
        print(f"Resuming evaluation at {completed}/{len(loader)} batches", flush=True)
    for position, raw_batch in enumerate(loader):
        if position < completed:
            continue
        batch, gold_routes = extract_routes(raw_batch, num_blocks, device)
        labels = batch["labels"].detach().cpu().numpy().astype(int)
        batch_count = len(labels)
        learned_start = time.time()
        learned_logits, learned_routes, learned_probabilities = run_single_route(model, routers, batch)
        learned_elapsed += time.time() - learned_start
        teacher_logits, _, _ = run_single_route(model, routers, batch, forced_routes=gold_routes)
        learned_loss = torch.nn.functional.cross_entropy(learned_logits, torch.tensor(labels), reduction="none").numpy()
        teacher_loss = torch.nn.functional.cross_entropy(teacher_logits, torch.tensor(labels), reduction="none").numpy()
        learned_losses.extend(learned_loss.tolist())
        teacher_losses.extend(teacher_loss.tolist())
        labels_all.extend(labels.tolist())
        for index in range(batch_count):
            routes_correct = True
            route_row = {"position": position * int(config.get("evaluation_batch_size", 4)) + index}
            for block in range(num_blocks):
                predicted = int(learned_routes[block][index])
                gold = int(gold_routes[block][index].detach().cpu())
                usage[block][predicted] += 1
                routes_correct = routes_correct and predicted == gold
                route_row[f"block_{block}_expert"] = predicted
                route_row[f"block_{block}_gold_group"] = gold
                route_row[f"block_{block}_probabilities"] = " ".join(
                    f"{float(value):.8f}" for value in learned_probabilities[block][index].tolist()
                )
            route_vector_correct += int(routes_correct)
            route_rows.append(route_row)
        position_offset = position * int(config.get("evaluation_batch_size", 4))
        learned_rows.extend(topk_rows(learned_logits.numpy(), labels, position_offset).to_dict("records"))
        teacher_rows.extend(topk_rows(teacher_logits.numpy(), labels, position_offset).to_dict("records"))

        for block in range(num_blocks):
            replaced_logits, _, _ = run_single_route(model, routers, batch, forced_routes={block: gold_routes[block]})
            replacement_losses[block].extend(torch.nn.functional.cross_entropy(replaced_logits, torch.tensor(labels), reduction="none").numpy().tolist())

        best_logits = torch.full_like(learned_logits, -float("inf"))
        best_rank = torch.full((batch_count,), 10**9, dtype=torch.long)
        best_true_logit = torch.full((batch_count,), -float("inf"))
        candidate_batch_size = int(config.get("oracle_candidate_batch_size", 16))
        label_tensor = torch.tensor(labels, device=device, dtype=torch.long)

        def run_candidate_chunk(candidate_chunk):
            """Run a candidate chunk, halving it automatically on CUDA OOM."""
            candidate_count = len(candidate_chunk)
            expanded = repeat_batch(batch, candidate_count)
            forced = {
                block: torch.tensor(
                    [candidate[block] for _ in range(batch_count) for candidate in candidate_chunk],
                    device=device,
                    dtype=torch.long,
                )
                for block in range(num_blocks)
            }
            try:
                logits, _, _ = run_single_route(model, routers, expanded, forced_routes=forced)
                return logits.reshape(batch_count, candidate_count, -1)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not is_oom or candidate_count <= 1:
                    raise
                model.clear_routes()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                midpoint = candidate_count // 2
                return torch.cat(
                    [run_candidate_chunk(candidate_chunk[:midpoint]), run_candidate_chunk(candidate_chunk[midpoint:])],
                    dim=1,
                )

        for candidate_start in range(0, len(all_candidates), candidate_batch_size):
            candidate_chunk = all_candidates[candidate_start : candidate_start + candidate_batch_size]
            candidate_count = len(candidate_chunk)
            candidate_logits = run_candidate_chunk(candidate_chunk)
            sorted_labels = candidate_logits.argsort(dim=-1, descending=True)
            matches = sorted_labels == label_tensor[:, None, None].cpu()
            ranks = matches.to(torch.long).argmax(dim=-1)
            true_logits = candidate_logits[:, :, :].gather(
                2, label_tensor.cpu()[None, :, None].transpose(0, 1).expand(batch_count, candidate_count, 1)
            ).squeeze(-1)
            min_rank = ranks.min(dim=1).values
            masked_true = true_logits.masked_fill(ranks != min_rank[:, None], -float("inf"))
            chosen = masked_true.argmax(dim=1)
            chosen_true = true_logits[torch.arange(batch_count), chosen]
            improve = (min_rank < best_rank) | ((min_rank == best_rank) & (chosen_true > best_true_logit))
            best_logits[improve] = candidate_logits[torch.arange(batch_count), chosen][improve]
            best_rank[improve] = min_rank[improve]
            best_true_logit[improve] = chosen_true[improve]
        best_rows.extend(topk_rows(best_logits.numpy(), labels, position_offset).to_dict("records"))
        if position == 0 or (position + 1) % 25 == 0 or position + 1 == len(loader):
            atomic_torch_save({
                "identity": identity, "batch_size": int(config["evaluation_batch_size"]),
                "num_groups": config["num_groups"], "completed": position + 1,
                "rows": (learned_rows, teacher_rows, best_rows, route_rows),
                "losses": (learned_losses, teacher_losses, replacement_losses),
                "usage": usage, "route_vector_correct": route_vector_correct, "learned_elapsed": learned_elapsed,
            }, checkpoint_path)
            write_json(output_dir / "partial_metrics.json", {
                "complete": position + 1 == len(loader), "num_samples": len(learned_rows),
                "learned_router": metrics(pd.DataFrame(learned_rows)),
                "compatibility_teacher_route": metrics(pd.DataFrame(teacher_rows)),
                "label_informed_best_expert_oracle": metrics(pd.DataFrame(best_rows)),
            })
            print(f"evaluation: {min((position + 1) * int(config.get('evaluation_batch_size', 4)), len(test_df))}/{len(test_df)} samples", flush=True)

    learned_df = pd.DataFrame(learned_rows)
    teacher_df = pd.DataFrame(teacher_rows)
    best_df = pd.DataFrame(best_rows)
    route_df = pd.DataFrame(route_rows)
    learned_metrics = metrics(learned_df)
    teacher_metrics = metrics(teacher_df)
    best_metrics = metrics(best_df)
    router_accuracy = {}
    router_macro_f1 = {}
    for block in range(num_blocks):
        predicted = route_df[f"block_{block}_expert"].astype(int)
        gold = route_df[f"block_{block}_gold_group"].astype(int)
        router_accuracy[str(block)] = float((predicted == gold).mean())
        router_macro_f1[str(block)] = float(f1_score(gold, predicted, average="macro", labels=list(range(block_value(config["num_groups"], block))), zero_division=0))
    learned_loss = float(np.mean(learned_losses))
    teacher_loss = float(np.mean(teacher_losses))
    regret = {str(block): float(learned_loss - np.mean(replacement_losses[block])) for block in range(num_blocks)}
    payload = {
        "format": "adaptive_blockwise_evaluation_v1",
        "same_test_manifest": True,
        "test_identity_sha256": dataframe_identity(test_df),
        "learned_router": learned_metrics,
        "compatibility_teacher_route": teacher_metrics,
        "label_informed_best_expert_oracle": best_metrics,
        "oracle_gap_teacher_minus_learned_acc1": teacher_metrics["acc1"] - learned_metrics["acc1"],
        "router_accuracy": router_accuracy,
        "router_macro_f1": router_macro_f1,
        "route_vector_accuracy": route_vector_correct / max(len(test_df), 1),
        "blockwise_route_regret": regret,
        "expert_utilization": {str(block): {str(group): int(value) for group, value in values.items()} for block, values in usage.items()},
        "exact_best_oracle_route_vectors": len(all_candidates),
        "learned_inference_seconds": learned_elapsed,
        "learned_inference_seconds_per_sample": learned_elapsed / max(len(test_df), 1),
        "router_parameter_count": int(sum(parameter.numel() for parameter in routers.parameters())),
    }
    learned_df.to_csv(output_dir / "learned_router_predictions.csv", index=False)
    teacher_df.to_csv(output_dir / "compatibility_teacher_route_predictions.csv", index=False)
    best_df.to_csv(output_dir / "label_informed_best_expert_oracle_predictions.csv", index=False)
    route_df.to_csv(output_dir / "learned_router_assignments.csv", index=False)
    write_json(output_dir / "metrics.json", payload)
    return payload


def main():
    args = parse_args()
    config = read_json(args.config)
    seed_everything(int(config["seed"]))
    run_dir = Path(config["run_dir"])
    split_manifest = read_json(run_dir / "00_splits" / "manifest.json")
    assignment_manifest = read_json(run_dir / "04_sample_groups" / "manifest.json")
    partition = read_json(run_dir / "02_partition" / "partition.json")
    validate_partition_groups(config, partition)
    test_df = load_split_csv(assignment_manifest["paths"]["test"], require_groups=True, num_blocks=len(partition["blocks"]))
    expected_identity = split_manifest["dataframe_sha256"]["test"]
    actual_identity = dataframe_identity(test_df[["sample_id", "text", "label"]])
    if actual_identity != expected_identity:
        raise ValueError("Test assignment rows do not match the immutable test manifest")
    tokenizer = load_tokenizer(config["model_name_or_path"])
    model, routers = load_model_and_routers(config, partition, run_dir, tokenizer)
    output_dir = run_dir / "07_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate_all(config, partition, test_df, model, routers, tokenizer, output_dir, args.allow_large_oracle)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
