#!/usr/bin/env python3
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.formulas import class_weights, confidence_class_balanced_loss
from adaptive_blockwise.io_utils import atomic_torch_save, load_split_csv, parse_str_list, read_json, seed_everything, write_json
from adaptive_blockwise.experiment_contract import validate_partition_groups
from adaptive_blockwise.modeling import LinearSoftmaxTop1Router, load_adapter_state, replace_with_blockwise_adapters
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer, make_dataset, resolve_model_compute_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage teacher-forced and sequential predicted-route router training.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def num_labels(config):
    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


class RouterCollator:
    def __init__(self, tokenizer, num_blocks):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)
        self.num_blocks = int(num_blocks)

    def __call__(self, features):
        labels, confidences, stripped = {}, {}, []
        for feature in features:
            item = dict(feature)
            for block in range(self.num_blocks):
                labels.setdefault(block, []).append(int(item.pop(f"block_{block}_group")))
                confidences.setdefault(block, []).append(float(item.pop(f"block_{block}_confidence")))
                item.pop(f"block_{block}_probabilities", None)
            stripped.append(item)
        batch = self.base(stripped)
        for block in range(self.num_blocks):
            batch[f"block_{block}_group"] = torch.tensor(labels[block], dtype=torch.long)
            batch[f"block_{block}_confidence"] = torch.tensor(confidences[block], dtype=torch.float32)
        return batch


def input_device(model):
    return model.model.device if hasattr(model.model, "device") and model.model.device.type != "meta" else next(model.parameters()).device


def move_router_batch(raw_batch, device, num_blocks):
    labels = {block: raw_batch.pop(f"block_{block}_group").to(device) for block in range(num_blocks)}
    confidence = {block: raw_batch.pop(f"block_{block}_confidence").to(device) for block in range(num_blocks)}
    batch = {key: value.to(device) for key, value in raw_batch.items()}
    return batch, labels, confidence


def route_mode(labels, target_block, num_blocks):
    if target_block is None:
        return labels
    return {block: labels[block] for block in range(target_block, num_blocks)}


def split_router_weight_sum(frame, weights, num_blocks):
    """Compute Z = sum_{i,b} confidence_i,b * class_weight_b,label."""
    total = 0.0
    cpu_weights = {block: value.detach().cpu() for block, value in weights.items()}
    for block in range(num_blocks):
        labels = torch.tensor(
            frame[f"block_{block}_group"].astype(int).to_numpy(copy=True), dtype=torch.long
        )
        confidence = torch.tensor(
            frame[f"block_{block}_confidence"].astype(float).to_numpy(copy=True), dtype=torch.float64
        )
        total += float((confidence * cpu_weights[block].double()[labels]).sum().item())
    if total <= 0.0:
        print(
            "WARNING: router global Z is zero; using Z=1 so the zero-weight router loss remains zero",
            flush=True,
        )
        return 1.0
    return total


def run_epoch(
    model,
    routers,
    loader,
    device,
    num_blocks,
    weights,
    dataset_size,
    global_z,
    target_block=None,
    train=False,
    optimizer=None,
    checkpoint_path=None,
    epoch_seed=0,
):
    # Experts are frozen during router optimization; keep their dropout deterministic.
    model.eval()
    for router in routers.values():
        router.train(train)
    total_loss = 0.0
    total_items = 0
    accuracies = {block: [0, 0] for block in range(num_blocks)}
    completed = 0
    loader.generator = torch.Generator().manual_seed(epoch_seed)
    if loader.sampler is not None and hasattr(loader.sampler, "generator"):
        loader.sampler.generator = torch.Generator().manual_seed(epoch_seed)
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["num_batches"] != len(loader) or saved["target_block"] != target_block:
            raise ValueError("Router checkpoint loader/target contract changed")
        load_router_states(routers, saved["routers"])
        if train:
            optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        completed = saved["completed"]
        total_loss, total_items, accuracies = saved["total_loss"], saved["total_items"], saved["accuracies"]
        print(f"Resume routers {checkpoint_path.stem}: {completed}/{len(loader)}", flush=True)
    for step, raw_batch in enumerate(loader, start=1):
        if step <= completed:
            continue
        batch, labels, confidence = move_router_batch(raw_batch, device, num_blocks)
        forced = route_mode(labels, target_block, num_blocks)
        model.set_online(routers, batch["attention_mask"], forced_routes=forced)
        with torch.set_grad_enabled(train):
            _ = model(**batch)
            selected = range(num_blocks) if target_block is None else [target_block]
            losses = []
            batch_normalizer = batch["input_ids"].shape[0] * float(global_z) / float(dataset_size)
            for block in selected:
                logits = model.route_context.router_logits.get(block)
                if logits is None:
                    raise RuntimeError(f"Router logits missing for block {block}")
                losses.append(
                    confidence_class_balanced_loss(
                        logits,
                        labels[block],
                        confidence[block],
                        weights[block],
                        normalizer=batch_normalizer,
                    ).to(device)
                )
                prediction = logits.argmax(dim=-1)
                block_labels = labels[block].to(prediction.device)
                accuracies[block][0] += int((prediction == block_labels).sum().item())
                accuracies[block][1] += int(labels[block].numel())
            loss = torch.stack(losses).sum()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch["input_ids"].shape[0]
        total_items += int(batch["input_ids"].shape[0])
        model.clear_routes()
        if checkpoint_path is not None and (step % 100 == 0 or step == len(loader)):
            atomic_torch_save({
                "routers": copy_router_states(routers),
                "optimizer": optimizer.state_dict() if train else None,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "completed": step, "num_batches": len(loader), "target_block": target_block,
                "total_loss": total_loss, "total_items": total_items, "accuracies": accuracies,
            }, checkpoint_path)
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"router {'train' if train else 'validation'} target={target_block}: {step}/{len(loader)}; loss={total_loss / max(total_items, 1):.6f}", flush=True)
    metrics = {"loss": total_loss / max(total_items, 1)}
    for block in range(num_blocks):
        if accuracies[block][1]:
            metrics[f"block_{block}_accuracy"] = accuracies[block][0] / accuracies[block][1]
    return metrics


def block_value(value, block_id):
    if isinstance(value, dict):
        return int(value.get(str(block_id), value.get(block_id)))
    return int(value)


def build_routers(model, blocks, hidden_size, num_groups):
    routers = torch.nn.ModuleDict()
    for block in blocks:
        block_id = int(block["block_id"])
        start = int(block["start"])
        device = next(model.decoder_layers[start].parameters()).device
        routers[str(block_id)] = LinearSoftmaxTop1Router(hidden_size, block_value(num_groups, block_id)).to(device)
    return routers


def copy_router_states(routers):
    return {key: {name: value.detach().cpu().clone() for name, value in router.state_dict().items()} for key, router in routers.items()}


def load_router_states(routers, state):
    for key, router in routers.items():
        router.load_state_dict(state[key])


def main():
    args = parse_args()
    config = read_json(args.config)
    seed_everything(int(config["seed"]))
    run_dir = Path(config["run_dir"])
    partition = read_json(run_dir / "02_partition" / "partition.json")
    validate_partition_groups(config, partition)
    assignment_manifest = read_json(run_dir / "04_sample_groups" / "manifest.json")
    expert_meta = read_json(run_dir / "05_experts" / "run_meta.json")
    output_dir = run_dir / "06_routers"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not (output_dir / "checkpoints").exists():
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    blocks = partition["blocks"]
    num_blocks = len(blocks)
    train_df = load_split_csv(assignment_manifest["paths"]["train"], require_groups=True, num_blocks=num_blocks)
    val_df = load_split_csv(assignment_manifest["paths"]["validation"], require_groups=True, num_blocks=num_blocks)
    tokenizer = load_tokenizer(config["model_name_or_path"])
    columns = [f"block_{block}_{suffix}" for block in range(num_blocks) for suffix in ("group", "confidence", "probabilities")]
    train_ds = make_dataset(train_df, tokenizer, int(config["max_length"]), extra_columns=columns)
    val_ds = make_dataset(val_df, tokenizer, int(config["max_length"]), extra_columns=columns)
    collator = RouterCollator(tokenizer, num_blocks)
    train_loader = DataLoader(train_ds, batch_size=int(config.get("router_train_batch_size", config.get("router_batch_size", 1))), shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("router_validation_batch_size", config.get("router_batch_size", 1))), shuffle=False, collate_fn=collator)

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
    model, replaced = replace_with_blockwise_adapters(
        base,
        blocks,
        parse_str_list(config["target_modules"]),
        config["num_groups"],
        int(expert_meta["r_shared"]),
        expert_meta["r_private"],
        int(expert_meta["alpha"]),
        float(expert_meta["dropout"]),
    )
    load_adapter_state(model, run_dir / "05_experts" / "blockwise_adapters.pt")
    for parameter in model.parameters():
        parameter.requires_grad = False
    hidden_size = int(model.model.config.hidden_size)
    routers = build_routers(model, blocks, hidden_size, config["num_groups"])
    device = input_device(model)
    weights = {}
    train_group_counts = {}
    validation_group_counts = {}
    empty_train_groups = {}
    for block in range(num_blocks):
        train_labels = train_df[f"block_{block}_group"].astype(int).to_numpy(copy=True)
        validation_labels = val_df[f"block_{block}_group"].astype(int).to_numpy(copy=True)
        groups = block_value(config["num_groups"], block)
        train_counts = np.bincount(train_labels, minlength=groups)
        validation_counts = np.bincount(validation_labels, minlength=groups)
        train_group_counts[str(block)] = [int(value) for value in train_counts]
        validation_group_counts[str(block)] = [int(value) for value in validation_counts]
        missing = np.flatnonzero(train_counts == 0).astype(int).tolist()
        if missing:
            empty_train_groups[str(block)] = missing
            print(
                f"router block {block}: empty train groups {missing}; "
                "their class weights are zero and no synthetic samples are added",
                flush=True,
            )
        weights[block] = class_weights(train_labels, groups).to(device)
    train_z = split_router_weight_sum(train_df, weights, num_blocks)
    validation_z = split_router_weight_sum(val_df, weights, num_blocks)

    history = []
    stage1_params = [parameter for router in routers.values() for parameter in router.parameters()]
    optimizer = torch.optim.AdamW(stage1_params, lr=float(config["router_learning_rate"]), weight_decay=0.0)
    for epoch in range(int(config["router_stage1_epochs"])):
        train_metrics = run_epoch(
            model,
            routers,
            train_loader,
            device,
            num_blocks,
            weights,
            len(train_df),
            train_z,
            train=True,
            optimizer=optimizer,
            checkpoint_path=checkpoint_dir / f"stage1_epoch_{epoch}_train.pt",
            epoch_seed=int(config["seed"]) + epoch,
        )
        val_metrics = run_epoch(
            model,
            routers,
            val_loader,
            device,
            num_blocks,
            weights,
            len(val_df),
            validation_z,
            train=False,
            checkpoint_path=checkpoint_dir / f"stage1_epoch_{epoch}_validation.pt",
        )
        history.append({"stage": "teacher_forced", "epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}})
        print(history[-1], flush=True)

    for target_block in range(num_blocks):
        if block_value(config["num_groups"], target_block) == 1:
            print(f"Block {target_block} has K=1: fixed route 0, no stage-2 optimization needed", flush=True)
            continue
        target_params = list(routers[str(target_block)].parameters())
        optimizer = torch.optim.AdamW(target_params, lr=float(config["router_learning_rate"]), weight_decay=0.0)
        best_loss = float("inf")
        best_state = copy_router_states(routers)
        for epoch in range(int(config["router_stage2_epochs"])):
            train_metrics = run_epoch(
                model,
                routers,
                train_loader,
                device,
                num_blocks,
                weights,
                len(train_df),
                train_z,
                target_block=target_block,
                train=True,
                optimizer=optimizer,
                checkpoint_path=checkpoint_dir / f"stage2_block_{target_block}_epoch_{epoch}_train.pt",
                epoch_seed=int(config["seed"]) + 1000 * (target_block + 1) + epoch,
            )
            val_metrics = run_epoch(
                model,
                routers,
                val_loader,
                device,
                num_blocks,
                weights,
                len(val_df),
                validation_z,
                target_block=target_block,
                train=False,
                checkpoint_path=checkpoint_dir / f"stage2_block_{target_block}_epoch_{epoch}_validation.pt",
            )
            history.append({"stage": f"predicted_route_block_{target_block}", "epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}})
            print(history[-1], flush=True)
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_state = copy_router_states(routers)
        load_router_states(routers, best_state)

    torch.save(copy_router_states(routers), output_dir / "routers.pt")
    for block, router in routers.items():
        torch.save(router.state_dict(), output_dir / f"block_{block}_router.pt")
    pd.DataFrame(history).to_csv(output_dir / "router_training_history.csv", index=False)
    write_json(
        output_dir / "run_meta.json",
        {
            "format": "adaptive_blockwise_routers_v1",
            "router_formula": "r_i_b=W_b z_i_b+b_b; rho=softmax(r); e=argmax rho",
            "routing_mode": "top1",
            "stage_1": "all blocks teacher-forced hidden states",
            "stage_2": "sequential block target with upstream predicted routes and downstream teacher routes",
            "blocks": blocks,
            "hidden_size": hidden_size,
            "num_groups": config["num_groups"],
            "router_loss_normalization": "global_Z",
            "train_router_Z": train_z,
            "validation_router_Z": validation_z,
            "train_group_counts": train_group_counts,
            "validation_group_counts": validation_group_counts,
            "empty_train_groups": empty_train_groups,
            "empty_group_policy": "zero class weight, no synthetic samples, training continues",
            "train_identity_sha256": assignment_manifest["split_identity_sha256"]["train"],
            "validation_identity_sha256": assignment_manifest["split_identity_sha256"]["validation"],
            "replaced_modules": replaced,
        },
    )
    print(f"Wrote two-stage block routers to {output_dir}")


if __name__ == "__main__":
    main()
