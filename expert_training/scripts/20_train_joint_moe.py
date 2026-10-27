#!/usr/bin/env python3
"""Joint supervised LoRA-MoE training; validation-only model selection, exact resume."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Dataset
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_blockwise.io_utils import atomic_torch_save, dataframe_identity, load_split_csv, read_json, seed_everything, sha256_file, write_json
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer, make_dataset
from adaptive_blockwise.modeling import load_adapter_state, replace_with_blockwise_adapters
from adaptive_blockwise.joint_moe import JointMoE


class GlobalBatchSampler(Sampler):
    """Resume directly at a global minibatch; padding has zero weight, never duplicate loss."""
    def __init__(self, size, local_batch, rank, world, seed, epoch, shuffle, start=0):
        self.size, self.local, self.rank, self.world = size, local_batch, rank, world
        self.seed, self.epoch, self.shuffle, self.start = seed, epoch, shuffle, start

    def __len__(self):
        return math.ceil(self.size / (self.local * self.world)) - self.start

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist() if self.shuffle else list(range(self.size))
        global_batch = self.local * self.world
        for step in range(self.start, math.ceil(self.size / global_batch)):
            start = step * global_batch + self.rank * self.local
            yield [order[i] if i < self.size else -1 for i in range(start, start + self.local)]


class WeightedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[max(index, 0)])
        row["sample_weight"] = float(index >= 0)
        row["position"] = index
        return row


class Collator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, rows):
        rows = [dict(row) for row in rows]
        weights = torch.tensor([row.pop("sample_weight") for row in rows])
        positions = torch.tensor([row.pop("position") for row in rows])
        output = self.base(rows)
        output["sample_weight"], output["position"] = weights, positions
        return output


def build_model(source, config, local_rank):
    source_run = Path(source["run_dir"])
    expert = read_json(source_run / "05_experts/run_meta.json")
    tokenizer = load_tokenizer(source["model_name_or_path"])
    base = load_base_classifier(source["model_name_or_path"], expert["num_labels"], tokenizer,
                                no_4bit=source["no_4bit"], device_map={"": local_rank}, compute_dtype=torch.bfloat16)
    if not source["no_4bit"]:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base.config.use_cache = False
    for name, parameter in base.named_parameters():
        parameter.requires_grad = any(fragment in name for fragment in ("score.", "classifier.", "classification_head."))
    core, _ = replace_with_blockwise_adapters(base, expert["blocks"], source["target_modules"], source["num_groups"],
                                             expert["r_shared"], expert["r_private"], expert["alpha"], expert["dropout"])
    load_adapter_state(core, source_run / "05_experts/blockwise_adapters.pt")
    model = JointMoE(core, expert["blocks"], source["num_groups"], config["router_width"])
    if config.get("gradient_checkpointing", False):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError("All trainable parameters must be FP32")
    return model, tokenizer


def prepare_split(name, source, config, tokenizer, rank):
    run = Path(config["run_dir"])
    manifest = read_json(Path(source["run_dir"]) / "00_splits/manifest.json")
    target = run / "tokenized" / name
    contract = {"identity": manifest["dataframe_sha256"][name], "max_length": source["max_length"],
                "model": source["model_name_or_path"], "prediction": "sequence_classification"}
    if rank == 0:
        done = target / "contract.json"
        if done.exists():
            if read_json(done) != contract:
                raise ValueError("Cached tokenized dataset contract mismatch")
        else:
            frame = load_split_csv(manifest["paths"][name])
            if dataframe_identity(frame) != contract["identity"]:
                raise ValueError(f"Immutable {name} identity mismatch")
            dataset = make_dataset(frame, tokenizer, source["max_length"])
            dataset.save_to_disk(target)
            write_json(done, contract)
    dist.barrier()
    return load_from_disk(target)


def loader_for(dataset, config, tokenizer, rank, world, epoch=0, train=False, start=0):
    sampler = GlobalBatchSampler(len(dataset), config["local_batch_size"] if train else config["eval_local_batch_size"],
                                 rank, world, config["seed"], epoch, train, start)
    return DataLoader(WeightedDataset(dataset), batch_sampler=sampler, collate_fn=Collator(tokenizer),
                      generator=torch.Generator().manual_seed(config["seed"] + epoch), num_workers=0)


def evaluate(model, dataset, config, tokenizer, rank, world, device, tag, predictions=False):
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    rows = []
    loader = loader_for(dataset, config, tokenizer, rank, world)
    started = time.time()
    with torch.no_grad():
        for step, raw in enumerate(loader, 1):
            labels = raw["labels"].to(device)
            weights = raw["sample_weight"].to(device)
            logits, _ = model(input_ids=raw["input_ids"].to(device), attention_mask=raw["attention_mask"].to(device))
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            if not torch.isfinite(losses).all():
                raise FloatingPointError(f"Non-finite {tag} loss")
            order = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
            totals[0] += weights.sum()
            totals[1] += (losses * weights).sum()
            for slot, k in enumerate((1, 5, 10), 2):
                totals[slot] += ((order[:, :k] == labels[:, None]).any(-1) * weights).sum()
            if predictions:
                for position, label, prediction, weight in zip(raw["position"].tolist(), labels.tolist(), order.tolist(), weights.tolist()):
                    if weight:
                        rows.append({"position": position, "true_label": label, "pred_label": prediction[0],
                                     "top5_labels": " ".join(map(str, prediction[:5])), "top10_labels": " ".join(map(str, prediction))})
            if rank == 0 and (step == 1 or step % 50 == 0 or step == len(loader)):
                print(f"{tag}: {step}/{len(loader)} distributed batches", flush=True)
    dist.all_reduce(totals)
    n, loss, acc1, acc5, acc10 = totals.tolist()
    if int(n) != len(dataset):
        raise ValueError("Evaluation duplicated or dropped real samples")
    metrics = {"num_samples": int(n), "loss": loss / n, "acc1": acc1 / n, "acc5": acc5 / n, "acc10": acc10 / n,
               "wall_seconds": time.time() - started, "uses_test_labels_for_routing": False}
    if predictions:
        directory = Path(config["run_dir"]) / "test_predictions"
        directory.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(directory / f"rank_{rank}.csv", index=False)
        dist.barrier()
        if rank == 0:
            result = pd.concat([pd.read_csv(directory / f"rank_{r}.csv") for r in range(world)]).sort_values("position")
            if result.position.tolist() != list(range(len(dataset))):
                raise ValueError("Prediction positions are not an exact test cover")
            result.to_csv(directory / "predictions.csv", index=False)
    if rank == 0:
        print(tag, json.dumps(metrics), flush=True)
    model.train()
    return metrics


def save_latest(model, optimizer, config, cursor, best, history, local_totals, rank, world):
    local = {"rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state(), "totals": local_totals.cpu()}
    all_rank_states = [None] * world if rank == 0 else None
    dist.gather_object(local, all_rank_states, dst=0)
    if rank == 0:
        atomic_torch_save({"format": "joint_moe_checkpoint_v1", "config": config, "world_size": world,
                           "model": model.trainable_state(), "optimizer": optimizer.state_dict(), "cursor": cursor,
                           "best": best, "history": history, "ranks": all_rank_states}, Path(config["run_dir"]) / "latest.pt")
        write_json(Path(config["run_dir"]) / "progress.json", {"status": "training", **cursor, "best_validation_acc1": best,
                                                               "updated_at": time.time(), "checkpoint": "latest.pt"})
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_joint_moe_v1.json")
    args = parser.parse_args()
    config = read_json(args.config)
    source = read_json(config["source_config"])
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if world * config["local_batch_size"] != config["effective_batch_size"]:
        raise ValueError("This run requires global effective batch=world*local, without accumulation")
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    if rank == 0:
        provenance = {"config": config, "source_config": source,
                      "source_expert_sha256": sha256_file(Path(source["run_dir"]) / "05_experts/blockwise_adapters.pt"),
                      "source_split_manifest_sha256": sha256_file(Path(source["run_dir"]) / "00_splits/manifest.json"),
                      "joint_source_sha256": sha256_file(ROOT / "adaptive_blockwise/joint_moe.py"),
                      "trainer_source_sha256": sha256_file(Path(__file__))}
        if (run / "provenance.json").exists() and read_json(run / "provenance.json") != provenance:
            raise ValueError("Run provenance changed; do not mix different experiments")
        write_json(run / "provenance.json", provenance)
    dist.barrier()
    model, tokenizer = build_model(source, config, local_rank)
    train_ds = prepare_split("train", source, config, tokenizer, rank)
    val_ds = prepare_split("validation", source, config, tokenizer, rank)
    groups = {"expert": [], "head": [], "router": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = "router" if name.startswith("routers.") else "head" if ".score." in name else "expert"
            groups[key].append(p)
    optimizer = torch.optim.AdamW([{"params": params, "lr": config[f"{key}_lr"], "initial_lr": config[f"{key}_lr"]} for key, params in groups.items()], weight_decay=0.)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    seed_everything(config["seed"] + rank)
    batches = math.ceil(len(train_ds) / config["effective_batch_size"])
    cursor = {"epoch": 0, "next_batch": 0, "global_step": 0}
    best, history = -1., []
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    if (run / "latest.pt").exists():
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config or saved["world_size"] != world:
            raise ValueError("Training checkpoint config/world mismatch")
        model.load_trainable_state(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        cursor, best, history = saved["cursor"], saved["best"], saved["history"]
        state = saved["ranks"][rank]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        totals = state["totals"].to(device)
        if rank == 0:
            print("RESUME", cursor, flush=True)
        del saved
    else:
        initial = evaluate(model, val_ds, config, tokenizer, rank, world, device, "initial_uniform_validation")
        best = initial["acc1"]
        history.append({"global_step": 0, "epoch": 0, "validation": initial})
        if rank == 0:
            atomic_torch_save({"model": model.trainable_state(), "validation": initial, "global_step": 0, "config": config}, run / "best.pt")
            write_json(run / "history.json", history)
        save_latest(model, optimizer, config, cursor, best, history, totals, rank, world)
    if rank == 0:
        print("Trainable parameters:", {key: sum(p.numel() for p in params) for key, params in groups.items()}, flush=True)
    for epoch in range(cursor["epoch"], config["epochs"]):
        start_batch = cursor["next_batch"] if epoch == cursor["epoch"] else 0
        loader = loader_for(train_ds, config, tokenizer, rank, world, epoch, True, start_batch)
        started = time.time()
        model.train()
        for step, raw in enumerate(loader, start_batch):
            labels, weights = raw["labels"].to(device), raw["sample_weight"].to(device)
            global_count = min(config["effective_batch_size"], len(train_ds) - step * config["effective_batch_size"])
            factor = min(1., (cursor["global_step"] + 1) / config["warmup_steps"])
            factor *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cursor["global_step"] / (batches * config["epochs"])))
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            logits, balance = ddp(input_ids=raw["input_ids"].to(device), attention_mask=raw["attention_mask"].to(device))
            per_sample = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            ce_sum = (per_sample * weights).sum()
            # DDP averages ranks. All real examples contribute equally, including the final tail.
            loss = ce_sum * world / global_count
            if global_count == config["effective_batch_size"]:
                loss = loss + config["balance_coefficient"] * balance
            finite = torch.tensor(int(torch.isfinite(loss)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite joint loss; stopping at last committed checkpoint")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack([weights.sum(), ce_sum.detach(), ((logits.argmax(-1) == labels) * weights).sum()]).double()
            cursor = {"epoch": epoch, "next_batch": step + 1, "global_step": cursor["global_step"] + 1}
            if (step + 1) % config["log_steps"] == 0 or step == start_batch:
                global_totals = totals.clone()
                dist.all_reduce(global_totals)
                if rank == 0:
                    n, ce, correct = global_totals.tolist()
                    print(f"epoch={epoch + 1} batch={step + 1}/{batches} global_step={cursor['global_step']} loss={ce/n:.6f} acc1={correct/n:.4f} grad_norm={float(grad_norm):.4f} seconds_per_step={(time.time()-started)/(step+1-start_batch):.3f}", flush=True)
            if (step + 1) % config["validation_every_steps"] == 0 or step + 1 == batches:
                validation = evaluate(model, val_ds, config, tokenizer, rank, world, device, f"validation_step_{cursor['global_step']}")
                history.append({"epoch": epoch + 1, "global_step": cursor["global_step"], "validation": validation})
                if validation["acc1"] > best:
                    best = validation["acc1"]
                    if rank == 0:
                        atomic_torch_save({"model": model.trainable_state(), "validation": validation, "global_step": cursor["global_step"], "config": config}, run / "best.pt")
                if rank == 0:
                    write_json(run / "history.json", history)
                save_latest(model, optimizer, config, cursor, best, history, totals, rank, world)
            elif (step + 1) % config["checkpoint_steps"] == 0:
                save_latest(model, optimizer, config, cursor, best, history, totals, rank, world)
        cursor = {"epoch": epoch + 1, "next_batch": 0, "global_step": cursor["global_step"]}
        totals.zero_()
        save_latest(model, optimizer, config, cursor, best, history, totals, rank, world)
    test_evaluated = bool(config.get("evaluate_test_after_training", True)) and best >= config["min_validation_for_test"]
    if test_evaluated:
        chosen = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model.load_trainable_state(chosen["model"])
        # First access to the test examples in this training program occurs only after selection is frozen.
        test_ds = prepare_split("test", source, config, tokenizer, rank)
        test = evaluate(model, test_ds, config, tokenizer, rank, world, device, "final_test", predictions=True)
        if rank == 0:
            manifest = read_json(Path(source["run_dir"]) / "00_splits/manifest.json")
            write_json(run / "metrics.json", {"validation_selected_step": chosen["global_step"], "validation": chosen["validation"],
                                               "test": test, "test_identity_sha256": manifest["dataframe_sha256"]["test"],
                                               "deployable": True, "routing": "per_sample_soft_MoE_LoRA_no_labels"})
    if rank == 0:
        write_json(run / "progress.json", {"status": "run_complete", "best_validation_acc1": best,
                                           "test_evaluated": test_evaluated, "updated_at": time.time()})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
