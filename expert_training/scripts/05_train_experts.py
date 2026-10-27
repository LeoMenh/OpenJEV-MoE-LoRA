#!/usr/bin/env python3
import argparse
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from transformers import DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_blockwise.io_utils import load_split_csv, parse_str_list, read_json, seed_everything, write_json
from adaptive_blockwise.modeling import replace_with_blockwise_adapters
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer, make_dataset, resolve_model_compute_dtype
from adaptive_blockwise.experiment_contract import local_accumulation, validate_partition_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Train layer-specific shared/private LoRA experts with teacher routes.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def num_labels(config):
    return int(pd.read_csv(config["id2label_csv"])["label"].nunique())


def distributed_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed expert training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def trainable_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model, state):
    parameters = dict(model.named_parameters())
    missing = [name for name in state if name not in parameters]
    if missing:
        raise RuntimeError(f"Checkpoint has unknown trainable parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_training_checkpoint(path, model, optimizer, epoch, next_step, loader, device):
    payload = {
        "format": "adaptive_blockwise_expert_checkpoint_v1",
        "epoch": int(epoch),
        "next_step": int(next_step),
        "num_batches": int(len(loader)),
        "world_size": int(dist.get_world_size()) if dist.is_initialized() else 1,
        "trainable_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_state(device),
        "running_totals": getattr(loader, "running_totals", (0.0, 0)),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ExactDistributedSampler(Sampler):
    """Equal collective counts; -1 slots are masked, never repeated training data."""

    def __init__(self, dataset, rank, world_size, shuffle, seed):
        self.size = len(dataset)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
        else:
            indices = list(range(self.size))
        total_size = int(math.ceil(self.size / self.world_size)) * self.world_size
        if total_size > self.size:
            indices.extend([-1] * (total_size - self.size))
        return iter(indices[self.rank : total_size : self.world_size])

    def __len__(self):
        return int(math.ceil(self.size / self.world_size))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class RouteCollator:
    def __init__(self, tokenizer, num_blocks):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)
        self.num_blocks = int(num_blocks)

    def __call__(self, features):
        routes = {block: [] for block in range(self.num_blocks)}
        stripped = []
        weights = []
        for feature in features:
            item = dict(feature)
            weights.append(float(item.pop("_sample_weight", 1.0)))
            for block in range(self.num_blocks):
                routes[block].append(int(item.pop(f"block_{block}_group")))
            for block in range(self.num_blocks):
                item.pop(f"block_{block}_confidence", None)
                item.pop(f"block_{block}_probabilities", None)
            stripped.append(item)
        batch = self.base(stripped)
        batch["_sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        for block, values in routes.items():
            batch[f"block_{block}_group"] = torch.tensor(values, dtype=torch.long)
        return batch


class MaskedPaddingDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[max(index, 0)])
        # A shape-only placeholder keeps all ranks in the same collectives.
        item["_sample_weight"] = 0.0 if index < 0 else 1.0
        return item


def input_device(model):
    if hasattr(model.model, "device") and model.model.device.type != "meta":
        return model.model.device
    return next(model.parameters()).device


def move_batch(batch, device, num_blocks):
    routes = {block: batch.pop(f"block_{block}_group").to(device) for block in range(num_blocks)}
    return {key: value.to(device) for key, value in batch.items()}, routes


def train_or_eval(
    model,
    loader,
    device,
    num_blocks,
    optimizer=None,
    split_name="train",
    log_steps=250,
    start_step=0,
    checkpoint_callback=None,
):
    training = optimizer is not None
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    accumulation_steps = int(getattr(loader, "gradient_accumulation_steps", 1))
    model.train(training)
    core_model = unwrap_model(model)
    total_loss, total = getattr(loader, "resume_totals", (0.0, 0)) if training else (0.0, 0)
    started = time.time()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(loader, start=1):
        if step <= start_step:
            continue
        batch, routes = move_batch(raw_batch, device, num_blocks)
        sample_weights = batch.pop("_sample_weight")
        core_model.set_teacher_routes(routes)
        with torch.set_grad_enabled(training):
            output = model(**batch)
            local_count = int(sample_weights.sum().item())
            per_sample_loss = F.cross_entropy(
                output.logits.float(), batch["labels"].to(output.logits.device), reduction="none"
            )
            local_loss_sum = (per_sample_loss * sample_weights.to(output.logits.device)).sum()
            if training:
                window_start = ((step - 1) // accumulation_steps) * accumulation_steps
                window_capacity = int(loader.batch_size) * world_size * accumulation_steps
                window_count = min(window_capacity, len(loader.dataset) - window_start * int(loader.batch_size) * world_size)
                # DDP averages ranks; this recovers the exact real-sample window mean.
                loss = local_loss_sum * world_size / max(window_count, 1)
            else:
                loss = local_loss_sum / max(local_count, 1)
            finite = torch.tensor(
                1 if loss is not None and torch.isfinite(loss).all() else 0,
                device=device,
                dtype=torch.int32,
            )
            if dist.is_initialized():
                # Keep all DDP ranks on the same branch when one rank sees a
                # bad batch; otherwise the healthy rank can deadlock in NCCL.
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() == 0:
                raise FloatingPointError(f"Non-finite {split_name} minibatch {step}; resume from the last committed checkpoint")
            if training:
                loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(local_loss_sum.detach().cpu())
        total += local_count
        loader.running_totals = (total_loss, total)
        if training and checkpoint_callback is not None and (step % accumulation_steps == 0 or step == len(loader)):
            checkpoint_callback(step)
        core_model.clear_routes()
        if is_main_process() and (step == 1 or step % int(log_steps) == 0 or step == len(loader)):
            elapsed = time.time() - started
            rate = (step - start_step) / max(elapsed, 1e-9)
            remaining = (len(loader) - step) / max(rate, 1e-9)
            print(
                f"{split_name}: {step}/{len(loader)} minibatches; "
                f"loss={total_loss / max(total, 1):.6f}; eta_hours={remaining / 3600.0:.2f}",
                flush=True,
            )
    if dist.is_initialized():
        totals = torch.tensor([total_loss, float(total)], device=device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total = float(totals[0].item()), int(totals[1].item())
    return total_loss / max(total, 1)


def main():
    args = parse_args()
    config = read_json(args.config)
    rank, local_rank, world_size = distributed_setup()
    seed = int(config["seed"])
    seed_everything(seed)
    run_dir = Path(config["run_dir"])
    partition = read_json(run_dir / "02_partition" / "partition.json")
    validate_partition_groups(config, partition)
    split_manifest = read_json(run_dir / "00_splits" / "manifest.json")
    assignment_manifest = read_json(run_dir / "04_sample_groups" / "manifest.json")
    output_dir = run_dir / "05_experts"
    checkpoint_dir = output_dir / "checkpoints"
    committed_path = checkpoint_dir / "committed.json"
    committed = read_json(committed_path) if committed_path.exists() else None
    checkpoint_path = checkpoint_dir / committed["directory"] / f"rank_{rank}.pt" if committed else checkpoint_dir / f"rank_{rank}.pt"
    has_checkpoint = checkpoint_path.exists()
    existing_outputs = (
        [path for path in output_dir.iterdir() if path.name != "checkpoints"]
        if output_dir.exists()
        else []
    )
    if existing_outputs and not args.overwrite and not has_checkpoint:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    blocks = partition["blocks"]
    num_blocks = len(blocks)
    train_df = load_split_csv(assignment_manifest["paths"]["train"], require_groups=True, num_blocks=num_blocks)
    val_df = load_split_csv(assignment_manifest["paths"]["validation"], require_groups=True, num_blocks=num_blocks)
    tokenizer = load_tokenizer(config["model_name_or_path"])
    train_ds = make_dataset(train_df, tokenizer, int(config["max_length"]), extra_columns=[f"block_{b}_group" for b in range(num_blocks)] + [f"block_{b}_confidence" for b in range(num_blocks)] + [f"block_{b}_probabilities" for b in range(num_blocks)])
    val_ds = make_dataset(val_df, tokenizer, int(config["max_length"]), extra_columns=[f"block_{b}_group" for b in range(num_blocks)] + [f"block_{b}_confidence" for b in range(num_blocks)] + [f"block_{b}_probabilities" for b in range(num_blocks)])
    collator = RouteCollator(tokenizer, num_blocks)
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = ExactDistributedSampler(train_ds, rank, world_size, shuffle=True, seed=seed)
        val_sampler = ExactDistributedSampler(val_ds, rank, world_size, shuffle=False, seed=seed)
    train_loader = DataLoader(
        MaskedPaddingDataset(train_ds),
        batch_size=int(config["expert_train_batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        MaskedPaddingDataset(val_ds),
        batch_size=int(config.get("expert_validation_batch_size", config["expert_train_batch_size"])),
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collator,
    )
    effective_batch = int(config.get("expert_effective_batch_size", int(config["expert_train_batch_size"]) * int(config["expert_gradient_accumulation_steps"])))
    train_loader.gradient_accumulation_steps = local_accumulation(effective_batch, int(config["expert_train_batch_size"]), world_size)

    no_4bit = bool(config.get("no_4bit", False))
    base = load_base_classifier(
        config["model_name_or_path"],
        num_labels(config),
        tokenizer,
        no_4bit=no_4bit,
        device_map={"": local_rank} if world_size > 1 else "auto",
        compute_dtype=resolve_model_compute_dtype(config.get("expert_compute_dtype", "bfloat16")),
    )
    if not no_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    base.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad = False
    for name, parameter in base.named_parameters():
        if any(fragment in name for fragment in ("score.", "classifier.", "classification_head.")):
            parameter.requires_grad = True
    model, replaced = replace_with_blockwise_adapters(
        base,
        blocks,
        parse_str_list(config["target_modules"]),
        config["num_groups"],
        int(config["r_shared"]),
        config["r_private"],
        int(config["expert_lora_alpha"]),
        float(config["expert_lora_dropout"]),
    )
    if config.get("expert_gradient_checkpointing", False):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = input_device(model)
    trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    non_fp32 = [(name, str(parameter.dtype)) for name, parameter in trainable_named if parameter.dtype != torch.float32]
    if non_fp32:
        raise RuntimeError(f"Expert trainable parameters must be FP32; found {non_fp32[:10]}")
    trainable = [parameter for _, parameter in trainable_named]
    if is_main_process():
        print(
            f"Expert trainable parameters: {sum(parameter.numel() for parameter in trainable):,}; "
            f"dtype=float32; data_parallel_world_size={world_size}; "
            f"local_batch={int(config['expert_train_batch_size'])}; "
            f"local_accumulation={train_loader.gradient_accumulation_steps}; "
            f"effective_batch={int(config['expert_train_batch_size']) * world_size * train_loader.gradient_accumulation_steps}",
            flush=True,
        )
    training_model = model
    if world_size > 1:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        # Keep identical initialized parameters, then decorrelate dropout RNGs.
        seed_everything(seed + rank)
    optimizer = torch.optim.AdamW(trainable, lr=float(config["expert_learning_rate"]), weight_decay=0.0)
    checkpoint_steps = max(1, int(config.get("expert_checkpoint_steps", 1000)))
    resume_epoch = 0
    resume_step = 0
    if has_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("world_size") != world_size:
            raise RuntimeError(
                f"Checkpoint world size {checkpoint.get('world_size')} does not match current {world_size}"
            )
        if checkpoint.get("num_batches") != len(train_loader):
            raise RuntimeError("Checkpoint batch count does not match the current training loader")
        load_trainable_state_dict(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint["rng_state"], device)
        resume_epoch = int(checkpoint["epoch"])
        resume_step = int(checkpoint["next_step"])
        train_loader.resume_totals = checkpoint.get("running_totals", (0.0, 0))
        if is_main_process():
            print(
                f"Resuming Expert training from epoch {resume_epoch + 1}, "
                f"minibatch {resume_step}/{len(train_loader)}",
                flush=True,
            )
    best_loss = float("inf")
    best_state = None
    history = []
    epochs = max(1, int(math.ceil(float(config["expert_epochs"]))))
    for epoch in range(resume_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        def checkpoint_callback(step):
            if step % checkpoint_steps != 0 and step != len(train_loader):
                return
            directory = f"epoch_{epoch}_step_{step}"
            step_dir = checkpoint_dir / directory
            step_dir.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(
                step_dir / f"rank_{rank}.pt",
                model,
                optimizer,
                epoch,
                step,
                train_loader,
                device,
            )
            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])
            if is_main_process():
                previous = read_json(committed_path)["directory"] if committed_path.exists() else None
                write_json(committed_path, {"directory": directory, "epoch": epoch, "step": step, "world_size": world_size})
                for old in checkpoint_dir.iterdir():
                    if old.is_dir() and old.name.startswith("epoch_") and old.name not in {directory, previous}:
                        shutil.rmtree(old)
                print(f"Saved Expert checkpoint at epoch {epoch + 1}, minibatch {step}", flush=True)

        train_loss = train_or_eval(
            training_model,
            train_loader,
            device,
            num_blocks,
            optimizer,
            split_name=f"train epoch {epoch + 1}",
            log_steps=int(config.get("expert_log_steps", 250)),
            start_step=resume_step if epoch == resume_epoch else 0,
            checkpoint_callback=checkpoint_callback,
        )
        resume_step = 0
        train_loader.resume_totals = (0.0, 0)
        val_loss = train_or_eval(
            training_model,
            val_loader,
            device,
            num_blocks,
            optimizer=None,
            split_name=f"validation epoch {epoch + 1}",
            log_steps=int(config.get("expert_log_steps", 250)),
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss}
        history.append(row)
        if is_main_process():
            print(row, flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if any(fragment in key for fragment in ("shared_A", "shared_B", "private_A", "private_B", ".score.", ".classifier.", ".classification_head."))}
    if best_state is None:
        raise RuntimeError("Expert training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=False)
    if is_main_process():
        model.save_adapters(output_dir)
        tokenizer.save_pretrained(output_dir)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        write_json(
            output_dir / "run_meta.json",
            {
                "format": "adaptive_blockwise_experts_v1",
                "model_name_or_path": config["model_name_or_path"],
                "num_labels": num_labels(config),
                "blocks": blocks,
                "num_groups": config["num_groups"],
                "r_shared": int(config["r_shared"]),
                "r_private": config["r_private"],
                "alpha": int(config["expert_lora_alpha"]),
                "dropout": float(config["expert_lora_dropout"]),
                "target_modules": parse_str_list(config["target_modules"]),
                "replaced_modules": replaced,
                "train_identity_sha256": split_manifest["dataframe_sha256"]["train"],
                "validation_identity_sha256": split_manifest["dataframe_sha256"]["validation"],
                "route_supervision": "batch_level_stage03_spectral_group_copied_to_samples",
                "best_validation_loss": best_loss,
                "data_parallel_world_size": world_size,
                "max_length": int(config["max_length"]),
                "local_train_batch_size": int(train_loader.batch_size),
                "local_validation_batch_size": int(val_loader.batch_size),
                "gradient_checkpointing": bool(config.get("expert_gradient_checkpointing", False)),
                "effective_batch_size": int(config["expert_train_batch_size"])
                * world_size
                * train_loader.gradient_accumulation_steps,
                "trainable_dtype": "float32",
                "model_compute_dtype": str(config.get("expert_compute_dtype", "bfloat16")),
            },
        )
        print(f"Wrote trained blockwise experts to {output_dir}")
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
