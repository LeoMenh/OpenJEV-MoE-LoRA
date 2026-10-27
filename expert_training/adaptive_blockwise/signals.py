import math
import os
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding

from .io_utils import TARGET_MODULES, atomic_torch_save, read_json, seed_everything, write_json


def initialize_signal_distributed():
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(days=7), device_id=torch.device("cuda", local_rank))
    return dist.get_rank() if world > 1 else 0, world, local_rank


def resolve_model_compute_dtype(value):
    normalized = str(value).lower()
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float16", "fp16"):
        return torch.float16
    if normalized in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported model compute dtype: {value}")


def quantization_config(no_4bit, compute_dtype=torch.float16):
    if no_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_classifier(
    model_name_or_path,
    num_labels,
    tokenizer,
    no_4bit=False,
    device_map="auto",
    compute_dtype=None,
):
    compute_dtype = compute_dtype or torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        quantization_config=quantization_config(no_4bit, compute_dtype),
        device_map=device_map,
        torch_dtype=torch.float32 if no_4bit else compute_dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def make_dataset(frame, tokenizer, max_length, extra_columns=()):
    columns = ["text", "label"] + [name for name in extra_columns if name in frame.columns]
    dataset = Dataset.from_pandas(frame[columns], preserve_index=False)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        desc="tokenize",
    )
    dataset = dataset.remove_columns(["text"]).rename_column("label", "labels")
    return dataset


def infer_layer_id(module_name):
    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\."):
        match = re.search(pattern, module_name)
        if match:
            return int(match.group(1))
    return -1


def first_adapter(module_dict):
    key = "default" if "default" in module_dict else next(iter(module_dict.keys()))
    return module_dict[key]


def lora_a(module):
    return first_adapter(module.lora_A).weight


def lora_b(module):
    return first_adapter(module.lora_B).weight


def selected_lora_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and infer_layer_id(name) >= 0
    }


def task_head_state(model):
    fragments = (".score.", ".classifier.", ".classification_head.")
    return {key: value.detach().cpu() for key, value in model.state_dict().items() if any(x in key for x in fragments)}


class ExactSignalComputer:
    """Fixed-A, zero-B implementation of Sections 3 and 4."""

    def __init__(
        self,
        model_name_or_path,
        num_labels,
        tokenizer,
        coordinate_path,
        seed,
        rank=8,
        alpha=8,
        ridge=1e-3,
        target_modules=TARGET_MODULES,
        no_4bit=False,
        device_map="auto",
    ):
        self.ridge = float(ridge)
        self.coordinate_path = Path(coordinate_path)
        seed_everything(seed)
        base = load_base_classifier(
            model_name_or_path,
            num_labels,
            tokenizer,
            no_4bit=no_4bit,
            device_map=device_map,
        )
        if not no_4bit:
            base = prepare_model_for_kbit_training(base)
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=list(target_modules),
        )
        self.model = get_peft_model(base, config)
        self.selected = selected_lora_modules(self.model)
        if not self.selected:
            raise RuntimeError("No LoRA modules were found after PEFT injection")
        self.layer_modules = {}
        for name in self.selected:
            self.layer_modules.setdefault(infer_layer_id(name), []).append(name)
        self.layer_modules = {layer: sorted(names) for layer, names in sorted(self.layer_modules.items())}
        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        if distributed and dist.get_rank() != 0:
            dist.barrier()
        self._load_or_create_coordinate(model_name_or_path, num_labels, rank, alpha, target_modules, seed)
        if distributed and dist.get_rank() == 0:
            dist.barrier()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for module in self.selected.values():
            lora_a(module).requires_grad = False
            lora_b(module).requires_grad = True
            lora_b(module).data.zero_()
        self.active_names = set(self.selected)
        self.mask = None
        self.covariances = {}
        self.counts = {}
        self.hooks = [module.register_forward_hook(self._metric_hook(name)) for name, module in self.selected.items()]

    def _load_or_create_coordinate(self, model_name, num_labels, rank, alpha, target_modules, seed):
        if self.coordinate_path.exists():
            payload = torch.load(self.coordinate_path, map_location="cpu")
            expected = {
                "model_name_or_path": str(model_name),
                "num_labels": int(num_labels),
                "rank": int(rank),
                "alpha": int(alpha),
                "target_modules": list(target_modules),
                "seed": int(seed),
            }
            if payload["config"] != expected:
                raise ValueError("Probe coordinate config differs from the existing coordinate file")
            current = self.model.state_dict()
            unexpected = sorted(set(payload["state"]) - set(current))
            if unexpected:
                raise ValueError(f"Probe coordinate has unexpected keys: {unexpected[:5]}")
            self.model.load_state_dict(payload["state"], strict=False)
            return
        state = {f"{name}.lora_A.default.weight": lora_a(module).detach().cpu() for name, module in self.selected.items()}
        # PEFT prefixes vary across versions, so resolve the actual state keys by tensor identity/name suffix.
        actual_state = self.model.state_dict()
        state = {
            key: value.detach().cpu()
            for key, value in actual_state.items()
            if ".lora_A." in key or any(fragment in key for fragment in (".score.", ".classifier.", ".classification_head."))
        }
        payload = {
            "format": "fixed_A_zero_B_probe_coordinate_v1",
            "config": {
                "model_name_or_path": str(model_name),
                "num_labels": int(num_labels),
                "rank": int(rank),
                "alpha": int(alpha),
                "target_modules": list(target_modules),
                "seed": int(seed),
            },
            "state": state,
        }
        atomic_torch_save(payload, self.coordinate_path)

    def set_active_layers(self, layers):
        missing = sorted(set(layers) - set(self.layer_modules))
        if missing:
            raise ValueError(f"Backbone is missing requested layers {missing}")
        self.active_names = {name for layer in layers for name in self.layer_modules[layer]}
        for name, module in self.selected.items():
            lora_b(module).requires_grad = name in self.active_names

    def _metric_hook(self, name):
        def hook(module, inputs, _output):
            if self.mask is None or name not in self.active_names:
                return
            hidden = inputs[0].detach().float()
            projected = hidden @ lora_a(module).detach().float().transpose(0, 1)
            covariance = self.covariances[name]
            # device_map="auto" places different decoder layers on different
            # GPUs. Keep each metric matrix on the device of its hook input.
            if covariance.device != projected.device:
                covariance = covariance.to(projected.device)
                self.covariances[name] = covariance
                if torch.is_tensor(self.counts[name]):
                    self.counts[name] = self.counts[name].to(projected.device)
            if covariance.ndim == 3 and projected.ndim == 3 and self.mask is not None:
                mask = self.mask.to(projected.device, dtype=torch.float32)
                if mask.shape[:2] != projected.shape[:2]:
                    raise ValueError("Metric mask and hidden-state shapes do not match")
                covariance.add_(torch.einsum("btr,bts,bt->brs", projected, projected, mask))
                self.counts[name].add_(mask.sum(dim=1))
            else:
                if projected.ndim == 3 and self.mask is not None and projected.shape[:2] == self.mask.shape:
                    weights = self.mask.to(projected.device, dtype=torch.float32).reshape(-1)
                else:
                    weights = torch.ones(projected.numel() // projected.shape[-1], device=projected.device)
                flat = projected.reshape(-1, projected.shape[-1])
                covariance.add_(flat.transpose(0, 1) @ (flat * weights[:, None]))
                self.counts[name] += float(weights.sum().item())

        return hook

    def _device_batch(self, batch):
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in batch.items()}

    def compute_on_batch_raw(self, batch):
        batch = self._device_batch(batch)
        self.model.zero_grad(set_to_none=True)
        self.covariances = {}
        self.counts = {}
        for name in self.active_names:
            rank = lora_a(self.selected[name]).shape[0]
            self.covariances[name] = torch.zeros(
                rank,
                rank,
                device=lora_a(self.selected[name]).device,
                dtype=torch.float32,
            )
            self.counts[name] = 0.0
        self.model.train()
        # M depends on the same zero-B module inputs as g. Capture it during
        # this forward; detach in the hook so metric accumulation adds no graph.
        self.mask = batch.get("attention_mask")
        output = self.model(**batch)
        self.mask = None
        if output.loss is None:
            raise ValueError("Gradient signatures require labels")
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite classification loss while extracting a signature")
        output.loss.backward()
        gradients = {}
        for name in self.active_names:
            gradient = lora_b(self.selected[name]).grad
            if gradient is not None and not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite signature gradient: {name}")
            gradients[name] = (
                torch.zeros_like(lora_b(self.selected[name]), dtype=torch.float32)
                if gradient is None
                else gradient.detach().float().clone()
            )
        covariances = {name: self.covariances[name].detach().float().clone() for name in self.active_names}
        counts = {name: float(self.counts[name]) for name in self.active_names}
        return gradients, covariances, counts, float(output.loss.detach().float().item())

    def compute_on_batch(self, batch):
        gradients, covariances, counts, loss = self.compute_on_batch_raw(batch)
        metrics = {}
        inverses = {}
        for name in sorted(self.active_names):
            covariance = covariances[name] / max(counts[name], 1.0)
            covariance = covariance + self.ridge * torch.eye(covariance.shape[0], device=covariance.device)
            metrics[name] = covariance.cpu()
            inverses[name] = torch.linalg.inv(covariance).cpu()
            gradients[name] = gradients[name].cpu()
        return gradients, metrics, inverses, loss

    def compute_per_sample_on_batch(self, batch):
        """Compute exact per-example g_i and M_i from one padded batch.

        The forward is shared across the examples, while the CE gradient is
        retained separately for every example. ``is_grads_batched`` is used
        when supported by the installed PyTorch; the loop fallback traverses
        the same graph and is mathematically identical.
        """
        batch = self._device_batch(batch)
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Per-sample signatures require labels")
        batch_size = int(labels.shape[0])
        self.model.zero_grad(set_to_none=True)
        self.covariances = {}
        self.counts = {}
        for name in self.active_names:
            rank = lora_a(self.selected[name]).shape[0]
            self.covariances[name] = torch.zeros(
                batch_size,
                rank,
                rank,
                device=lora_a(self.selected[name]).device,
                dtype=torch.float32,
            )
            self.counts[name] = torch.zeros(
                batch_size,
                device=lora_a(self.selected[name]).device,
                dtype=torch.float32,
            )
        self.model.train()
        self.mask = batch.get("attention_mask")
        output = self.model(**batch)
        self.mask = None
        logits = output.logits
        losses = torch.nn.functional.cross_entropy(logits, labels.to(logits.device), reduction="none")
        names = list(self.active_names)
        params = [lora_b(self.selected[name]) for name in names]
        gradients = None
        # Batched VJP avoids repeating the large model traversal. Some older
        # autograd/CUDA combinations do not implement every operation; retain
        # a reliable exact fallback for those environments.
        try:
            identity = torch.eye(batch_size, device=losses.device, dtype=losses.dtype)
            values = torch.autograd.grad(
                losses,
                params,
                grad_outputs=identity,
                is_grads_batched=True,
                retain_graph=True,
                allow_unused=True,
            )
            gradients = {
                name: torch.zeros((batch_size,) + tuple(param.shape), dtype=torch.float32)
                if value is None
                else value.detach().float().cpu().clone()
                for name, param, value in zip(names, params, values)
            }
        except (RuntimeError, torch.OutOfMemoryError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gradients = {name: [] for name in names}
            for index in range(batch_size):
                values = torch.autograd.grad(
                    losses[index],
                    params,
                    retain_graph=index + 1 < batch_size,
                    allow_unused=True,
                )
                for name, value, param in zip(names, values, params):
                    gradients[name].append(
                        torch.zeros_like(param, dtype=torch.float32).cpu()
                        if value is None
                        else value.detach().float().cpu().clone()
                    )
            gradients = {name: torch.stack(values, dim=0) for name, values in gradients.items()}

        metrics = {}
        inverses = {}
        for name in names:
            counts = self.counts[name].to(self.covariances[name].device).clamp_min(1.0)
            covariance = self.covariances[name] / counts[:, None, None]
            covariance = covariance + self.ridge * torch.eye(
                covariance.shape[-1], device=covariance.device
            )[None]
            metrics[name] = covariance.detach().float().cpu()
            inverses[name] = torch.linalg.inv(covariance).detach().float().cpu()
        return gradients, metrics, inverses, losses.detach().float().cpu()

    def pooled_hidden_on_batch(self, batch):
        batch = self._device_batch({key: value for key, value in batch.items() if key != "labels"})
        with torch.no_grad():
            output = self.model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].float().unsqueeze(-1)
        pooled = []
        for hidden in output.hidden_states:
            local_mask = mask.to(hidden.device)
            value = (hidden.float() * local_mask).sum(1) / local_mask.sum(1).clamp_min(1.0)
            pooled.append(value.detach().cpu())
        return torch.stack(pooled, dim=1)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_distributed_on_batch(computer, global_batch):
    """Mean CE gradient and token-weighted metric of one global signature batch."""
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    device = next(computer.model.parameters()).device
    count = int(global_batch["labels"].shape[0])
    lo, hi = count * rank // world, count * (rank + 1) // world
    local_count = hi - lo
    if local_count:
        batch = {key: value[lo:hi] for key, value in global_batch.items()}
        gradients, covariances, counts, loss = computer.compute_on_batch_raw(batch)
    else:
        gradients, covariances, counts = {}, {}, {}
        loss = 0.0
        for name in sorted(computer.active_names):
            parameter = lora_b(computer.selected[name])
            gradients[name] = torch.zeros_like(parameter, dtype=torch.float32)
            covariances[name] = torch.zeros((parameter.shape[1], parameter.shape[1]), device=parameter.device)
            counts[name] = 0.0
    for name in sorted(computer.active_names):
        gradients[name].mul_(float(local_count) / float(count))
        dist.all_reduce(gradients[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(covariances[name], op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor([counts[name]], device=covariances[name].device, dtype=torch.float32)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts[name] = float(count_tensor.item())
    loss_tensor = torch.tensor([loss * local_count / count], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    if rank != 0:
        return None, None, None, float(loss_tensor.item())
    metrics, inverses = {}, {}
    for name in sorted(computer.active_names):
        covariance = covariances[name] / max(counts[name], 1.0)
        covariance = covariance + computer.ridge * torch.eye(covariance.shape[0], device=covariance.device)
        metrics[name] = covariance.cpu()
        inverses[name] = torch.linalg.inv(covariance).cpu()
    return {name: gradient.cpu() for name, gradient in gradients.items()}, metrics, inverses, float(loss_tensor.item())


def _stack_records(records, dtype):
    names = records[0][0].keys()
    return {
        "grads": {name: torch.stack([record[0][name] for record in records]).to(dtype) for name in names},
        "metrics": {name: torch.stack([record[1][name] for record in records]).float() for name in names},
        "metric_inverses": {name: torch.stack([record[2][name] for record in records]).float() for name in names},
    }


def extract_signal_cache(computer, dataset, tokenizer, layers, output_dir, batch_size=1, cache_batches=16, storage_dtype=torch.float16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    computer.set_active_layers(layers)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    part_records = []
    manifest_path = output_dir / "manifest.json"
    contract = {
        "layers": [int(layer) for layer in layers],
        "batch_size": int(batch_size),
        "num_batches": len(loader),
        "num_examples": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "storage_dtype": str(storage_dtype),
    }
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError(f"Signal cache contract changed: {output_dir}")
        part_records = previous["parts"]
        seen = [batch_id for part in part_records for batch_id in part["batch_ids"]]
        if seen != list(range(len(seen))) or any(not (output_dir / part["path"]).exists() for part in part_records):
            raise ValueError("Signal checkpoint parts are missing or not a contiguous prefix")
        print(f"Resuming signals {output_dir.name}: {len(seen)}/{len(loader)} batches", flush=True)
    records = []
    ids = []
    next_id = sum(part["num_batches"] for part in part_records)
    completed_batches = next_id
    for step, batch in enumerate(loader, start=1):
        if step <= completed_batches:
            continue
        gradients, metrics, inverses, loss = computer.compute_on_batch(batch)
        records.append((gradients, metrics, inverses))
        ids.append(next_id)
        next_id += 1
        if len(records) >= cache_batches or step == len(loader):
            payload = _stack_records(records, storage_dtype)
            payload["batch_ids"] = ids
            part_path = output_dir / f"part_{len(part_records):05d}.pt"
            atomic_torch_save(payload, part_path)
            part_records.append({"path": part_path.name, "batch_ids": ids, "num_batches": len(ids)})
            records, ids = [], []
            write_json(manifest_path, {
                "format": "exact_fixed_A_signal_cache_v1", **contract,
                "layer_modules": {str(layer): computer.layer_modules[layer] for layer in layers},
                "parts": part_records, "complete": step == len(loader),
            })
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"signals {output_dir.name}: {step}/{len(loader)} batches; loss={loss:.6f}", flush=True)
    manifest = {
        "format": "exact_fixed_A_signal_cache_v1",
        "layers": [int(layer) for layer in layers],
        "layer_modules": {str(layer): computer.layer_modules[layer] for layer in layers},
        "batch_size": int(batch_size),
        "num_batches": int(len(loader)),
        "num_examples": int(len(dataset)),
        "parts": part_records,
        "storage_dtype": str(storage_dtype),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "complete": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def extract_signal_cache_distributed(
    computer,
    dataset,
    tokenizer,
    layers,
    output_dir,
    batch_size=16,
    cache_batches=16,
    storage_dtype=torch.float16,
):
    """Extract exact signatures for global batches across one process per GPU.

    Each rank computes a disjoint local slice of a global batch. Gradients are
    sample-weighted and covariance/count statistics are summed before the
    rank-zero process writes the same cache format as the serial extractor.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed signature extraction requires an initialized process group")
    rank = dist.get_rank()
    world = dist.get_world_size()
    batch_size = int(batch_size)
    if batch_size < world or batch_size % world:
        raise ValueError("Global signature batch must be divisible by the number of GPUs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    computer.set_active_layers(layers)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_batches = math.ceil(len(dataset) / batch_size)
    contract = {
        "layers": [int(layer) for layer in layers],
        "batch_size": batch_size,
        "num_batches": num_batches,
        "num_examples": len(dataset),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "storage_dtype": str(storage_dtype),
        "world_size": world,
    }
    part_records = []
    completed = 0
    manifest_path = output_dir / "manifest.json"
    if rank == 0 and manifest_path.exists():
        previous = read_json(manifest_path)
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError(f"Distributed signal cache contract changed: {output_dir}")
        part_records = previous["parts"]
        completed = sum(int(part["num_batches"]) for part in part_records)
        seen = [batch_id for part in part_records for batch_id in part["batch_ids"]]
        if (seen != list(range(completed)) or completed > num_batches
                or any(not (output_dir / part["path"]).exists() for part in part_records)):
            raise ValueError("Distributed signal checkpoint parts are missing or not a contiguous prefix")
    completed_tensor = torch.tensor([completed], device=next(computer.model.parameters()).device, dtype=torch.long)
    dist.broadcast(completed_tensor, src=0)
    completed = int(completed_tensor.item())
    if rank != 0:
        part_records = []
    records = []
    ids = []
    for batch_id in range(completed, num_batches):
        start = batch_id * batch_size
        end = min(start + batch_size, len(dataset))
        # Pad every rank to the global batch's length, matching serial B=16.
        global_batch = collator([dataset[index] for index in range(start, end)])
        gradients, metrics, inverses, loss = compute_distributed_on_batch(computer, global_batch)
        if rank == 0:
            records.append((gradients, metrics, inverses))
            ids.append(batch_id)
        dist.barrier()
        if rank == 0 and (len(records) >= cache_batches or batch_id == num_batches - 1):
            payload = _stack_records(records, storage_dtype)
            payload["batch_ids"] = ids
            part_path = output_dir / f"part_{len(part_records):05d}.pt"
            atomic_torch_save(payload, part_path)
            part_records.append({"path": part_path.name, "batch_ids": ids, "num_batches": len(ids)})
            records, ids = [], []
            write_json(manifest_path, {
                "format": "exact_fixed_A_signal_cache_v1",
                **contract,
                "layer_modules": {str(layer): computer.layer_modules[layer] for layer in layers},
                "parts": part_records,
                "complete": batch_id == num_batches - 1,
            })
            print(f"distributed signals {output_dir.name}: {batch_id + 1}/{num_batches} global batches; loss={loss:.6f}", flush=True)
    dist.barrier()
    if rank == 0:
        write_json(manifest_path, {
            "format": "exact_fixed_A_signal_cache_v1",
            **contract,
            "layer_modules": {str(layer): computer.layer_modules[layer] for layer in layers},
            "parts": part_records,
            "complete": True,
        })
    dist.barrier()
    return read_json(manifest_path) if rank == 0 else None


def load_signal_part(cache_dir, record, module_names, device="cpu"):
    payload = torch.load(Path(cache_dir) / record["path"], map_location="cpu")
    work = {"batch_ids": [int(value) for value in payload["batch_ids"]]}
    for family in ("grads", "metrics", "metric_inverses"):
        work[family] = {name: payload[family][name].float().to(device) for name in module_names}
    return work


def group_part_records(records, max_batches):
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    groups = []
    current = []
    current_batches = 0
    for record in records:
        record_batches = int(record["num_batches"])
        if current and current_batches + record_batches > max_batches:
            groups.append(current)
            current = []
            current_batches = 0
        current.append(record)
        current_batches += record_batches
    if current:
        groups.append(current)
    return groups


def load_signal_group(cache_dir, records, module_names, device):
    parts = [load_signal_part(cache_dir, record, module_names, device="cpu") for record in records]
    grouped = {"batch_ids": [batch_id for part in parts for batch_id in part["batch_ids"]]}
    for family in ("grads", "metrics", "metric_inverses"):
        grouped[family] = {
            name: torch.cat([part[family][name] for part in parts], dim=0).to(device, non_blocking=True)
            for name in module_names
        }
    grouped["grad_grams"] = {
        name: torch.einsum("bor,bot->brt", grouped["grads"][name], grouped["grads"][name])
        for name in module_names
    }
    grouped["kernels"] = {
        name: torch.matmul(
            torch.matmul(grouped["metric_inverses"][name], grouped["metrics"][name]),
            grouped["metric_inverses"][name].transpose(1, 2),
        )
        for name in module_names
    }
    return grouped


def score_signal_groups(source, query, module_names, device):
    scores = torch.zeros(
        (len(source["batch_ids"]), len(query["batch_ids"])),
        dtype=torch.float32,
        device=device,
    )
    for name in module_names:
        weighted_query = torch.einsum(
            "qor,qrs->qos", query["grads"][name], query["kernels"][name]
        )
        cross = torch.einsum("sor,qor->sq", source["grads"][name], weighted_query)
        energy = torch.einsum(
            "srt,qrt->sq", source["grad_grams"][name], query["kernels"][name]
        )
        scores.add_(2.0 * cross - energy)
    return scores


def resolve_compute_device(value="auto"):
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for compatibility computation but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_directional_matrix(
    source_cache,
    query_cache,
    layer,
    output_path,
    device="auto",
    score_block_batches=256,
    preload=True,
):
    source_cache = Path(source_cache)
    query_cache = Path(query_cache)
    source_manifest = read_json(source_cache / "manifest.json")
    query_manifest = read_json(query_cache / "manifest.json")
    module_names = source_manifest["layer_modules"][str(layer)]
    if query_manifest["layer_modules"][str(layer)] != module_names:
        raise ValueError("Source/query signal module layouts differ")
    target_device = resolve_compute_device(device)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(source_manifest["num_batches"], query_manifest["num_batches"]),
    )
    source_groups = group_part_records(source_manifest["parts"], int(score_block_batches))
    query_groups = group_part_records(query_manifest["parts"], int(score_block_batches))
    same_cache = source_cache.resolve() == query_cache.resolve()
    loaded_source = None
    loaded_query = None
    if preload:
        try:
            print(
                f"compat layer {layer}: preloading {len(source_groups)} source and "
                f"{len(query_groups)} query scoring blocks to {target_device}",
                flush=True,
            )
            loaded_source = [
                load_signal_group(source_cache, records, module_names, target_device)
                for records in source_groups
            ]
            loaded_query = loaded_source if same_cache else [
                load_signal_group(query_cache, records, module_names, target_device)
                for records in query_groups
            ]
        except torch.OutOfMemoryError:
            loaded_source = None
            loaded_query = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"compat layer {layer}: GPU preload did not fit; falling back to exact streamed scoring",
                flush=True,
            )
    for query_index, query_records in enumerate(query_groups):
        query = (
            loaded_query[query_index]
            if loaded_query is not None
            else load_signal_group(query_cache, query_records, module_names, target_device)
        )
        for source_index, source_records in enumerate(source_groups):
            source = (
                loaded_source[source_index]
                if loaded_source is not None
                else load_signal_group(source_cache, source_records, module_names, target_device)
            )
            scores = score_signal_groups(source, query, module_names, target_device)
            output[np.ix_(source["batch_ids"], query["batch_ids"])] = scores.detach().cpu().numpy()
            del scores
            if loaded_source is None:
                del source
        output.flush()
        print(f"compat layer {layer}: query block {query_index + 1}/{len(query_groups)}", flush=True)
        if loaded_query is None:
            del query
    return output


def symmetrize_and_normalize_memmap(directional_path, normalized_path, chunk_rows=256):
    directional = np.load(directional_path, mmap_mode="r")
    if directional.shape[0] != directional.shape[1]:
        raise ValueError("Symmetrization requires a square directional matrix")
    count = directional.shape[0]
    total = 0.0
    total_square = 0.0
    observations = 0
    for start in range(0, count, chunk_rows):
        end = min(start + chunk_rows, count)
        block = 0.5 * (np.asarray(directional[start:end, :]) + np.asarray(directional[:, start:end]).T)
        mask = np.ones(block.shape, dtype=bool)
        mask[np.arange(end - start), np.arange(start, end)] = False
        values = block[mask].astype(np.float64)
        total += float(values.sum())
        total_square += float(np.square(values).sum())
        observations += int(values.size)
    mean = total / observations
    variance = max(total_square / observations - mean * mean, 0.0)
    std = math.sqrt(variance)
    normalized = np.lib.format.open_memmap(normalized_path, mode="w+", dtype=np.float32, shape=directional.shape)
    for start in range(0, count, chunk_rows):
        end = min(start + chunk_rows, count)
        block = 0.5 * (np.asarray(directional[start:end, :]) + np.asarray(directional[:, start:end]).T)
        block = ((block - mean) / (std + 1e-8)).astype(np.float32)
        block[np.arange(end - start), np.arange(start, end)] = 0.0
        normalized[start:end] = block
    normalized.flush()
    return normalized, mean, std
