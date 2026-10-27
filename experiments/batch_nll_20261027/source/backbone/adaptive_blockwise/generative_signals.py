"""Causal-LM teacher-forced DRG-code signatures.

The target is the DRG-34 code appended after the note.  The loss is computed
only on target code tokens, so the resulting gradients describe generative
label prediction rather than a sequence-classification score head.
"""

import re
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig

from .io_utils import TARGET_MODULES, atomic_torch_save, seed_everything, write_json
from .signals import infer_layer_id, lora_a, lora_b, selected_lora_modules


class GenerativeDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, label_to_code):
        self.rows = []
        target_token_ids = {
            int(label): tokenizer(str(code), add_special_tokens=False).input_ids
            for label, code in label_to_code.items()
        }
        max_target = max(len(value) for value in target_token_ids.values())
        for row in frame.itertuples(index=False):
            target = target_token_ids[int(row.label)]
            prompt = str(row.text) + "\nDRG-34 code:"
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max(1, int(max_length) - max_target),
            ).input_ids
            # Keep the target intact and reserve one position before it: the
            # causal LM logit at position t predicts the token at t+1.
            input_ids = prompt_ids + target
            if len(input_ids) > int(max_length):
                input_ids = input_ids[-int(max_length):]
                target_start = max(0, len(input_ids) - len(target))
            else:
                target_start = len(prompt_ids)
            labels = [-100] * len(input_ids)
            labels[target_start:] = input_ids[target_start:]
            metric_mask = [0.0] * len(input_ids)
            # Causal shift: target label at position t uses hidden state t-1.
            for position in range(target_start, len(input_ids)):
                metric_mask[max(0, position - 1)] = 1.0
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "metric_mask": torch.tensor(metric_mask, dtype=torch.float32),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class GenerativeCollator:
    def __init__(self, tokenizer):
        self.pad_id = int(tokenizer.pad_token_id)

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        output = {}
        for key in ("input_ids", "attention_mask", "labels", "metric_mask"):
            values = []
            for row in rows:
                value = row[key]
                pad = width - len(value)
                fill = self.pad_id if key == "input_ids" else (-100 if key == "labels" else 0)
                values.append(torch.cat([value, torch.full((pad,), fill, dtype=value.dtype)]))
            output[key] = torch.stack(values)
        return output


def load_generative_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_label_to_code(path):
    mapping = pd.read_csv(path, usecols=["drg_34_code", "label"])
    return {int(row.label): str(row.drg_34_code) for row in mapping.itertuples(index=False)}


def _layer_id(name):
    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


class GenerativeSignalComputer:
    def __init__(self, model_name_or_path, tokenizer, coordinate_path, seed, rank, alpha, ridge, target_modules, no_4bit=False):
        self.ridge = float(ridge)
        seed_everything(seed)
        quant = None if no_4bit else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float32 if no_4bit else torch.float16,
            trust_remote_code=True,
        )
        self.model.config.use_cache = False
        if not no_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=list(target_modules),
        )
        self.model = get_peft_model(self.model, config)
        self.selected = selected_lora_modules(self.model)
        self.layer_modules = {}
        for name in self.selected:
            self.layer_modules.setdefault(_layer_id(name), []).append(name)
        self.layer_modules = {layer: sorted(names) for layer, names in sorted(self.layer_modules.items())}
        state = {key: value.detach().cpu() for key, value in self.model.state_dict().items() if ".lora_A." in key}
        if coordinate_path.exists():
            payload = torch.load(coordinate_path, map_location="cpu")
            self.model.load_state_dict(payload["state"], strict=False)
        else:
            atomic_torch_save({"format": "generative_fixed_A_zero_B_v1", "state": state}, coordinate_path)
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

    def set_active_layers(self, layers):
        self.active_names = {name for layer in layers for name in self.layer_modules[int(layer)]}
        for name, module in self.selected.items():
            lora_b(module).requires_grad = name in self.active_names

    def _metric_hook(self, name):
        def hook(module, inputs, _output):
            if self.mask is None or name not in self.active_names:
                return
            hidden = inputs[0].detach().float()
            projected = hidden @ lora_a(module).detach().float().transpose(0, 1)
            covariance = self.covariances[name]
            if covariance.device != projected.device:
                covariance = covariance.to(projected.device)
                self.covariances[name] = covariance
                self.counts[name] = self.counts[name].to(projected.device)
            mask = self.mask.to(projected.device, dtype=torch.float32)
            covariance.add_(torch.einsum("btr,bts,bt->brs", projected, projected, mask))
            self.counts[name].add_(mask.sum(dim=1))
        return hook

    def compute_on_batch(self, batch):
        device = next(self.model.parameters()).device
        metric_mask = batch.pop("metric_mask").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        self.model.zero_grad(set_to_none=True)
        self.covariances, self.counts = {}, {}
        for name in self.active_names:
            rank = lora_a(self.selected[name]).shape[0]
            self.covariances[name] = torch.zeros(rank, rank, device=lora_a(self.selected[name]).device, dtype=torch.float32)
            self.counts[name] = torch.zeros(metric_mask.shape[0], device=lora_a(self.selected[name]).device, dtype=torch.float32)
        self.mask = metric_mask
        output = self.model(**batch)
        self.mask = None
        if output.loss is None:
            raise ValueError("Generative signature loss is missing")
        output.loss.backward()
        gradients, metrics, inverses = {}, {}, {}
        for name in self.active_names:
            parameter = lora_b(self.selected[name])
            gradients[name] = torch.zeros_like(parameter, dtype=torch.float32).cpu() if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
            covariance = self.covariances[name] / max(float(metric_mask.shape[0]), 1.0)
            covariance = covariance + self.ridge * torch.eye(covariance.shape[-1], device=covariance.device)
            metrics[name] = covariance.detach().cpu()
            inverses[name] = torch.linalg.inv(covariance).detach().cpu()
        return gradients, metrics, inverses, float(output.loss.detach().float().item())

    def close(self):
        for hook in self.hooks:
            hook.remove()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_generative_signal_cache(computer, dataset, tokenizer, layers, output_dir, batch_size=16, cache_batches=16, storage_dtype=torch.float16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    computer.set_active_layers(layers)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=GenerativeCollator(tokenizer))
    records, ids, parts = [], [], []
    for step, batch in enumerate(loader, start=1):
        gradients, metrics, inverses, loss = computer.compute_on_batch(batch)
        records.append((gradients, metrics, inverses))
        ids.append(step - 1)
        if len(records) >= int(cache_batches) or step == len(loader):
            names = records[0][0].keys()
            payload = {
                "grads": {name: torch.stack([item[0][name] for item in records]).to(storage_dtype) for name in names},
                "metrics": {name: torch.stack([item[1][name] for item in records]).float() for name in names},
                "metric_inverses": {name: torch.stack([item[2][name] for item in records]).float() for name in names},
                "batch_ids": ids,
            }
            path = output_dir / f"part_{len(parts):05d}.pt"
            atomic_torch_save(payload, path)
            parts.append({"path": path.name, "batch_ids": ids, "num_batches": len(ids)})
            records, ids = [], []
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f"generative signals {output_dir.name}: {step}/{len(loader)} batches; loss={loss:.6f}", flush=True)
    manifest = {"format": "generative_causal_lm_signal_cache_v1", "layers": [int(x) for x in layers], "layer_modules": {str(x): computer.layer_modules[int(x)] for x in layers}, "batch_size": int(batch_size), "num_batches": len(loader), "num_examples": len(dataset), "parts": parts, "storage_dtype": str(storage_dtype)}
    write_json(output_dir / "manifest.json", manifest)
    return manifest
