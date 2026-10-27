"""Causal-LM teacher-forced DRG-code signature extraction."""
import re
from pathlib import Path
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .io_utils import TARGET_MODULES, atomic_torch_save, seed_everything, write_json
from .signals import infer_layer_id, lora_a, lora_b, selected_lora_modules

class GenerativeDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, label_to_code):
        self.rows = []
        targets = {int(k): tokenizer(str(v), add_special_tokens=False).input_ids for k, v in label_to_code.items()}
        max_target = max(len(x) for x in targets.values())
        for row in frame.itertuples(index=False):
            target = targets[int(row.label)]
            prompt_ids = tokenizer(str(row.text) + "\nDRG-34 code:", add_special_tokens=True, truncation=True, max_length=max(1, int(max_length) - max(len(target), 1))).input_ids
            ids = prompt_ids + target
            if len(ids) > int(max_length):
                ids = ids[-int(max_length):]
                target_start = max(0, len(ids) - len(target))
            else:
                target_start = len(prompt_ids)
            labels = [-100] * len(ids)
            labels[target_start:] = ids[target_start:]
            metric_mask = [0.0] * len(ids)
            for pos in range(target_start, len(ids)):
                metric_mask[max(0, pos - 1)] = 1.0
            self.rows.append({"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), dtype=torch.long), "labels": torch.tensor(labels), "metric_mask": torch.tensor(metric_mask), "target_positions": torch.tensor([p - 1 for p in range(target_start, len(ids))] + [-1] * (max_target - len(target)), dtype=torch.long), "target_ids": torch.tensor(ids[target_start:] + [0] * (max_target - len(target)), dtype=torch.long)})
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]

class GenerativeCollator:
    def __init__(self, tokenizer): self.pad_id = int(tokenizer.pad_token_id)
    def __call__(self, rows):
        width = max(len(x["input_ids"]) for x in rows)
        out = {}
        for key in ("input_ids", "attention_mask", "labels", "metric_mask"):

            vals = []
            for row in rows:
                value = row[key]; pad = width - len(value)
                fill = self.pad_id if key == "input_ids" else (-100 if key == "labels" else 0)
                vals.append(torch.cat([value, torch.full((pad,), fill, dtype=value.dtype)]))
            out[key] = torch.stack(vals)
        out["target_positions"] = torch.stack([row["target_positions"] for row in rows])
        out["target_ids"] = torch.stack([row["target_ids"] for row in rows])
        return out

def load_generative_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_label_to_code(path):
    frame = pd.read_csv(path, usecols=["drg_34_code", "label"])
    return {int(row.label): str(row.drg_34_code) for row in frame.itertuples(index=False)}

class GenerativeSignalComputer:
    def __init__(self, model_name_or_path, tokenizer, coordinate_path, seed, rank, alpha, ridge, target_modules, no_4bit=False):
        self.ridge = float(ridge); seed_everything(seed)
        quant = None if no_4bit else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quant, device_map="auto", torch_dtype=torch.float32 if no_4bit else torch.float16, trust_remote_code=True)
        self.model.config.use_cache = False
        if not no_4bit: self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(rank), lora_alpha=int(alpha), lora_dropout=0.0, bias="none", target_modules=list(target_modules)))
        self.selected = selected_lora_modules(self.model)
        self.layer_modules = {}
        for name in self.selected: self.layer_modules.setdefault(infer_layer_id(name), []).append(name)
        self.layer_modules = {layer: sorted(names) for layer, names in sorted(self.layer_modules.items())}
        if coordinate_path.exists(): self.model.load_state_dict(torch.load(coordinate_path, map_location="cpu")["state"], strict=False)
        else: atomic_torch_save({"format": "generative_fixed_A_zero_B_v1", "state": {k: v.detach().cpu() for k, v in self.model.state_dict().items() if ".lora_A." in k}}, coordinate_path)
        for parameter in self.model.parameters(): parameter.requires_grad = False
        for module in self.selected.values(): lora_a(module).requires_grad = False; lora_b(module).requires_grad = True; lora_b(module).data.zero_()
        self.active_names = set(self.selected); self.mask = None; self.covariances = {}; self.counts = {}
        self.hooks = [module.register_forward_hook(self._metric_hook(name)) for name, module in self.selected.items()]
    def set_active_layers(self, layers):
        self.active_names = {name for layer in layers for name in self.layer_modules[int(layer)]}
        for name, module in self.selected.items(): lora_b(module).requires_grad = name in self.active_names
    def _metric_hook(self, name):
        def hook(module, inputs, _output):
            if self.mask is None or name not in self.active_names: return
            hidden = inputs[0].detach().float(); projected = hidden @ lora_a(module).detach().float().transpose(0, 1)
            covariance = self.covariances[name]
            if covariance.device != projected.device: covariance = covariance.to(projected.device); self.covariances[name] = covariance
            mask = self.mask.to(projected.device, dtype=torch.float32); flat = projected.reshape(-1, projected.shape[-1]); weights = mask.reshape(-1)
            covariance.add_(flat.transpose(0, 1) @ (flat * weights[:, None])); self.counts[name] += weights.sum()
        return hook
    def compute_on_batch(self, batch, microbatch_size=4):
        device = next(self.model.parameters()).device
        metric_mask = batch.pop("metric_mask").to(device)
        target_positions = batch.pop("target_positions").to(device)
        target_ids = batch.pop("target_ids").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        self.model.zero_grad(set_to_none=True); self.covariances = {}; self.counts = {}
        for name in self.active_names:
            rank = lora_a(self.selected[name]).shape[0]; dev = lora_a(self.selected[name]).device
            self.covariances[name] = torch.zeros(rank, rank, device=dev); self.counts[name] = torch.tensor(0.0, device=dev)
        causal = self.model.base_model.model
        total_loss_value = 0.0
        total_count = int(batch["input_ids"].shape[0])
        for start in range(0, total_count, int(microbatch_size)):
            stop = min(start + int(microbatch_size), total_count)
            sub_input = batch["input_ids"][start:stop]
            sub_attention = batch["attention_mask"][start:stop]
            sub_positions = target_positions[start:stop]
            sub_ids = target_ids[start:stop]
            self.mask = metric_mask[start:stop]
            base_out = causal.model(input_ids=sub_input, attention_mask=sub_attention, use_cache=False, return_dict=True)
            hidden = base_out.last_hidden_state
            losses = []
            for row in range(hidden.shape[0]):
                valid = sub_positions[row] >= 0
                positions = sub_positions[row][valid].to(hidden.device)
                ids = sub_ids[row][valid].to(hidden.device)
                selected_hidden = hidden[row, positions]
                logits = causal.lm_head(selected_hidden)
                losses.append(torch.nn.functional.cross_entropy(logits.float(), ids, reduction="mean"))
            sub_loss = torch.stack(losses).mean()
            total_loss_value += float(sub_loss.detach().float().cpu().item()) * float(stop - start) / float(total_count)
            (sub_loss * float(stop - start) / float(total_count)).backward()
            del base_out, hidden, logits, losses, sub_loss
        self.mask = None
        loss = total_loss_value
        gradients = {}; metrics = {}; inverses = {}
        for name in self.active_names:
            parameter = lora_b(self.selected[name]); gradients[name] = torch.zeros_like(parameter, dtype=torch.float32).cpu() if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
            covariance = self.covariances[name] / self.counts[name].clamp_min(1.0); covariance = covariance + self.ridge * torch.eye(covariance.shape[-1], device=covariance.device)
            metrics[name] = covariance.detach().cpu(); inverses[name] = torch.linalg.inv(covariance).detach().cpu()
        return gradients, metrics, inverses, float(loss)

    def close(self):
        for hook in self.hooks: hook.remove()
        del self.model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def extract_generative_signal_cache(computer, dataset, tokenizer, layers, output_dir, batch_size=16, cache_batches=16, storage_dtype=torch.float16):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); computer.set_active_layers(layers)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=GenerativeCollator(tokenizer)); records = []; ids = []; parts = []
    for step, batch in enumerate(loader, start=1):
        result = computer.compute_on_batch(batch, microbatch_size=4); records.append(result[:3]); ids.append(step - 1)
        if len(records) >= int(cache_batches) or step == len(loader):
            names = records[0][0].keys(); payload = {"grads": {n: torch.stack([r[0][n] for r in records]).to(storage_dtype) for n in names}, "metrics": {n: torch.stack([r[1][n] for r in records]).float() for n in names}, "metric_inverses": {n: torch.stack([r[2][n] for r in records]).float() for n in names}, "batch_ids": ids}
            path = output_dir / f"part_{len(parts):05d}.pt"; atomic_torch_save(payload, path); parts.append({"path": path.name, "batch_ids": ids, "num_batches": len(ids)}); records = []; ids = []
        if step == 1 or step % 25 == 0 or step == len(loader): print(f"generative signals {output_dir.name}: {step}/{len(loader)} batches; loss={result[3]:.6f}", flush=True)
    manifest = {"format": "generative_causal_lm_signal_cache_v1", "layers": [int(x) for x in layers], "layer_modules": {str(x): computer.layer_modules[int(x)] for x in layers}, "batch_size": int(batch_size), "num_batches": len(loader), "num_examples": len(dataset), "parts": parts, "storage_dtype": str(storage_dtype)}; write_json(output_dir / "manifest.json", manifest); return manifest
