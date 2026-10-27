"""Frozen expert execution with one route shared by a complete logical batch."""
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import DataCollatorWithPadding
from adaptive_blockwise.io_utils import read_json, sha256_file, seed_everything
from adaptive_blockwise.modeling import replace_with_blockwise_adapters, adapter_state_dict
from adaptive_blockwise.signals import load_base_classifier, load_tokenizer

class EntryCaptured(Exception):
    pass

class FrozenExperts:
    def __init__(self, config):
        self.config = config
        seed_everything(config['seed'])
        self.root = Path(config['run_dir'])
        path = self.root/'05_experts/blockwise_adapters.pt'
        self.sha256 = sha256_file(path)
        self.tokenizer = load_tokenizer(config['model_name_or_path'])
        self.collator = DataCollatorWithPadding(self.tokenizer)
        self.micro = int(config.get('compute_microbatch',4))
        self.num_labels = pd.read_csv(config['id2label_csv']).label.nunique()
        base = load_base_classifier(config['model_name_or_path'], self.num_labels, self.tokenizer,
            no_4bit=config['no_4bit'], compute_dtype=torch.bfloat16)
        base.config.use_cache = False
        blocks = read_json(self.root/'02_partition/partition.json')['blocks']
        self.model,_ = replace_with_blockwise_adapters(base,blocks,config['target_modules'],
            config['num_groups'],config['r_shared'],config['r_private'],config['expert_lora_alpha'],config['expert_lora_dropout'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected = adapter_state_dict(self.model)
        if set(state) != set(expected):
            raise ValueError('Incomplete expert/classification-head checkpoint')
        self.model.load_state_dict(state, strict=False)
        self.model.requires_grad_(False).eval()
        self.device = next(self.model.parameters()).device
        self.hidden_size = int(base.config.hidden_size)

    def encode(self, frame):
        enc = self.tokenizer(frame.text.astype(str).tolist(), truncation=True,max_length=self.config['max_length'])
        return [{k:v[i] for k,v in enc.items()} for i in range(len(frame))]

    def _inputs(self, encoded):
        # Deployment does not accept labels, oracle or group assignments here.
        return {k:v.to(self.device) for k,v in self.collator(encoded).items()}

    def _set_route(self, route, n):
        if len(route)!=4 or route[0]!=0 or any(a not in range(self.config['num_groups'][str(b)]) for b,a in enumerate(route)):
            raise ValueError(f'Invalid route {route}')
        self.model.set_teacher_routes({b:torch.full((n,),int(a),device=self.device,dtype=torch.long) for b,a in enumerate(route)})

    @torch.no_grad()
    def logits(self, encoded, route):
        rows=[]
        try:
            for start in range(0,len(encoded),self.micro):
                inputs=self._inputs(encoded[start:start+self.micro]);self._set_route(route,len(inputs['input_ids']))
                with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    rows.append(self.model(**inputs).logits.float().cpu())
        finally:self.model.clear_routes()
        return torch.cat(rows)

    @torch.no_grad()
    def entry(self, encoded, block, prefix):
        if len(prefix)!=block or prefix[0]!=0 or block not in (1,2,3):
            raise ValueError('Entry must use the executed upstream prefix')
        rows=[];mask=None
        def capture(_module,args,kwargs):
            h=args[0] if args else kwargs['hidden_states']
            m=mask.to(h.device).float().unsqueeze(-1)
            rows.append(((h.float()*m).sum(1)/m.sum(1).clamp_min(1)).cpu())
            raise EntryCaptured()
        handle=self.model.decoder_layers[block*10].register_forward_pre_hook(capture,with_kwargs=True)
        try:
            for start in range(0,len(encoded),self.micro):
                inputs=self._inputs(encoded[start:start+self.micro]);mask=inputs['attention_mask']
                self._set_route([*prefix,*([0]*(4-len(prefix)))],len(inputs['input_ids']))
                try:
                    with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                        self.model(**inputs)
                except EntryCaptured:pass
                else:raise RuntimeError('Block-entry hook did not run')
        finally:
            handle.remove();self.model.clear_routes()
        return torch.cat(rows)

    def learned_route(self, encoded, bank):
        prefix=[0];probs=[]
        for b in (1,2,3):
            z=self.entry(encoded,b,prefix)
            with torch.no_grad():p=bank.logits(b,[z]).float().softmax(-1)[0].cpu()
            prefix.append(int(p.argmax()));probs.append(p.tolist())
        return prefix,probs
