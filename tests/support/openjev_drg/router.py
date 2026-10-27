"""One frozen OpenJev decoder, three independent adapters and set bridges."""
from pathlib import Path
import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from adaptive_blockwise.io_utils import atomic_torch_save, sha256_file

class SetBridge(nn.Module):
    def __init__(self, hidden, jev_hidden, width=128, prefix=4):
        super().__init__();self.prefix=prefix;self.jev_hidden=jev_hidden
        self.norm=nn.LayerNorm(hidden);self.member=nn.Linear(hidden,width)
        self.bridge=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,prefix*jev_hidden))
        self.decision=nn.Parameter(torch.empty(1,jev_hidden));nn.init.normal_(self.decision,std=0.02)
        self.head=nn.Linear(jev_hidden,3)

    def embeddings(self, z):
        if z.ndim!=2 or len(z)==0:raise ValueError('Expected nonempty logical-batch member states')
        u=torch.nn.functional.gelu(self.member(self.norm(z.float())))
        mean=u.mean(0);std=((u-mean).square().mean(0)+1e-8).sqrt()
        prefix=self.bridge(torch.cat([mean,std])).reshape(self.prefix,self.jev_hidden)
        return torch.cat([prefix,self.decision],dim=0)

class RouterBank(nn.Module):
    def __init__(self, decoder, hidden, config):
        super().__init__();self.config=config
        def lora():return LoraConfig(r=config['router_lora_rank'],lora_alpha=config['router_lora_alpha'],
            lora_dropout=config['router_lora_dropout'],target_modules=['q_proj','v_proj'],bias='none')
        self.decoder=get_peft_model(decoder,lora(),adapter_name='block_1')
        for b in (2,3):self.decoder.add_adapter(f'block_{b}',lora())
        d=int(decoder.config.hidden_size);device=next(decoder.parameters()).device
        self.features=nn.ModuleDict({str(b):SetBridge(hidden,d,config['router_set_width'],config['router_prefix_length']) for b in (1,2,3)}).to(device)
        for name,p in self.decoder.named_parameters():
            if 'lora_' in name:p.data=p.data.float()
        self.requires_grad_(False);self.eval()

    @classmethod
    def from_config(cls,config,hidden,device='cuda:0'):
        checkpoint=Path(config['openjev_checkpoint'])
        meta=checkpoint.with_suffix('.manifest.json')
        if not meta.exists():raise FileNotFoundError('RLCD checkpoint provenance manifest is required')
        from adaptive_blockwise.io_utils import read_json
        m=read_json(meta)
        assets=read_json(checkpoint.parent/'assets.json')
        if (m['checkpoint_sha256']!=sha256_file(checkpoint) or m['phase']!='O2'
                or m['steps']!=config['openjev_o2_steps'] or m['author_commit']!=config['openjev_commit']
                or assets['model_revision']!=config['openjev_revision']):
            raise ValueError('OpenJev RLCD checkpoint hash/protocol mismatch')
        lm=AutoModelForCausalLM.from_pretrained(config['openjev_local_model'],dtype=torch.bfloat16,attn_implementation='sdpa')
        lm.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        decoder=lm.model.to(device);del lm
        return cls(decoder,hidden,config)

    def activate(self,block,training=False):
        self.decoder.set_adapter(f'block_{block}')
        self.requires_grad_(False)
        if training:
            self.features[str(block)].requires_grad_(True)
            for name,p in self.decoder.named_parameters():
                if 'lora_' in name and f'.block_{block}.' in name:p.requires_grad_(True)
        self.train(training)

    def logits(self,block,states):
        if self.decoder.active_adapter!=f'block_{block}':
            self.activate(block,training=self.training)
        f=self.features[str(block)];device=f.decision.device
        embeds=torch.stack([f.embeddings(z.to(device)) for z in states])
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            h=self.decoder(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],device=device,dtype=torch.long),use_cache=False).last_hidden_state[:,-1]
            return f.head(h.float()).float()

    def adapted_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if n.startswith('features.') or 'lora_' in n}

    def save(self,path,meta=None):
        atomic_torch_save({'state':self.adapted_state(),'meta':meta or {}},path)

    def load(self,path):
        payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['state']
        if set(state)!=set(self.adapted_state()):raise ValueError('Incomplete router checkpoint')
        self.load_state_dict(state,strict=False);self.requires_grad_(False);self.eval()
        return payload['meta']
