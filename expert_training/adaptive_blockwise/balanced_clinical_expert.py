"""Text-only inference for a balanced clinical section LoRA expert."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from peft import LoraConfig,TaskType,get_peft_model
from .io_utils import read_json,sha256_file
from .balanced_clinical_sections import batch_balanced_sections


class BalancedClinicalExpert:
    def __init__(self,root,checkpoint,device='cuda:0'):
        root=Path(root)
        checkpoint=Path(checkpoint)
        with checkpoint.open('rb') as stream:
            digest=hashlib.sha256()
            while chunk:=stream.read(1024*1024):digest.update(chunk)
            checkpoint_digest=digest.hexdigest()
            stream.seek(0)
            saved=torch.load(stream,map_location='cpu',weights_only=False)
        provenance=saved['provenance']
        assert provenance['fit_split']=='train' and not provenance['uses_test_data']
        assert provenance['source_sha256']==sha256_file(root/'scripts/184_train_balanced_clinical_lora.py')
        token_root=Path(provenance['config']['tokens'])
        assert provenance['token_provenance_sha256']==sha256_file(token_root/'provenance.json')
        tokens=read_json(token_root/'provenance.json')
        assert tokens['text_transform']=='balanced_problem_sections_v1'
        assert tokens['transform_sha256']==sha256_file(root/'adaptive_blockwise/balanced_clinical_sections.py')
        assert tokens['anchor_source_sha256']==sha256_file(root/'adaptive_blockwise/clinical_summary.py')
        assert provenance['input_transform_provenance']==tokens
        self.prefix_budget=tokens['prefix_budget']
        path=Path(tokens['model_path'])
        assert tokens['model_revision']==provenance['model_revision']
        assert provenance['model_sha256']==sha256_file(path/'model.safetensors')
        assert tokens['tokenizer_sha256']==sha256_file(path/'tokenizer.json')
        self.tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True)
        self.max_length=provenance['config']['max_length']
        self.tokenized_length=tokens['max_length']
        self.device=torch.device(device)
        raw=AutoModelForSequenceClassification.from_pretrained(path,num_labels=738,local_files_only=True,
                        dtype=torch.bfloat16,attn_implementation=provenance['config']['attention'])
        self.model=get_peft_model(raw,LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=.05,
                        target_modules=['Wqkv','Wo','Wi'],modules_to_save=['head','classifier']))
        self.model.to(device)
        parameters={name:p for name,p in self.model.named_parameters() if p.requires_grad}
        assert set(parameters)==set(saved['state'])==set(provenance['trained_parameters'])
        with torch.no_grad():
            for name,p in parameters.items():
                p.data=p.data.float()
                p.copy_(saved['state'][name].to(device))
        self.model.requires_grad_(False).eval()
        self.provenance=provenance
        self.selection=saved.get('selected')
        self.step=saved.get('step',self.selection.get('step') if self.selection else None)
        self.artifact_hashes={str(checkpoint):checkpoint_digest,str(path/'model.safetensors'):provenance['model_sha256'],
                             str(path/'tokenizer.json'):tokens['tokenizer_sha256'],
                             str(token_root/'provenance.json'):provenance['token_provenance_sha256']}

    def tokenize(self,texts):
        result=batch_balanced_sections(self.tokenizer,texts,max_length=self.max_length,
                                       prefix_budget=self.prefix_budget,return_tensors='pt')
        ids=result['input_ids'][:,:self.max_length]
        attention=result['attention_mask'][:,:self.max_length]
        length=int(attention.sum(1).max())
        return {'input_ids':ids[:,:length].to(self.device),'attention_mask':attention[:,:length].to(self.device)}

    @torch.inference_mode()
    def scores(self,texts,batch_size=16):
        chunks=[]
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            for start in range(0,len(texts),batch_size):
                logits=self.model(**self.tokenize(texts[start:start+batch_size])).logits.float()
                assert torch.isfinite(logits).all()
                chunks.append(logits.cpu().numpy())
        return np.concatenate(chunks)

    @torch.inference_mode()
    def fuse(self,texts,prior,batch_size=16):
        assert self.selection is not None and self.selection['weight']>0
        assert prior.shape==(len(texts),738)
        logits=torch.tensor(self.scores(texts,batch_size))
        probabilities=(logits/self.selection['temperature']).softmax(-1).numpy()
        weight=self.selection['weight']
        return (1-weight)*prior+weight*probabilities
