#!/usr/bin/env python3
"""Pin model and MMLU-Pro assets before author O1/O2 training."""
import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from adaptive_blockwise.io_utils import read_json, write_json, sha256_file

ROOT = Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config_openjev.json');a=p.parse_args();c=read_json(a.config)
    model=Path(c['openjev_local_model']);model.mkdir(parents=True,exist_ok=True)
    snapshot_download(c['openjev_model'], revision=c['openjev_revision'], local_dir=model,
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'])
    revision=HfApi().dataset_info('TIGER-Lab/MMLU-Pro').sha
    data=ROOT/'datasets/MMLU-Pro'/revision
    snapshot_download('TIGER-Lab/MMLU-Pro',repo_type='dataset',revision=revision,local_dir=data,
        allow_patterns=['*.parquet','*.json','*.md'])
    from datasets import load_dataset
    import random
    ds=load_dataset('parquet',data_files={'test':[str(x) for x in sorted(data.rglob('*test*.parquet'))]})['test']
    items=[]
    for x in ds:
        y=int(x['answer_index']);q=[float(i==y) for i in range(len(x['options']))]
        items.append(dict(uid=str(x['question_id']),question=x['question'],options=list(x['options']),category=x['category'],q=q,counts=q))
    random.Random(0).shuffle(items)
    out=ROOT/'runs/A_OPENJEV/initialization';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'mmlu_splits.json', dict(train=items[:4000],dev=items[4000:4500],test=items[4500:6000],pool=items[6000:]))
    hashes={str(x.relative_to(model)):sha256_file(x) for x in model.glob('*') if x.is_file()}
    write_json(out/'assets.json',dict(author_commit=c['openjev_commit'],model_id=c['openjev_model'],model_revision=c['openjev_revision'],model_files_sha256=hashes,dataset_revision=revision,dataset_id='TIGER-Lab/MMLU-Pro',split_seed=0,split_sizes=[4000,500,1500],split_sha256=sha256_file(out/'mmlu_splits.json')))
    print('Pinned model, dataset and author-defined MMLU-Pro splits ready',flush=True)

if __name__=='__main__':main()
