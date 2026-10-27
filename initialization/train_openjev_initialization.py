#!/usr/bin/env python3
"""Execute the pinned author's unmodified trainer on locally pinned assets."""
import argparse
import sys
from pathlib import Path
from adaptive_blockwise.io_utils import read_json, write_json, sha256_file
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config_openjev.json');p.add_argument('--phase',choices=['O1','O2'],required=True);a=p.parse_args();c=read_json(a.config)
    assets_root=ROOT/'runs/A_OPENJEV/initialization'
    out=Path(c['openjev_checkpoint']).parent;out.mkdir(parents=True,exist_ok=True)
    assets=read_json(assets_root/'assets.json');splits=read_json(assets_root/'mmlu_splits.json')
    write_json(out/'assets.json',assets)
    if assets['model_revision']!=c['openjev_revision'] or assets['author_commit']!=c['openjev_commit']:raise ValueError('Initialization provenance mismatch')
    sys.path.insert(0,str(ROOT/'vendor/openjev-rlcd'))
    import openjev_rlcd.train as author
    author.TASKS['mmlupro']['load']=lambda:splits
    checkpoint=out/f'{a.phase}.pt';steps=c[f'openjev_{a.phase.lower()}_steps'];seed=17 if a.phase=='O1' else 1017
    argv=['openjev_rlcd.train','--task','mmlupro','--model',c['openjev_local_model'],'--objective','rlcd',
        '--steps',str(steps),'--seed',str(seed),'--m','4','--m-eval','4','--items-per-step','8',
        '--max-new-tokens',str(c.get('openjev_max_new_tokens',256)),'--lr','2e-6','--diversity-weight','0',
        '--dev-cap',str(c.get('openjev_dev_cap',300)),'--test-cap',str(c.get('openjev_test_cap',1000)),
        '--eval-every','400','--save-checkpoint',str(checkpoint),'--output',str(out/f'{a.phase}_author_log.json')]
    if a.phase=='O1':argv+=['--no-score-function']
    else:
        prior=read_json(out/'O1.manifest.json')
        if prior['checkpoint_sha256']!=sha256_file(out/'O1.pt'):raise ValueError('O1 checkpoint changed')
        argv+=['--init-checkpoint',str(out/'O1.pt'),'--sf-coef','0.3','--kl-beta','0.04']
    write_json(out/f'{a.phase}_arguments.json',{'argv':argv,'assets':assets})
    sys.argv=argv;author.main()
    write_json(checkpoint.with_suffix('.manifest.json'),{'phase':a.phase,'steps':steps,'seed':seed,
        'checkpoint_sha256':sha256_file(checkpoint),'author_commit':c['openjev_commit'],
        'assets_sha256':sha256_file(out/'assets.json'),'arguments':argv,
        'o1_sha256':sha256_file(out/'O1.pt') if a.phase=='O2' else None})

if __name__=='__main__':main()
