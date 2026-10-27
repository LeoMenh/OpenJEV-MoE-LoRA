#!/usr/bin/env python3
"""Supervise the matched, casewise-soft batch-NLL router experiment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from launch_soft_openjev import environment, read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--detach',action='store_true')
    a=p.parse_args();root=a.run_dir.resolve();gpus=a.gpus.split(',')
    if len(gpus)!=8 or len(set(gpus))!=8:raise ValueError('Require eight distinct GPUs')
    c=read(root/'config.json')
    if c['routing_granularity']!='casewise_soft_copied_batch_target':raise ValueError('Policy not confirmed')
    if a.detach:
        with (root/'logs/launcher.log').open('a') as f:
            proc=subprocess.Popen([sys.executable,str(root/'source/launch_batch_nll_routers.py'),
                '--run-dir',str(root),'--gpus',a.gpus],cwd=root/'source',env=environment(root,a.gpus),
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launcher.json',{'pid':proc.pid,'started_at':time.time()});print('Launched',proc.pid);return
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[];stage='starting';completed=[]
        def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupt)
        def progress(status='running',**extra):
            write(root/'progress.json',{'status':status,'stage':stage,'completed':completed,
                'launcher_pid':os.getpid(),'children':[child.pid for child,_ in children],
                'updated_at':time.time(),'test_evaluated':False,**extra})
        def execute(name,jobs):
            nonlocal stage,children
            stage=name;marker=root/f'{name}.done.json'
            if marker.exists():completed.append(name);return
            if (root/'STOP').exists():raise KeyboardInterrupt('STOP file')
            children=[]
            try:
                for index,(command,visible) in enumerate(jobs):
                    log=(root/f'logs/{name}_{index}.log').open('a',buffering=1)
                    child=subprocess.Popen(command,cwd=root/'source',env=environment(root,visible),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children.append((child,log))
                while True:
                    codes=[child.poll() for child,_ in children]
                    if any(code not in (None,0) for code in codes):raise RuntimeError(f'{name} failed: {codes}')
                    if (root/'STOP').exists():raise KeyboardInterrupt('STOP file')
                    progress()
                    if all(code==0 for code in codes):break
                    time.sleep(5)
            finally:
                for child,log in children:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                    log.close()
                children=[]
            write(marker,{'finished_at':time.time()});completed.append(name)
        script=root/'source/train_batch_nll_routers.py'
        base=[sys.executable,str(script),'--config',str(root/'config.json')]
        oracle_base=[sys.executable,str(root/'source/batch_nll_oracle.py'),'--config',str(root/'config.json')]
        def train_jobs(phase,block=None):
            return [(base+['--stage','train','--phase',phase,'--method',m]+(['--block',str(block)] if block else []),gpus[i])
                    for i,m in enumerate(('mlp','openjev'))]
        def eval_jobs(phase):
            return [(base+['--stage','evaluate','--phase',phase,'--method',m,'--rank',str(rank),'--world','4'],gpus[mi*4+rank])
                    for mi,m in enumerate(('mlp','openjev')) for rank in range(4)]
        try:
            contract=read(root/'experiment_contract.json')
            expected=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
            if expected!=c['experiment_contract']:raise ValueError('Configuration contract changed')
            if contract['configuration']!={k:v for k,v in c.items() if k!='experiment_contract'}:
                raise ValueError('Runtime hyperparameters differ from the recorded contract')
            for relative,digest in contract['source_hashes'].items():
                if sha(root/'source'/relative)!=digest:raise ValueError('Snapshot changed: '+relative)
            if sha(root/'reused/expert_and_head.pt')!=contract['expert_sha256']:raise ValueError('Expert changed')
            if sha(c['openjev_checkpoint'])!=contract['o2_sha256']:raise ValueError('O2 changed')
            if sha(root/'oracle/train.pt')!=contract['train_oracle_sha256']:raise ValueError('Train oracle changed')
            if sha(c['method_document'])!=contract['method_document_sha256']:raise ValueError('Methods changed')
            # The first live batch verification was performed during preparation.
            if not read(root/'oracle/verification.json')['passed']:raise ValueError('Cost verification failed')
            execute('00_preflight',[(base+['--stage','preflight'],gpus[0])])
            if not read(root/'preflight.json')['passed']:raise ValueError('Preflight failed')
            execute('01_optimal_features',[(base+['--stage','extract','--phase','optimal','--rank',str(i),'--world','8'],gpu)
                                         for i,gpu in enumerate(gpus)])
            # Training on GPUs 0/1 and the isolated validation oracle on GPUs 2..7.
            validation_jobs=[(oracle_base+['--stage','validation','--rank',str(i),'--world','6'],gpu)
                             for i,gpu in enumerate(gpus[2:])]
            execute('02_optimal_train_and_validation_oracle',train_jobs('optimal')+validation_jobs)
            execute('03_merge_validation_oracle',[([sys.executable,str(root/'source/audit_batch_nll_routers.py'),
                '--run-dir',str(root),'--merge-only'],'')])
            execute('04_evaluate_optimal',eval_jobs('optimal'))
            for block in (1,2,3):
                if block>1:
                    jobs=[(base+['--stage','extract','--phase','soft','--method',method,'--block',str(block),
                        '--rank',str(rank),'--world','4'],gpus[mi*4+rank])
                        for mi,method in enumerate(('mlp','openjev')) for rank in range(4)]
                    execute(f'05_soft_features_block_{block}',jobs)
                execute(f'06_soft_train_block_{block}',train_jobs('soft',block))
            execute('07_evaluate_soft',eval_jobs('soft'))
            execute('08_audit',[([sys.executable,str(root/'source/audit_batch_nll_routers.py'),'--run-dir',str(root)],'')])
            stage='complete';progress('complete')
        except BaseException as exc:
            progress('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise


if __name__=='__main__':main()
