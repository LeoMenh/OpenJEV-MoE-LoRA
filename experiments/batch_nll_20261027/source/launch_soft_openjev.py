#!/usr/bin/env python3
"""Snapshot source artifacts and supervise the authorized soft OpenJev run."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import time

JEV=Path(__file__).resolve().parents[1]
BACKBONE=JEV.parent/'drg-llama/new_layerwise'
SOURCE=BACKBONE/'runs/k1233_qwen1024_train16_eval8_sig16_8gpu_20261022'
DEFAULT_RUN=JEV/'runs/A_OPENJEV/oracle_task_1233_1024_20261025_v3'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(root):
    import torch
    if read(SOURCE/'06_joint/progress.json')['status']!='run_complete':
        raise RuntimeError('Source backbone is not complete')
    provenance=read(SOURCE/'06_joint/provenance.json');source=provenance['source_config']
    if source['num_groups']!={'0':1,'1':2,'2':3,'3':3} or source['max_length']!=1024:
        raise ValueError('Wrong source expert pool')
    groups=read(SOURCE/'03_full_groups/manifest.json');assignment=read(SOURCE/'04_sample_groups/manifest.json')
    if groups['signature_batch_size']!=16 or groups['num_groups']!=source['num_groups']:
        raise ValueError('Signature contract mismatch')
    for split,path in assignment['paths'].items():
        if not Path(path).is_file():raise FileNotFoundError(path)
    o2=JEV/'runs/A_OPENJEV/initialization/O2.pt';manifest=read(o2.with_suffix('.manifest.json'))
    if manifest['phase']!='O2' or manifest['checkpoint_sha256']!=sha(o2):raise ValueError('O2 hash mismatch')
    root.mkdir(parents=True,exist_ok=True)
    snapshot=root/'source';snapshot.mkdir(exist_ok=True)
    shutil.copytree(BACKBONE/'adaptive_blockwise',snapshot/'backbone/adaptive_blockwise',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ('soft_openjev_runtime.py','train_soft_openjev.py','audit_soft_openjev.py'):
        shutil.copy2(JEV/'scripts'/name,snapshot/name)
    shutil.copy2(JEV/'openjev_drg/oracle.py',snapshot/'prefix_cache.py')
    shutil.copy2(Path(__file__),snapshot/'launch_soft_openjev.py')
    reused=root/'reused';reused.mkdir(exist_ok=True)
    links={k:SOURCE/k for k in ['00_splits','01_calibration','02_partition','03_full_groups','04_sample_groups']}
    for name,target in links.items():
        link=reused/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        elif link.resolve()!=target.resolve():raise ValueError('Reuse link changed')
    chosen=torch.load(SOURCE/'06_joint/best.pt',map_location='cpu',weights_only=False)
    state=chosen['model'];expert={k.removeprefix('core.'):v for k,v in state.items() if k.startswith('core.')}
    mlp={k.removeprefix('routers.'):v for k,v in state.items() if k.startswith('routers.')}
    if len(state)!=len(expert)+len(mlp) or not expert or not mlp:raise ValueError('Unrecognized joint checkpoint')
    if any('router' in k for k in expert):raise ValueError('Router state leaked into expert export')
    torch.save(expert,reused/'expert_and_head.pt')
    (root/'baseline_only').mkdir(exist_ok=True);torch.save(mlp,root/'baseline_only/mlp.pt')
    c={'run_dir':str(root),'seed':20260718,'source':source,
       'expert_meta':read(SOURCE/'05_experts/run_meta.json'),
       'split_manifest':read(SOURCE/'00_splits/manifest.json'),
       'source_best_step':chosen['global_step'],'baseline_validation':chosen['validation'],
       'openjev_checkpoint':str(o2),'openjev_local_model':str(JEV/'models/Qwen3-1.7B'),
       'router_lora_rank':8,'router_lora_alpha':16,'router_lora_dropout':0.05,
       'router_width':128,'router_prefix_length':4,'router_lr':1e-4,'router_adapter_lr':1e-5,
       'oracle_temperature':1.0,'oracle_chunk_size':64,'oracle_microbatch':8,'oracle_epochs':3,
       'task_epochs':1,'task_local_batch':1,'task_effective_batch':16,'task_lr_scale':0.1,
       'task_checkpoint_steps':100,'task_validation_steps':2000,'eval_batch':8,
       'test_policy':'No test access; development selection only; report historical test reuse.',
       'training_supervision':'per-case independent route-cost CE, then true DRG CE; no old-router targets',
       'signature_usage':'reuse split/group/prototype artifacts and expert provenance; raw gradient signatures were deleted upstream by policy; prefix-dependent hidden features are recomputed',
       'seed_policy':'First full seed 20260718; three-seed confirmation is not automatically launched'}
    hashes={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*.py'))}
    contract={'source_best_sha256':sha(SOURCE/'06_joint/best.pt'),'exported_expert_sha256':sha(reused/'expert_and_head.pt'),
        'o2_sha256':sha(o2),'signature_manifest_sha256':sha(SOURCE/'03_full_groups/manifest.json'),
        'assignment_manifest_sha256':sha(SOURCE/'04_sample_groups/manifest.json'),
        'split_manifest_sha256':sha(SOURCE/'00_splits/manifest.json'),'source_hashes':hashes,
        'source_run':str(SOURCE),'selected_step':chosen['global_step']}
    c['contract_sha256']=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    write(root/'config.json',c);write(root/'reuse_contract.json',contract)
    (root/'README.md').write_text('# 逐病例 OpenJev 软路由独立训练\n\n源：'+str(SOURCE)+'\n\nK=1233，1024 tokens；复用 signature 分组/原型与选中专家/分类头。原始 gradient signature 已按上游策略清理，不声称复用不存在的 raw cache。\n\n流程：真实模型 preflight → 全训练集每病例 18 路 oracle → OpenJev O2 + 三个独立 bridge/qv-LoRA/head 训练 3 轮 → 冻结专家的 DRG soft-mixture 任务训练 1 轮 → 开发集选择与完整概率导出。\n\n旧 MLP 仅 B0 评测；不蒸馏其概率。首轮 seed=20260718。此入口不读测试集。查看 progress.json、logs/ 和 oracle_train/worker_*_progress.json。STOP 文件或 SIGTERM 可停止本流水线，不影响源实验。\n')
    return c


def environment(root,gpus):
    e=dict(os.environ);snapshot=root/'source'
    e.update(CUDA_VISIBLE_DEVICES=gpus,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
             PYTHONPATH=str(snapshot/'backbone')+os.pathsep+str(snapshot))
    packages=Path(sysconfig.get_paths()['purelib']);libraries=sorted((packages/'nvidia').glob('*/lib'))+[packages/'torch/lib']
    e['LD_LIBRARY_PATH']=os.pathsep.join(str(p) for p in libraries if p.is_dir())+os.pathsep+e.get('LD_LIBRARY_PATH','')
    nccl=packages/'nvidia/nccl/lib/libnccl.so.2'
    if nccl.exists():e['LD_PRELOAD']=str(nccl)+(':'+e['LD_PRELOAD'] if e.get('LD_PRELOAD') else '')
    return e


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--prepare',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=a.run_dir.resolve()
    if not root.is_relative_to(JEV/'runs/A_OPENJEV'):raise ValueError('Use isolated A_OPENJEV run directory')
    if a.prepare:
        if (root/'config.json').exists():raise ValueError('Prepared run already exists; do not overwrite')
        prepare(root);print('Prepared',root);return
    c=read(root/'config.json');gpus=a.gpus.split(',');(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        with (root/'logs/launcher.log').open('a') as f:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run-dir',str(root),'--gpus',a.gpus],
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launcher.json',{'pid':child.pid,'started_at':time.time()});print('Launched PID',child.pid);return
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[];stage='starting';completed=[]
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,interrupted)
        def progress(status='running',**extra):
            write(root/'progress.json',{'status':status,'stage':stage,'launcher_pid':os.getpid(),
                'completed':completed,'children':[p.pid for p,f in children], 'updated_at':time.time(),**extra})
        def execute(name,commands):
            nonlocal stage,children
            stage=name;marker=root/f'{name}.done.json'
            if marker.exists():completed.append(name);return
            if (root/'STOP').exists():raise KeyboardInterrupt('STOP file')
            children=[]
            try:
                for i,(command,visible) in enumerate(commands):
                    log=(root/f'logs/{name}_{i}.log').open('a',buffering=1)
                    child=subprocess.Popen(command,cwd=root/'source',env=environment(root,visible),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    children.append((child,log))
                while True:
                    codes=[p.poll() for p,f in children]
                    if (root/'STOP').exists():raise KeyboardInterrupt('STOP file')
                    if any(x not in (None,0) for x in codes):raise RuntimeError(f'{name} child failed: {codes}; see logs')
                    progress()
                    if all(x==0 for x in codes):break
                    time.sleep(5)
            finally:
                for child,log in children:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                    log.close()
                children=[]
            write(marker,{'finished_at':time.time()});completed.append(name)
        script=str(root/'source/train_soft_openjev.py');base=[sys.executable,script,'--config',str(root/'config.json')]
        try:
            contract=read(root/'reuse_contract.json')
            for relative,digest in contract['source_hashes'].items():
                if sha(root/'source'/relative)!=digest:raise ValueError('Snapshot changed: '+relative)
            if sha(root/'reused/expert_and_head.pt')!=contract['exported_expert_sha256']:raise ValueError('Expert export changed')
            execute('00_preflight',[(base+['--stage','preflight'],gpus[0])])
            if not read(root/'preflight.json')['passed']:raise RuntimeError('Preflight not passed')
            execute('01_oracle',[(base+['--stage','oracle','--rank',str(i),'--world',str(len(gpus))],gpu) for i,gpu in enumerate(gpus)])
            expected=math.ceil(c['split_manifest']['counts']['train']/c['oracle_chunk_size'])
            if len(list((root/'oracle_train').glob('chunk_*.pt')))!=expected:raise RuntimeError('Incomplete oracle')
            execute('02_warm',[(base+['--stage','warm'],gpus[0])])
            task=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',script,'--config',str(root/'config.json'),'--stage','task']
            execute('03_task',[(task,a.gpus)])
            execute('04_audit',[([sys.executable,str(root/'source/audit_soft_openjev.py'),'--run-dir',str(root)],'')])
            stage='complete';progress('complete',test_evaluated=False)
        except BaseException as exc:
            progress('stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise


if __name__=='__main__':main()
