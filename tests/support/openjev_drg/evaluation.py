import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from adaptive_blockwise.io_utils import write_json


def evaluate(experts,bank,frame,members,out,save_predictions=True,bootstrap=0,seed=20260718):
    out.mkdir(parents=True,exist_ok=True);bank.requires_grad_(False);bank.eval()
    n=len(frame);classes=experts.num_labels
    probabilities=np.lib.format.open_memmap(out/'probabilities.npy',mode='w+',dtype=np.float32,shape=(n,classes)) if save_predictions else None
    predicted=[];true=[];batch_stats=[];route_rows=[];bins=np.zeros((15,3),dtype=np.float64);started=time.perf_counter()
    for r in members['batches']:
        part=frame.iloc[r['start']:r['stop']];encoded=experts.encode(part)
        route,route_p=experts.learned_route(encoded,bank)
        logits=experts.logits(encoded,route);y=torch.tensor(part.label.to_numpy(),dtype=torch.long)
        p=logits.softmax(-1);pred=p.argmax(-1);confidence=p.max(-1).values
        nll=F.cross_entropy(logits,y,reduction='none')
        brier=p.square().sum(-1)-2*p[torch.arange(len(y)),y]+1
        top=logits.topk(min(10,classes),dim=-1).indices
        sums={'n':len(y),'nll_sum':float(nll.double().sum()),'brier_sum':float(brier.double().sum())}
        for k in (1,5,10):sums[f'acc{k}_sum']=int((top[:,:min(k,classes)]==y[:,None]).any(-1).sum())
        batch_stats.append(sums);predicted.extend(pred.tolist());true.extend(y.tolist())
        if probabilities is not None:probabilities[r['start']:r['stop']]=p.numpy()
        for ci,correct in zip(confidence.tolist(),(pred==y).tolist()):
            b=min(int(ci*15),14);bins[b]+=np.asarray([1,ci,float(correct)])
        route_rows.append({'batch_id':r['batch_id'],'members_sha256':r['sha256'],'n':r['n'],'route':route,'route_probabilities':route_p})
        if r['batch_id']%25==0:print(f'Validation routing: {r["batch_id"]+1}/{len(members["batches"])}',flush=True)
    if probabilities is not None:probabilities.flush()
    metrics={'n':n,'nll':sum(r['nll_sum'] for r in batch_stats)/n,'brier':sum(r['brier_sum'] for r in batch_stats)/n,
        'macro_f1':float(f1_score(true,predicted,labels=list(range(classes)),average='macro',zero_division=0)),
        'ece_15_equal_width':float(sum(abs(b[2]-b[1]) for b in bins)/n),
        'elapsed_seconds':time.perf_counter()-started,'members_sha256':members['sha256'],
        'split':'validation_new_test','independent_test_available':False}
    for k in (1,5,10):metrics[f'acc@{k}']=sum(r[f'acc{k}_sum'] for r in batch_stats)/n
    metrics['samples_per_second']=n/metrics['elapsed_seconds']
    metrics['route_usage']={str(b):np.bincount([r['route'][b] for r in route_rows],minlength=3).tolist() for b in (1,2,3)}
    if bootstrap:
        rng=np.random.default_rng(seed);stats=np.asarray([[r[k] for k in ('n','nll_sum','brier_sum','acc1_sum','acc5_sum','acc10_sum')] for r in batch_stats]);draws=[]
        for _ in range(bootstrap):
            total=stats[rng.integers(0,len(stats),len(stats))].sum(0);draws.append(total[1:]/total[0])
        bounds=np.quantile(draws,[0.025,0.975],axis=0)
        metrics['batch_bootstrap_95ci']={k:bounds[:,i].tolist() for i,k in enumerate(['nll','brier','acc@1','acc@5','acc@10'])}
        metrics['bootstrap_seed']=seed;metrics['bootstrap_repeats']=bootstrap
    write_json(out/'routes.json',route_rows);write_json(out/'batch_statistics.json',batch_stats)
    if save_predictions:
        import pandas as pd
        pd.DataFrame({'sample_id':frame.sample_id,'label':true,'prediction':predicted}).to_csv(out/'predictions.csv',index=False)
    write_json(out/'metrics.json',metrics)
    return metrics
