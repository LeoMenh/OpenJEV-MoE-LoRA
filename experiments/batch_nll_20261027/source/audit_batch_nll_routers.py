#!/usr/bin/env python3
"""Merge isolated validation oracles and report matched batch-target routers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'backbone'))

import numpy as np
import torch
from sklearn.metrics import f1_score

from adaptive_blockwise.io_utils import write_json
from train_batch_nll_routers import frame_for, sha


def merge_oracle(root, c):
    frame = frame_for(c, 'validation'); n = len(frame); costs = torch.empty(n, 18); cursor = 0
    files = sorted((root/'oracle/validation').glob('chunk_*.pt'))
    for path in files:
        d = torch.load(path, map_location='cpu', weights_only=True)
        assert d['start'] == cursor and d['source_contract'] == c['contract_sha256']
        end = d['stop']; assert d['sample_ids'] == frame.iloc[cursor:end].sample_id.astype(str).tolist()
        costs[cursor:end] = d['case_costs']; cursor = end
    assert cursor == n and torch.isfinite(costs).all()
    routes = torch.tensor([[0,a,b,d] for a in range(2) for b in range(3) for d in range(3)])
    starts = torch.arange(0,n,8); counts = torch.minimum(torch.full_like(starts,8),n-starts)
    means = torch.stack([costs[i:i+8].double().mean(0) for i in starts.tolist()])
    best = means.argmin(1)
    value = {'case_costs':costs,'costs':means,'routes':routes,'starts':starts,'counts':counts,
             'best_route_indices':best,'best_routes':routes[best],'batch_size':8,
             'validation_identity':c['split_manifest']['dataframe_sha256']['validation']}
    torch.save(value,root/'oracle/validation.pt')
    frame[['sample_id']].assign(position=np.arange(n),logical_batch=np.arange(n)//8).to_csv(
        root/'oracle/validation_members.csv',index=False)
    write_json(root/'oracle/validation_manifest.json',{'status':'complete','cases':n,
        'logical_batches':len(starts),'last_batch_cases':int(counts[-1]),'batch_size':8,
        'oracle_sha256':sha(root/'oracle/validation.pt'),'evaluation_only':True,
        'router_training_uses_this_oracle':False,'finished_at':time.time()})


def load_predictions(path,n):
    rows=[]
    for p in sorted(path.glob('rank_*.npz')):
        with np.load(p,allow_pickle=False) as d: rows.append({k:d[k] for k in d.files})
    positions=np.concatenate([d['positions'] for d in rows]); order=positions.argsort()
    if not np.array_equal(positions[order],np.arange(n)): raise ValueError('Prediction cover mismatch')
    return {k:np.concatenate([d[k] for d in rows])[order] for k in rows[0]}


def statistics(pred,labels,oracle):
    scores=torch.from_numpy(pred['scores']); target=torch.tensor(labels)
    top=scores.topk(10,dim=-1).indices
    result={'n':len(labels),'acc1':float((top[:,0]==target).float().mean()),
            'acc5':float((top[:,:5]==target[:,None]).any(-1).double().mean()),
            'acc10':float((top==target[:,None]).any(-1).double().mean()),
            'nll':float(torch.nn.functional.cross_entropy(scores,target,reduction='none').double().mean()),
            'correct':int((top[:,0]==target).sum()),
            'macro_f1':float(f1_score(labels,top[:,0].numpy(),average='macro',zero_division=0))}
    # Calculate Top1 from integers to preserve exact matching on checkpoint selection.
    result['acc1']=result['correct']/len(labels)
    targets=oracle['best_routes'].repeat_interleave(8,dim=0)[:len(labels)].numpy()
    chosen=np.zeros((len(labels),4),dtype=np.int64); routers={}
    for b in (1,2,3):
        p=pred[f'block_{b}']; k=p.shape[-1]
        if not (np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=2e-5)):
            raise ValueError('Invalid router probabilities')
        a=p.argmax(-1); chosen[:,b]=a
        entropy=-(p*np.log(np.clip(p,1e-30,1))).sum(-1)
        routers[str(b)]={'oracle_accuracy':float((a==targets[:,b]).mean()),
            'oracle_macro_f1':float(f1_score(targets[:,b],a,labels=list(range(k)),average='macro',zero_division=0)),
            'oracle_label_counts':np.bincount(targets[:,b],minlength=k).tolist(),
            'argmax_counts':np.bincount(a,minlength=k).tolist(),
            'mean_probability':p.mean(0).tolist(),'std_probability':p.std(0).tolist(),
            'mean_entropy':float(entropy.mean()),'normalized_entropy':float(entropy.mean()/np.log(k)),
            'mean_max_probability':float(p.max(-1).mean()),
            'target_definition':'component of globally lowest mean-NLL hard route for the fixed validation batch8; copied to each case',
            'compatibility_labels_used':False}
    result['full_route_oracle_accuracy']=float((chosen==targets).all(-1).mean())
    result['router']=routers
    return result


def audit(root,c):
    frame=frame_for(c,'validation');labels=frame.label.to_numpy();n=len(frame)
    oracle=torch.load(root/'oracle/validation.pt',map_location='cpu',weights_only=True)
    selected={}; predictions={};all_results={}
    for method in ('mlp','openjev'):
        candidates=[];all_results[method]={}
        for phase in ('optimal','soft'):
            directory=root/'evaluation'/method/phase
            p=load_predictions(directory,n);metrics=statistics(p,labels,oracle)
            metrics.update(method=method,phase=phase,test_evaluated=False)
            write_json(directory/'metrics.json',metrics);all_results[method][phase]=metrics
            candidates.append((metrics['nll'],phase,p,metrics))
        # Explicitly follow the supplied methods' NLL selection criterion.
        chosen=min(candidates,key=lambda x:x[0]);_,phase,p,metrics=chosen
        shutil.copy2(root/method/f'{phase}.pt',root/method/'selected.pt')
        selected[method]=metrics;predictions[method]=p
        write_json(root/method/'selected.json',{'phase':phase,'selection_metric':'validation_NLL',
            'checkpoint_sha256':sha(root/method/'selected.pt'),'metrics':metrics})
    a=predictions['mlp']['scores'].argmax(-1)==labels
    b=predictions['openjev']['scores'].argmax(-1)==labels
    delta=b.astype(np.int8)-a.astype(np.int8)
    starts=np.arange(0,n,8);counts=np.minimum(8,n-starts)
    batch_delta=np.asarray([delta[i:i+8].sum() for i in starts]);rng=np.random.default_rng(c['seed'])
    bootstrap=[]
    for _ in range(2000):
        ix=rng.integers(len(starts),size=len(starts));bootstrap.append(batch_delta[ix].sum()/counts[ix].sum()*100)
    comparison={'selected':selected,'all_phases':all_results,
        'openjev_minus_mlp_percentage_points':float(delta.mean()*100),
        'paired_batch8_bootstrap_95ci_percentage_points':np.quantile(bootstrap,[.025,.975]).tolist(),
        'mlp_correct_openjev_wrong':int((a&~b).sum()),'mlp_wrong_openjev_correct':int((~a&b).sum()),
        'train_batch':16,'validation_batch':8,'casewise_soft_execution':True,
        'supervision':'copied globally minimum batch-NLL hard route; CE',
        'compatibility_labels_used':False,'test_evaluated':False,'seed':c['seed'],
        'seed_count':1,'completed_at':time.time(),
        'soft_prefix_phase':'upstream soft mixtures; fixed global batch oracle labels. Not hard conditional R-B.'}
    write_json(root/'comparison.json',comparison)
    write_json(root/'router_statistics.json',{m:v['router'] for m,v in selected.items()})
    output=frame[['sample_id']].copy();output['true_label']=labels
    for m,p in predictions.items():
        output[m+'_prediction']=p['scores'].argmax(-1)
        for block in (1,2,3):output[f'{m}_block_{block}']=p[f'block_{block}'].argmax(-1)
    for block in (1,2,3):output[f'oracle_block_{block}']=oracle['best_routes'].repeat_interleave(8,dim=0)[:n,block].numpy()
    output.to_csv(root/'predictions.csv',index=False)
    lines=['# Batch NLL oracle: MLP versus OpenJev','',
        '1024 tokens, K=[1,2,3,3], frozen experts/head, train16 and validation8. '
        'Each case keeps independent soft weights; the batch-optimal hard target is copied to its cases. '
        'Both routers are trained from fresh initialization; no compatibility or old-router targets.',
        '', '| Method | Selected phase | Validation Top1 | NLL |','|---|---|---:|---:|']
    for m,v in selected.items():lines.append(f'| {m} | {v["phase"]} | {v["acc1"]*100:.6f}% | {v["nll"]:.6f} |')
    lines += ['', '| Router | MLP oracle accuracy | OpenJev oracle accuracy |','|---|---:|---:|']
    for b in (1,2,3):lines.append(f'| block {b} | {selected["mlp"]["router"][str(b)]["oracle_accuracy"]*100:.4f}% | {selected["openjev"]["router"][str(b)]["oracle_accuracy"]*100:.4f}% |')
    lines += ['', 'Router targets are components of the full-route batch NLL optimum, '
              'not independently optimized local experts. No test evaluation. One seed. '
              'The soft-prefix refresh phase keeps the copied global target; it is not the document\'s hard conditional R-B.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--merge-only',action='store_true');a=p.parse_args();root=a.run_dir
    c=json.loads((root/'config.json').read_text());torch.set_num_threads(1)
    if a.merge_only:merge_oracle(root,c)
    else:audit(root,c)


if __name__=='__main__':main()
