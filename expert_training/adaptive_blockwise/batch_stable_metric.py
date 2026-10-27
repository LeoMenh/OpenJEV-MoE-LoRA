"""Give the frozen retrieval encoder the same GEMM shape for every request."""
import torch
from torch.nn import functional as F


@torch.inference_mode()
def batch_stable_metric_encode(model,features,encoding_batch_size=1024):
    if features.ndim!=2 or len(features)==0:
        raise ValueError('Expected a nonempty matrix of patient features')
    if encoding_batch_size<1:
        raise ValueError('encoding_batch_size must be positive')
    previous=torch.get_float32_matmul_precision()
    try:
        # The selected training-case index used high precision and 1024-row
        # encoder batches. Small GEMMs can select another numeric kernel.
        # Repetition introduces no new patient data and is discarded afterward.
        torch.set_float32_matmul_precision('high')
        outputs=[]
        for start in range(0,len(features),encoding_batch_size):
            batch=features[start:start+encoding_batch_size]
            count=len(batch)
            if count<encoding_batch_size:
                batch=batch.repeat((encoding_batch_size+count-1)//count,1)[:encoding_batch_size].contiguous()
            outputs.append(F.normalize(model.encode(batch),dim=-1)[:count])
        return torch.cat(outputs)
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.inference_mode()
def batch_stable_metric_probabilities(model,features,index,labels,temperature,k,classes):
    if temperature<=0 or not 1<=k<=len(index):
        raise ValueError('Invalid retrieval temperature or neighbor count')
    previous=torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        queries=batch_stable_metric_encode(model,features)
        outputs=[]
        for start in range(0,len(queries),256):
            batch=queries[start:start+256]
            count=len(batch)
            # Single-query GEMV also has a different numeric path from GEMM.
            if count<256:
                batch=batch.repeat((256+count-1)//count,1)[:256].contiguous()
            values,neighbors=(batch@index.T).topk(k,dim=-1)
            probabilities=torch.zeros((len(batch),classes),device=features.device)
            probabilities.scatter_add_(1,labels[neighbors],(values/temperature).softmax(-1))
            outputs.append(probabilities[:count])
        return torch.cat(outputs)
    finally:
        torch.set_float32_matmul_precision(previous)
