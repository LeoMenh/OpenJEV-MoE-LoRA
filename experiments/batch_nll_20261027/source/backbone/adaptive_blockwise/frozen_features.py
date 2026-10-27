"""Label-free features shared by frozen MoE cache and residual-head inference."""
import torch


def last_and_mean(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError('Hidden/mask shape mismatch')
    mask=mask.to(hidden.device).bool()
    if not mask.any(1).all(): raise ValueError('Empty text sequence')
    hidden=hidden.float()
    positions=torch.arange(hidden.shape[1],device=hidden.device)[None,:].expand_as(mask)
    last=positions.masked_fill(~mask,-1).max(1).values
    final=hidden[torch.arange(hidden.shape[0],device=hidden.device),last]
    mean=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
    return torch.cat((final,mean),dim=1)
