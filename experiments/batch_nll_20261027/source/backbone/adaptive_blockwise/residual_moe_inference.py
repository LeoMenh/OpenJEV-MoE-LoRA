"""Online text-only MoE inference with the same feature transform used by the cache."""
from torch import nn
from .frozen_features import last_and_mean


class ResidualMoE(nn.Module):
    def __init__(self, moe, head):
        super().__init__()
        self.moe=moe
        self.head=head
        self.moe.requires_grad_(False)
        self._features=None
        self._handle=self.moe.core.model.model.register_forward_hook(self._capture)
        self.eval()

    @property
    def context(self):
        return self.moe.context

    def _capture(self, _module, _args, output):
        self._features=last_and_mean(output.last_hidden_state,self.moe.context.mask)

    def train(self, mode=True):
        super().train(mode)
        self.moe.eval()
        return self

    def forward(self, input_ids, attention_mask, mode='soft'):
        if self._handle is None:
            raise RuntimeError('Feature capture hook has been closed')
        self._features=None
        try:
            logits,balance=self.moe(input_ids=input_ids,attention_mask=attention_mask,mode=mode)
            if self._features is None:
                raise RuntimeError('Backbone did not expose final text features')
            return self.head(self._features,logits),balance
        finally:
            self._features=None

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle=None
        self._features=None
