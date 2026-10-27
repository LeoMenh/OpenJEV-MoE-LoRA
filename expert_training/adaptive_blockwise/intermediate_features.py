"""Capture last and masked-mean text features after decoder layers 20 and 30."""
import torch
from .frozen_features import last_and_mean


class IntermediateFeatureCapture:
    layers = (19, 29)

    def __init__(self, moe):
        self.moe = moe
        self.values = {}
        self.handles = [moe.core.model.model.layers[i].register_forward_hook(self._hook(i))
                        for i in self.layers]

    def _hook(self, layer):
        def capture(_module, _args, output):
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            self.values[layer] = last_and_mean(hidden, self.moe.context.mask)
        return capture

    def pop(self):
        if set(self.values) != set(self.layers):
            raise RuntimeError('Incomplete intermediate feature capture')
        result = torch.cat([self.values[i] for i in self.layers], dim=1)
        self.values.clear()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()
