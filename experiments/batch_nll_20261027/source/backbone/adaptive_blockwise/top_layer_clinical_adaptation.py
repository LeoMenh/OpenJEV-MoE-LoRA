"""Matched LoRA continuation with optional dense matrix updates in the last two blocks."""
from pathlib import Path

from .modern_clinical_expert import ModernClinicalExpert


def build_top_layer_clinical_model(root, checkpoint, arm):
    if arm not in ('lora_control', 'top2_dense'):
        raise ValueError(arm)
    expert = ModernClinicalExpert(Path(root), checkpoint)
    assert expert.step == 18000 and expert.max_length == 2048
    model = expert.model
    named = dict(model.named_parameters())
    original = list(expert.provenance['trained_parameters'])
    dense = [name for name, parameter in named.items()
             if any(f'.model.layers.{layer}.' in name for layer in (26, 27))
             and name.endswith('.base_layer.weight') and parameter.ndim == 2]
    assert len(dense) == 8, dense
    # Both arms use identical linear-weight dtypes and starting values.
    # All normalization parameters remain frozen with their inherited dtypes.
    for name in dense:
        named[name].data = named[name].data.float()
    train_names = original + (dense if arm == 'top2_dense' else [])
    for name in train_names:
        named[name].requires_grad_(True)
    trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    assert set(trainable) == set(train_names)
    assert all(parameter.dtype.is_floating_point for parameter in trainable.values())
    saved_names = set(original) | set(dense)
    saved_parameters = {name: parameter for name, parameter in named.items() if name in saved_names}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return expert, trainable, saved_parameters, dense
