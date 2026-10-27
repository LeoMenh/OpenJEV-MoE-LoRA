"""Original512-trained mature expert with an explicit frozen inference budget.

Training provenance is retained verbatim. The changed inference budget is
separate metadata; no claim is made that these weights were trained at1024.
"""
from pathlib import Path
import torch
from .modern_clinical_expert import ModernClinicalExpert
from .io_utils import sha256_file


class FrozenLateContextExpert(ModernClinicalExpert):
    def __init__(self,root,checkpoint,device='cuda:0',*,parent_step,inference_length):
        super().__init__(root,checkpoint,device)
        assert parent_step in (18000,19929) and inference_length in (1024,1536,2048)
        assert self.step == parent_step and self.max_length == 512 and self.tokenized_length == 2048
        self.training_max_length = self.max_length
        self.max_length = inference_length
        self.inference_config = {'max_length':inference_length,'trained_max_length':512,'operation':f'Original2048tokenizer output right-sliced{inference_length}',
                                 'parent_step':parent_step,'weights_changed':False}
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        parameters = dict(self.model.named_parameters())
        assert all(torch.equal(parameters[k].cpu(),v) for k,v in saved['state'].items())
        self.artifact_hashes[str(Path(__file__))] = sha256_file(__file__)
