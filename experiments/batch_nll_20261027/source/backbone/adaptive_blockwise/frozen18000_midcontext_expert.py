"""Original512-trained18k expert evaluated with explicit1024 input budget.

Training provenance is retained verbatim. The changed inference budget is
separate metadata; no claim is made that these weights were trained at1024.
"""
from pathlib import Path
import torch
from .modern_clinical_expert import ModernClinicalExpert
from .io_utils import sha256_file


class Frozen18000MidcontextExpert(ModernClinicalExpert):
    def __init__(self,root,checkpoint,device='cuda:0'):
        super().__init__(root,checkpoint,device)
        assert self.step == 18000 and self.max_length == 512 and self.tokenized_length == 2048
        self.training_max_length = self.max_length
        self.max_length = 1024
        self.inference_config = {'max_length':1024,'trained_max_length':512,'operation':'Original2048tokenizer output right-sliced1024',
                                 'parent_step':18000,'weights_changed':False}
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        parameters = dict(self.model.named_parameters())
        assert all(torch.equal(parameters[k].cpu(),v) for k,v in saved['state'].items())
        self.artifact_hashes[str(Path(__file__))] = sha256_file(__file__)
