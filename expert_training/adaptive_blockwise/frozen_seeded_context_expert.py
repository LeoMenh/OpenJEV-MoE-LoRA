"""Independent-seed frozen LoRA with explicitly separate inference context."""
from pathlib import Path
import torch
from .seeded_clinical_expert import SeededClinicalExpert
from .io_utils import sha256_file


class FrozenSeededContextExpert(SeededClinicalExpert):
    def __init__(self, root, checkpoint, device='cuda:0', *, inference_length):
        super().__init__(root, checkpoint, device)
        assert self.step == 18000 and self.max_length == 512
        assert self.tokenized_length == 2048 and inference_length in (512, 1536, 2048)
        assert self.provenance['config']['seed'] == 20261010
        self.training_max_length = 512
        self.max_length = inference_length
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        parameters = dict(self.model.named_parameters())
        assert all(torch.equal(parameters[key].cpu(), value) for key, value in saved['state'].items())
        self.artifact_hashes[str(Path(__file__))] = sha256_file(__file__)
