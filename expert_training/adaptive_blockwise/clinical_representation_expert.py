"""Text-only clinical hidden states for a fixed MoE residual expert."""
import numpy as np
import torch
from .modern_clinical_expert import ModernClinicalExpert


class ClinicalRepresentationExpert(ModernClinicalExpert):
    @torch.inference_mode()
    def representations(self,texts,batch_size=32):
        if not texts or batch_size<1:raise ValueError('Nonempty texts and positive batch size required')
        captured={}
        def capture(_module,inputs):
            if 'hidden' in captured:raise RuntimeError('Classifier invoked twice without consuming representation')
            captured['hidden']=inputs[0].float()
        classifier=self.model.base_model.model.classifier
        handle=classifier.register_forward_pre_hook(capture)
        features=[]
        scores=[]
        try:
            with torch.autocast(self.device.type,dtype=torch.bfloat16):
                for start in range(0,len(texts),batch_size):
                    batch=texts[start:start+batch_size]
                    logits=self.model(**self.tokenize(batch)).logits.float()
                    hidden=captured.pop('hidden')
                    assert hidden.shape==(len(batch),1024)
                    assert torch.isfinite(logits).all() and torch.isfinite(hidden).all()
                    features.append(hidden.cpu().numpy())
                    scores.append(logits.cpu().numpy())
            return np.concatenate(features),np.concatenate(scores)
        finally:
            handle.remove()
            captured.clear()
