"""Text-only full-document lexical correction of truncated-context MoE logits."""
import math
import numpy as np
import torch
from .joint_inference import validate_records


class LexicalScoreFusion:
    """A fixed scalar is chosen offline; inference never accepts a target label."""
    def __init__(self, vectorizer, classifier, scale, score_kind):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Fusion scale must be finite and nonnegative')
        if score_kind not in ('linear_svm','complement_nb'):
            raise ValueError('Unknown lexical score kind')
        self.vectorizer, self.classifier = vectorizer, classifier
        self.scale, self.score_kind = float(scale), score_kind

    def __call__(self, moe_logits, texts):
        if moe_logits.ndim != 2 or len(texts) != moe_logits.shape[0]:
            raise ValueError('Mismatched text and logit batch')
        if any(not isinstance(text,str) or not text.strip() for text in texts):
            raise ValueError('Nonempty text required')
        if not np.array_equal(self.classifier.classes_,np.arange(moe_logits.shape[1])):
            raise ValueError('Lexical classes must exactly match MoE class columns')
        features=self.vectorizer.transform(texts)
        scores=(self.classifier.decision_function(features) if self.score_kind=='linear_svm'
                else self.classifier.predict_log_proba(features))
        if scores.shape != tuple(moe_logits.shape) or not np.isfinite(scores).all():
            raise ValueError('Invalid lexical scores')
        fused=moe_logits.float()+self.scale*torch.as_tensor(scores,device=moe_logits.device,dtype=torch.float32)
        if not torch.isfinite(fused).all():
            raise FloatingPointError('Nonfinite fused scores')
        return fused


@torch.inference_mode()
def predict_records(model, tokenizer, fusion, records, device, max_length=1024, batch_size=2, top_k=10):
    if min(max_length,batch_size,top_k)<1:
        raise ValueError('Lengths and batch size must be positive')
    records=validate_records(records)
    model.eval()
    for start in range(0,len(records),batch_size):
        rows=records[start:start+batch_size]
        texts=[row['text'] for row in rows]
        encoded=tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
        logits,_=model(input_ids=encoded['input_ids'].to(device),attention_mask=encoded['attention_mask'].to(device),mode='soft')
        # Lexical features use the original full texts, before tokenizer truncation.
        scores=fusion(logits,texts)
        top=scores.topk(min(top_k,scores.shape[1]),dim=-1)
        for row,indices,values in zip(rows,top.indices.cpu().tolist(),top.values.cpu().tolist()):
            yield {'sample_id':row['sample_id'],'pred_class_id':indices[0],
                   'top_class_ids':indices,'top_fused_scores':values}
