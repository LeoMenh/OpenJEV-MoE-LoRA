"""Jointly encode a note and candidate descriptions; inference never accepts labels."""
import torch
from torch import nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class CandidateRanker(nn.Module):
    def __init__(self, model_path, rank=16):
        super().__init__()
        encoder = AutoModel.from_pretrained(model_path, local_files_only=True, add_pooling_layer=False)
        self.encoder = get_peft_model(encoder, LoraConfig(r=rank, lora_alpha=2*rank,
                                      lora_dropout=.1, target_modules=['query', 'value']))
        self.score = nn.Linear(encoder.config.hidden_size, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, input_ids, attention_mask, token_type_ids, candidate_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state.float()
        mask = candidate_mask.to(hidden.dtype)
        pooled = torch.einsum('bkl,blh->bkh', mask, hidden) / mask.sum(-1, keepdim=True).clamp_min(1)
        return self.score(pooled).squeeze(-1)
