from readability_dl.metrics import loss_fn
import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig

class CLRPModel(nn.Module):
    
    def __init__(self, checkpoint):
        super(CLRPModel, self).__init__()
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(self.checkpoint)
        self.config = AutoConfig.from_pretrained(self.checkpoint)
        self.linear = torch.nn.Linear(self.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.batchnorm = torch.nn.BatchNorm1d()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
        logits = self.linear(outputs[0])
        logits = self.dropout(logits)
        logits = self.batchnorm(logits)

        loss = None
        if labels is not None:
            loss = loss_fn(logits, labels)
        
        return (loss, logits) if loss is not None else logits