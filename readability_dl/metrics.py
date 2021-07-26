import torch
import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.count = 0
        self.max = 0
        self.min = 0
        self.avg = 0
        self.sum = 0
    
    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val*n
        self.avg = self.sum / self.count
        if val > self.max: self.max = val
        if val < self.min: self.min = val


def loss_fn(outputs, targets):
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    return torch.sqrt(nn.MSELoss()(outputs, targets))
