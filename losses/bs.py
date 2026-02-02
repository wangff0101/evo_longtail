import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BS(nn.Module):
    def __init__(self, dist):
        super().__init__()
        if not isinstance(dist, torch.Tensor):
            print("BS Loss: Converting input 'dist' list to tensor.")
            dist_tensor = torch.tensor(dist, dtype=torch.float)
        else:
            dist_tensor = dist.float()

        if not dist_tensor.is_cuda:
            print("BS Loss: Moving 'dist' tensor to CUDA.")
            dist_tensor = dist_tensor.cuda()

        self.prob = dist_tensor / dist_tensor.sum()
        self.log_prior = torch.log(self.prob + 1e-9).unsqueeze(0)
        
    def forward(self, logits, targets, epoch=None, reduction='mean'):
        adjusted_logits = logits + self.log_prior
        return F.cross_entropy(adjusted_logits, targets, reduction = reduction)
        
        
        # targets = F.one_hot(targets, num_classes=logits.size(1))
        # logits = logits + torch.log(self.prob.view(1, -1).expand(logits.shape[0], -1)).cuda()
        
        # if reduction == 'none':
        #     return -(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        # else:
        #     return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))