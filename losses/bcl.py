"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np



class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets):
        # features is expected to be (2*bs, D) here due to BCLLoss modification
        # targets is expected to be (bs,)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # Correct batch_size based on duplicated features input
        batch_size = features.shape[0] // 2 # Calculate original batch size
        
        # Prepare targets: original targets (bs,) -> targets_batch (bs, 1) -> combined targets (2*bs + C, 1)
        targets_batch = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
        # Note: targets variable gets redefined here
        targets = torch.cat([targets_batch.repeat(2, 1), targets_centers], dim=0)
        
        # Calculate batch_cls_count (remains the same logic)
        batch_cls_count = torch.eye(len(self.cls_num_list)).cuda()[targets].sum(dim=0).squeeze()

        # Calculate mask (remains the same logic)
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # class-complement
        # REMOVED faulty feature manipulation: features = torch.cat(torch.unbind(features, dim=1), dim=0)
        # Concatenate the input features (now 2*bs, D) with centers (C, D)
        features = torch.cat([features, centers1], dim=0) # Shape becomes (2*bs + C, D)
        
        # Calculate logits (remains the same logic)
        logits = features[:2 * batch_size].mm(features.T) # (2*bs, D) x (D, 2*bs+C) -> (2*bs, 2*bs+C)
        logits = torch.div(logits, self.temperature)

        # For numerical stability (remains the same logic)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging (remains the same logic, using direct indexing)
        exp_logits = torch.exp(logits) * logits_mask

        # --- Alternative calculation for per_ins_weight using direct indexing --- 
        target_indices = targets.squeeze() 
        indexed_counts = batch_cls_count[target_indices]
        per_ins_weight = indexed_counts.view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
        # --- End alternative calculation --- 

        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        
        # Calculate loss (remains the same logic)
        log_prob = logits - torch.log(exp_logits_sum)
        # Add epsilon to mask.sum(1) to avoid division by zero if a row in mask is all zeros
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9) 

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean() # Reshape based on original batch size
        return loss

class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class BCLLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None, temperature = 0.1, alpha=2.0, beta=0.6):
        super(BCLLoss, self).__init__()
        self.criterion_ce = LogitAdjust(cls_num_list).cuda()
        self.criterion_scl = BalSCL(cls_num_list, temperature).cuda()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, centers,  logits, features, targets):
        # Input: features (bs, D), targets (bs,)
        
        # Prepare features for BalSCL by repeating
        features_scl = features.repeat(2, 1) # Shape (2*bs, D)
        
        # Call BalSCL with centers, duplicated features, and original targets
        scl_loss = self.criterion_scl(centers, features_scl, targets) 
        
        # Call CE loss with original logits and targets
        ce_loss = self.criterion_ce(logits, targets)

        return self.alpha * ce_loss + self.beta * scl_loss
