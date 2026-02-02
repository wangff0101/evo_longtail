from __future__ import print_function, absolute_import
import numpy as np
import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad(): # Add torch.no_grad() for safety
        maxk = max(topk)
        batch_size = target.size(0)

        # Handle case where batch_size is 0
        if batch_size == 0:
             # Return 0 accuracy for all k
             return [torch.tensor(0.0, device=output.device)] * len(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            # Add epsilon to batch_size in division
            res.append(correct_k.mul_(100.0 / (batch_size + 1e-8)))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # Ensure val is a scalar number and finite before updating
        if isinstance(val, torch.Tensor):
            val = val.item() # Get Python number from tensor

        # Check if n is positive and val is a finite number
        if n > 0 and isinstance(val, (int, float)) and np.isfinite(val):
            self.val = val
            self.sum += val * n
            self.count += n
            # Avoid division by zero if count becomes 0 after reset (although unlikely)
            self.avg = self.sum / self.count if self.count > 0 else 0
        # else:
            # Optional: Log skipped updates
            # print(f"AverageMeter: Skipped update with val={val} (type={type(val)}), n={n}")