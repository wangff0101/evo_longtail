import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        import torch
        import copy

        if not grads:
            # Nothing to do
            return None

        # Copy grads so we are free to modify
        processed_grads = copy.deepcopy(grads)
        num_grads = len(processed_grads)
        device = processed_grads[0].device
        eps = 1e-8

        # Simple single-task shortcut
        if num_grads == 1:
            # Respect has_grads mask: if scalar True -> return grad, else mask per-dim
            g0 = processed_grads[0].view(-1).to(device)
            try:
                h0 = has_grads[0]
                if isinstance(h0, torch.Tensor):
                    if h0.numel() == 1:
                        if not bool(h0.item()):
                            return torch.zeros_like(g0)
                        return g0
                    else:
                        mask = h0.to(device).view(-1).bool()
                        out = g0 * mask.to(g0.dtype)
                        return out.view_as(processed_grads[0])
            except Exception:
                return g0

        # Stack gradients into matrix G: (num_grads, num_params)
        try:
            G = torch.stack([g.view(-1).to(device) for g in processed_grads], dim=0)
        except Exception:
            # Fallback: ensure each is flattened
            G = torch.cat([g.view(1, -1).to(device) for g in processed_grads], dim=0)

        num_params = G.shape[1]

        # Build masks per-task, same shape as G (num_grads, num_params)
        mask_list = []
        for i in range(num_grads):
            h = has_grads[i]
            if isinstance(h, torch.Tensor):
                if h.numel() == 1:
                    # scalar mask
                    if bool(h.item()):
                        mask = torch.ones(num_params, dtype=torch.float32, device=device)
                    else:
                        mask = torch.zeros(num_params, dtype=torch.float32, device=device)
                else:
                    # per-parameter mask
                    mask = h.to(device).view(-1).float()
                    if mask.numel() != num_params:
                        # Try to broadcast or fallback to ones
                        try:
                            mask = mask.expand(num_params)
                        except Exception:
                            mask = torch.ones(num_params, dtype=torch.float32, device=device)
            else:
                # Non-tensor (bool/int)
                if bool(h):
                    mask = torch.ones(num_params, dtype=torch.float32, device=device)
                else:
                    mask = torch.zeros(num_params, dtype=torch.float32, device=device)
            mask_list.append(mask)
        mask = torch.stack(mask_list, dim=0)  # float mask
        mask_bool = mask.bool()

        # Masked absolute gradients for entropy computation
        absG = (G.abs() * mask)
        sum_abs = absG.sum(dim=1, keepdim=True)  # shape (num_grads, 1)
        # Avoid division by zero for tasks with no grads: produce uniform tiny p (entropy ~0)
        safe_sum_abs = sum_abs + eps
        p = absG / safe_sum_abs

        # Entropy per task: higher -> more spread
        entropy = - (p * torch.log(p + eps)).sum(dim=1)  # shape (num_grads,)

        # Compute masked norms
        norms = torch.norm(G * mask, dim=1).clamp_min(eps)

        # Compute pairwise cosines using masked vectors (zeros in masked places)
        dot = (G * mask) @ (G * mask).t()
        denom = (norms[:, None] * norms[None, :]) + eps
        cos = dot / denom

        # Conflict intensity: sum of negative cosines (exclude diagonal)
        neg_cos = torch.clamp(-cos, min=0.0)
        if num_grads > 1:
            neg_cos.fill_diagonal_(0.0)
            conflict_score = neg_cos.sum(dim=1) / (max(1, num_grads - 1) + eps)
        else:
            conflict_score = torch.zeros(num_grads, device=device)

        # Multiplicative importance (entropy primary, boosted by conflict and inverse norm)
        # - Using multiplicative factors avoids linear/additive combos
        inv_norm = 1.0 / (norms + eps)
        # mild power factors to avoid dominance
        importance_raw = entropy.clamp_min(0.0) * torch.sqrt(conflict_score + 1.0) * (inv_norm ** 0.5)

        # If all zeros (e.g., no gradient magnitude), fallback to uniform importance
        if torch.all(importance_raw <= 0):
            importance_raw = torch.ones_like(importance_raw)

        # Softmax temperature for converting importance -> weights (configurable)
        temperature = getattr(self, '_entropy_temperature', 0.5)
        # numerical stability: shift by max
        imp_shift = importance_raw - importance_raw.max()
        weights = torch.softmax(imp_shift / (temperature + eps), dim=0)  # shape (num_grads,)

        # Now perform per-parameter weighted merge BUT only among tasks that have gradients
        # numerator: weights[:,None] * G * mask
        wcol = weights.view(-1, 1)
        numerator = (wcol * G) * mask
        denom_per_param = (wcol * mask).sum(dim=0)  # shape (num_params,)

        # Avoid division by zero: when denom_per_param == 0, result should be zero
        merged = numerator.sum(dim=0) / (denom_per_param + eps)
        # Zero out positions without any gradient (where denom ~ 0)
        no_grad_positions = denom_per_param <= eps * 10
        if no_grad_positions.any():
            merged = merged.masked_fill(no_grad_positions, 0.0)

        # Safety: remove NaN/Infs
        merged = torch.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)

        # Return merged gradient with original shape
        try:
            out = merged.view_as(processed_grads[0])
        except Exception:
            out = merged.view(-1)

        return out

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))  # Ensure length is a Python integer
            idx = int(idx)  # Ensure idx is a Python integer
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

