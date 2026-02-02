import torch
import torch.nn as nn
import torch.nn.functional as F

class ARS_LOSS(nn.Module):
    def __init__(self, cls_num_list=None, num_classes=100, reweight_epoch=16, max_m=0.5, s=30.0, temperature=0.1, weight=None):
        super(ARS_LOSS, self).__init__()
        if cls_num_list is None:
            raise ValueError('Loss requires cls_num_list')

        cls_num = torch.tensor(cls_num_list, dtype=torch.float)
        if cls_num.ndim == 0:
            cls_num = cls_num.unsqueeze(0)
        self.register_buffer('cls_num', cls_num)

        self.num_classes = int(num_classes)
        prob = cls_num / (cls_num.sum() + 1e-12)
        self.register_buffer('prob', prob)
        self.register_buffer('log_prior', torch.log(prob + 1e-12).unsqueeze(0))

        # effective number weights (class balanced)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, cls_num)
        cb = (1.0 - beta) / (effective_num + 1e-12)
        cb = cb / cb.mean()
        self.register_buffer('cb_weights', cb)

        # class rarity margin base
        max_n = float(cls_num.max().item()) if cls_num.numel() > 0 else 1.0
        rarity = 1.0 - torch.sqrt(cls_num / (max_n + 1e-12))
        rarity = rarity.clamp(min=0.0, max=1.0)
        class_margin = (rarity * float(max_m)).float()
        self.register_buffer('class_margin', class_margin)

        # EMA trackers for per-class accuracy and seen counts
        acc_ema = torch.zeros_like(cls_num)
        self.register_buffer('acc_ema', acc_ema)
        seen_ema = torch.zeros_like(cls_num)
        self.register_buffer('seen_ema', seen_ema)

        # EMA for prediction counts (precision denominator)
        pred_ema = torch.zeros_like(cls_num)
        self.register_buffer('pred_ema', pred_ema)

        # hyperparams
        self.reweight_epoch = int(reweight_epoch)
        self.max_m = float(max_m)
        self.s = float(s)
        self.temperature = float(temperature)
        self.weight = weight

        # dynamics
        self.ema_momentum = 0.9
        self.alpha_weight_from_acc = 1.0
        self.gamma_base = 2.0
        self.gamma_scale = 2.0
        self.smoothing_base = 0.06
        self.center_reg_base = 0.02
        self.inter_center_scale = 0.01
        self.head_protect_cap = 3.0  # don't over-boost tail beyond this factor

    def forward(self, logits, targets, epoch=None, reduction='mean', features=None, centers=None):
        if logits.dim() != 2:
            raise ValueError('logits should be 2-D [N, C]')
        device = logits.device
        N, C = logits.shape

        # move buffers
        prob = self.prob.to(device)
        log_prior = self.log_prior.to(device)
        cb_weights = self.cb_weights.to(device)
        class_margin = self.class_margin.to(device)
        cls_num = self.cls_num.to(device)

        # update EMA stats from this batch
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == targets).float()

            ones = torch.ones_like(targets, dtype=torch.float, device=device)
            class_counts = torch.zeros(C, device=device)
            class_correct = torch.zeros(C, device=device)
            pred_counts = torch.zeros(C, device=device)

            class_counts = class_counts.scatter_add(0, targets, ones)
            class_correct = class_correct.scatter_add(0, targets, correct)
            pred_counts = pred_counts.scatter_add(0, preds, ones)

            observed = class_counts > 0
            batch_acc = torch.zeros_like(class_counts)
            batch_acc[observed] = class_correct[observed] / (class_counts[observed] + 1e-12)

            batch_precision = torch.zeros_like(pred_counts)
            pred_observed = pred_counts > 0
            batch_precision[pred_observed] = class_correct[pred_observed] / (pred_counts[pred_observed] + 1e-12)

            acc_ema = self.acc_ema.to(device)
            seen_ema = self.seen_ema.to(device)
            pred_ema = self.pred_ema.to(device)

            m = float(self.ema_momentum)
            acc_ema[observed] = m * acc_ema[observed] + (1.0 - m) * batch_acc[observed]
            seen_ema[observed] = m * seen_ema[observed] + (1.0 - m) * class_counts[observed]
            pred_ema[pred_observed] = m * pred_ema[pred_observed] + (1.0 - m) * pred_counts[pred_observed]

            # write back to buffers (as CPU tensors since registered)
            self.acc_ema = acc_ema.cpu()
            self.seen_ema = seen_ema.cpu()
            self.pred_ema = pred_ema.cpu()

        # dynamic weight computation
        acc = self.acc_ema.to(device).clamp(min=0.0, max=1.0)
        pred_counts_ema = self.pred_ema.to(device)

        # protect head classes: if precision is high and pred_count >> true count, reduce aggressive upweighting
        # batch precision estimate: avoid zeros
        precision_est = torch.zeros_like(acc, device=device)
        # small smoothing using pred_ema and seen_ema
        denom = (pred_counts_ema + 1e-6)
        precision_est = (acc * seen_ema.to(device)) / denom.clamp(min=1e-6)
        precision_est = precision_est.clamp(min=0.0, max=1.0)

        # base dynamic weight: CB * (1 + alpha * (1 - acc)) scaled by rarity
        rarity = (1.0 - torch.sqrt(cls_num / (float(cls_num.max().item()) + 1e-12))).clamp(0.0, 1.0).to(device)
        dynamic_weights = cb_weights * (1.0 + self.alpha_weight_from_acc * (1.0 - acc))
        # boost by rarity but cap growth using head_protect_cap and precision
        boost = (1.0 + rarity * 2.0).clamp(max=self.head_protect_cap)
        # reduce boost for classes with high precision_est (avoid hurting well-modeled heads)
        protected_boost = boost * (1.0 - 0.5 * precision_est)
        dynamic_weights = dynamic_weights * protected_boost

        # anneal reweighting from uniform to dynamic
        if epoch is None:
            anneal = 1.0
        else:
            anneal = float(min(1.0, float(epoch) / max(1.0, float(self.reweight_epoch))))
        uniform = torch.ones_like(dynamic_weights)
        class_weights = (1.0 - anneal) * uniform + anneal * dynamic_weights

        if self.weight is not None:
            w = self.weight.to(device).float()
            if w.numel() == class_weights.numel():
                class_weights = class_weights * w

        class_weights = class_weights / (class_weights.mean() + 1e-12)

        # per-sample margin m_t: base margin amplified by difficulty and inverse sqrt of class count
        acc_smoothed = acc.clamp(min=0.0, max=1.0)
        difficulty_factor = (1.0 - acc_smoothed).clamp(min=0.0, max=1.0)
        idx = torch.arange(N, device=device)

        # scale by inverse sqrt frequency to favor rare classes
        inv_sqrt_freq = (1.0 / (torch.sqrt(cls_num + 1.0))).to(device)
        inv_fact = inv_sqrt_freq[targets]
        m_t = class_margin[targets] * (1.0 + 0.9 * difficulty_factor[targets]) * (1.0 + 0.5 * inv_fact)
        m_t = m_t.clamp(max=self.max_m)

        # prior scale anneal
        if epoch is None:
            prior_scale = 1.0
        else:
            prior_scale = float(min(1.0, float(epoch) / max(1.0, float(self.reweight_epoch))))

        logits_adj = logits.clone()

        # angular margin when features & centers provided
        if (features is not None) and (centers is not None):
            f = F.normalize(features, dim=1)
            c = F.normalize(centers, dim=1)
            cos_all = torch.matmul(f, c.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            cos_target = cos_all[idx, targets]
            theta = torch.acos(cos_target)
            cos_target_m = torch.cos(theta + m_t)
            cos_all = cos_all.clone()
            cos_all[idx, targets] = cos_target_m
            logits_adj = cos_all * self.s
            logits_adj = logits_adj + prior_scale * log_prior.to(device)
        else:
            # additive margin on logits (scaled by s to match cosine style)
            logits_adj[idx, targets] = logits_adj[idx, targets] - (m_t * self.s)
            logits_adj = logits_adj + prior_scale * log_prior.to(device)

        # class-adaptive temperature: tail -> slightly sharper (lower temp), head -> smoother
        # per-class temp factor in [0.7, 1.3]
        temp_factor = (1.0 - 0.5 * rarity).to(device)
        temp_per_class = (self.temperature * temp_factor).clamp(min=0.05)
        temp_samples = temp_per_class[targets].unsqueeze(1)
        logits_adj = logits_adj / (temp_samples + 1e-12)

        log_probs = F.log_softmax(logits_adj, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8)

        # focal gamma adapted by class difficulty
        gamma_class = (self.gamma_base + self.gamma_scale * (1.0 - acc_smoothed)).clamp(min=0.0, max=5.0).to(device)
        gamma = gamma_class[targets]
        focal_factor = torch.pow(1.0 - pt, gamma)

        # label smoothing per-class: rarer classes get slightly less smoothing
        smoothing_per_class = (self.smoothing_base * (1.0 - prob)).clamp(min=0.0, max=0.2).to(device)
        eps = smoothing_per_class[targets]
        # NLL with label smoothing: (1-eps)*(-log p_t) + eps * (-mean log p_j)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        mean_neg = -log_probs.mean(dim=1)
        nll_smooth = (1.0 - eps) * nll + eps * mean_neg

        # per-sample weights
        per_sample_w = class_weights[targets]

        # base loss per sample with focal and class weight
        loss_per_sample = nll_smooth * focal_factor * per_sample_w

        # Hard-negative mining penalty: encourage model to lower top-k negative probs
        k = min(3, C - 1)
        if k >= 1:
            neg_probs = probs.clone()
            neg_probs[idx, targets] = 0.0
            topk_vals, _ = neg_probs.topk(k, dim=1)
            sum_topk = topk_vals.sum(dim=1)
            # penalty scaled by rarity (more penalty for tail) and by confidence sum
            tail_factor = (1.0 + rarity[targets] * 2.0)
            hard_neg_penalty = (sum_topk * 0.5 * tail_factor).clamp(min=0.0)
            loss_per_sample = loss_per_sample + hard_neg_penalty
        else:
            hard_neg_penalty = torch.zeros_like(loss_per_sample)

        # center alignment regularizer
        center_reg = torch.tensor(0.0, device=device)
        inter_center_reg = torch.tensor(0.0, device=device)
        if (features is not None) and (centers is not None):
            f = F.normalize(features, dim=1)
            c = F.normalize(centers, dim=1)
            cos_target = (f * c[targets]).sum(dim=1).clamp(-1.0, 1.0)
            dist = (1.0 - cos_target)
            center_lambda = (self.center_reg_base * (1.0 + 2.0 * (1.0 - prob[targets]))).clamp(max=0.2)
            center_reg = (dist * center_lambda * per_sample_w).sum() / (per_sample_w.sum().clamp(min=1e-6))

            # centroid contrastive: push centers apart (minimize cosine similarity off-diagonal)
            Cc = F.normalize(c, dim=1)
            sim_matrix = torch.matmul(Cc, Cc.t())
            # zero diagonal
            eye = torch.eye(sim_matrix.size(0), device=device)
            off_diag = sim_matrix * (1.0 - eye)
            # encourage average off-diagonal similarity to be small
            off_mean = off_diag.sum() / (off_diag.numel() - off_diag.size(0) + 1e-12)
            inter_center_reg = (off_mean.clamp(min=0.0) * self.inter_center_scale)

        # reduction
        if reduction == 'mean':
            denom = per_sample_w.sum().clamp(min=1e-6)
            cls_loss = loss_per_sample.sum() / denom
            loss = cls_loss + center_reg + inter_center_reg
            return loss
        elif reduction == 'sum':
            return loss_per_sample.sum() + center_reg * float(N) + inter_center_reg * float(N)
        elif reduction == 'none':
            return loss_per_sample
        else:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")