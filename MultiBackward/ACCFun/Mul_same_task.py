import torch
import numpy as np


class Weight_acc:
    
    def __init__(self, num_class, tasks, eoss_temperature=1.0):
        self.tasks = tasks
        if not tasks:  
            raise ValueError("Task list cannot be empty for Weight_acc")
        self.num_class = num_class
        self.eoss_temperature = eoss_temperature
        self.weigh = {t: torch.ones(num_class, device='cpu') for t in tasks}
        self.max_weight_task = [tasks[0] for i in range(num_class)]
        self.weigh_save_list = {t: []
                                for t in tasks}  # Keep for potential future use
        self.weigh_save_list['max'] = []
        self.weigh_save_list['conflict'] = np.zeros(
            [400, 2])  # Assuming max 400 epochs

    def update(self, all_preds_np, all_targets_np):
        import numpy as np
        import torch


        if not hasattr(self, 'tasks') or not self.tasks:
            return None


        if not isinstance(all_targets_np, np.ndarray):
            try:
                all_targets_np = np.array(all_targets_np)
            except Exception:
                return None

        num_samples = len(all_targets_np)
        C = int(self.num_class)
        tasks = list(self.tasks)
        T = len(tasks)
        eps = 1e-12


        try:
            counts = np.bincount(all_targets_np.astype(int), minlength=C)[:C].astype(int)
        except Exception:
            counts = np.zeros(C, dtype=int)
            for cls in range(C):
                counts[cls] = int((all_targets_np == cls).sum())


        acc_matrix = np.zeros((T, C), dtype=float)   # per-task per-class accuracy
        pred_freq = np.zeros((T, C), dtype=float)    # how often task predicts each class across val set


        for ti, t in enumerate(tasks):
            preds = all_preds_np.get(t, None)
            if preds is None or len(preds) != num_samples:
                # leave row zeros if task missing or mismatch
                continue
            preds = np.array(preds).astype(int)
            # per-class accuracy (vectorized loop over classes)
            for cls in range(C):
                mask = (all_targets_np == cls)
                n = int(mask.sum())
                if n > 0:
                    cls_preds = preds[mask]
                    acc_matrix[ti, cls] = float((cls_preds == cls).sum()) / float(n)
            # prediction frequency across the whole validation set
            pc = np.bincount(preds, minlength=C)[:C].astype(float)
            pred_freq[ti, :] = pc / float(num_samples + eps)


        if np.all(acc_matrix == 0) and np.all(pred_freq == 0):
            return None


        mean_acc_per_class = acc_matrix.mean(axis=0)


        total_count = counts.sum() + eps
        freq = counts.astype(float) / float(total_count)
        rarity = 1.0 - freq                     
        difficulty = 1.0 - mean_acc_per_class   
        base_temp = getattr(self, 'eoss_base_temp', 0.9)
        tail_temp_scale = getattr(self, 'eoss_tail_temp_scale', 0.6)
        head_momentum = getattr(self, 'eoss_head_momentum', 0.94)
        tail_momentum = getattr(self, 'eoss_tail_momentum', 0.38)
        boost_strength = getattr(self, 'eoss_boost_strength', 2.2)
        confidence_weight = getattr(self, 'eoss_confidence_w', 0.9)
        min_weight = getattr(self, 'eoss_min_weight', 1e-8)
        tail_mask = (counts >= 50)
        # Class-adaptive temperature (tails -> smaller / sharper)
        temp_per_class = np.where(tail_mask, base_temp * tail_temp_scale, base_temp)
        temp_per_class = np.clip(temp_per_class, 0.05, 5.0)
        rel_adv = np.maximum(0.0, acc_matrix - mean_acc_per_class[np.newaxis, :])
        mean_pred_freq_per_task = pred_freq.mean(axis=1, keepdims=True)
        specialist_signal = pred_freq - mean_pred_freq_per_task
        specialist_signal = np.clip(specialist_signal, -0.05, 0.2)
        score_base = acc_matrix + 0.5 * rel_adv + confidence_weight * specialist_signal
        boost_multiplier = 1.0 + boost_strength * (rarity * difficulty) * tail_mask.astype(float)
        score_boosted = score_base * boost_multiplier[np.newaxis, :]
        # Temperature-scaled softmax across tasks for each class -> per-class normalized weights
        denom = temp_per_class[np.newaxis, :] + eps
        exp_scores = np.exp(score_boosted / denom)
        sum_exp = exp_scores.sum(axis=0, keepdims=True) + 1e-12
        weights_np = exp_scores / sum_exp   # shape [T, C]

        # Retrieve previous weights (if any) into matrix for blending
        old_matrix = np.zeros((T, C), dtype=float)
        for ti, t in enumerate(tasks):
            old_w = getattr(self, 'weigh', {}).get(t, None)
            if old_w is not None:
                try:
                    old_cpu = old_w.to(torch.device('cpu')).detach().numpy().astype(float)
                    if old_cpu.shape[0] == C:
                        old_matrix[ti, :] = old_cpu
                except Exception:
                    # ignore and leave zeros
                    pass

        # Class-adaptive momentum vector: tails adapt faster (lower momentum)
        momentum_vec = np.where(tail_mask, tail_momentum, head_momentum)

        # Blend old and new weights per-class
        blended = (momentum_vec[np.newaxis, :] * old_matrix) + ((1.0 - momentum_vec[np.newaxis, :]) * weights_np)

        # Re-normalize per-class to ensure sum==1 across tasks
        sum_blend = blended.sum(axis=0, keepdims=True) + eps
        blended = blended / sum_blend

        blended = np.clip(blended, min_weight, 1.0)

        cpu = torch.device('cpu')
        if not hasattr(self, 'weigh') or self.weigh is None:
            self.weigh = {}

        for ti, t in enumerate(tasks):
            vec = blended[ti, :].astype(np.float32)
            try:
                self.weigh[t] = torch.tensor(vec, dtype=torch.float32, device=cpu)
            except Exception:
                # fallback assignment
                self.weigh[t] = torch.tensor(vec, dtype=torch.float32, device=cpu)


        if not hasattr(self, 'max_weight_task') or self.max_weight_task is None:
            self.max_weight_task = [tasks[0] for _ in range(C)]

        for cls in range(C):
            best_ti = int(np.argmax(blended[:, cls]))
            if 0 <= best_ti < T:
                self.max_weight_task[cls] = tasks[best_ti]

        # Save counts for use by cat_targets_method
        try:
            self.class_counts = counts
        except Exception:
            pass

        return None

    def cat_out(self, logits):
        """Weighted sum of softmax probabilities."""
        # ... (Implementation from previous version with device handling) ...
        if not self.tasks:
            return None  # Or raise error
        # --- Ensure device consistency ---
        first_task_logit = logits.get(self.tasks[0], None)
        if isinstance(first_task_logit, torch.Tensor):
            device = first_task_logit.device
        elif self.weigh:  # If logits empty, try device from weights
            device = next(iter(self.weigh.values())).device
        else:  # Fallback
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        # Ensure weights and logits are on the same device
        output_sum = None
        for t in self.tasks:
            if t not in logits or not isinstance(logits[t], torch.Tensor):
                continue
            current_logits = logits[t].to(device)
            current_weights = self.weigh[t].to(device)
            if output_sum is None:  # Initialize output_sum with correct shape and device
                output_sum = torch.zeros_like(current_logits)
            # Ensure weights broadcast correctly (N, C) * (C,) -> (N, C)
            output_sum += (current_weights.unsqueeze(0) *
                           torch.softmax(current_logits, dim=1))
        return output_sum

    def cat_targets(self, logits, targets, epoch):

        import torch

        if not hasattr(self, 'tasks') or not self.tasks:
            return None

        # Determine device; prefer targets.device if targets is a tensor
        if isinstance(targets, torch.Tensor):
            device = targets.device
            targets = targets.to(device)
        else:
            device = torch.device('cpu')
            targets = torch.tensor(targets, device=device)

        # Prepare out_preds and ensure logits moved to device
        out_preds = {}
        first_task_name = None
        for t in self.tasks:
            if t not in logits or not isinstance(logits[t], torch.Tensor):
                continue
            if first_task_name is None:
                first_task_name = t
            logits[t] = logits[t].to(device)
            out_preds[t] = torch.argmax(logits[t], dim=1)

        if not out_preds or first_task_name is None:
            return None

        catout = torch.zeros_like(out_preds[first_task_name])


        if len(out_preds) > 1:
            task_list = list(out_preds.keys())
            same_index = torch.ones_like(catout, dtype=torch.bool)
            for i in range(1, len(task_list)):
                same_index = same_index & (out_preds[task_list[0]] == out_preds[task_list[i]])
        else:
            same_index = torch.ones_like(catout, dtype=torch.bool)

        catout[same_index] = out_preds[first_task_name][same_index]


        conflict_indices = torch.nonzero(~same_index).squeeze(1)
        if conflict_indices.numel() == 0:
            return torch.nn.functional.one_hot(catout, num_classes=self.num_class).float()


        class_counts = getattr(self, 'class_counts', None)


        tail_temp = getattr(self, 'eoss_cat_tail_temp', 0.45)
        head_temp = getattr(self, 'eoss_cat_head_temp', 1.0)
        margin_weight = getattr(self, 'eoss_cat_margin_w', 0.55)
        weight_bonus_scale = getattr(self, 'eoss_cat_weight_bonus', 1.05)
        tail_priority = getattr(self, 'eoss_cat_tail_priority', 1.25)
        eps = 1e-8


        for idx in conflict_indices:
            i = int(idx.item())
            true_cls = int(targets[i].item())
            if not (0 <= true_cls < self.num_class):
                catout[i] = out_preds[first_task_name][i]
                continue


            if class_counts is None:
                is_tail = (true_cls >= 50)  # fallback heuristic
            else:
                is_tail = (int(class_counts[true_cls]) >= 50)

            best_score = -float('inf')
            best_task = None

            for t in self.tasks:
                if t not in logits or t not in out_preds:
                    continue
                task_logits = logits[t][i]  # shape [C]

                # Choose temperature depending on tail/head
                temp = tail_temp if is_tail else head_temp
                temp = max(temp, 1e-6)

                # Temperature-scaled softmax confidence for the true class
                probs = torch.softmax(task_logits / temp, dim=0)
                confidence = float(probs[true_cls].item())

                # Normalized logit margin between top1 and top2
                topk = torch.topk(task_logits, k=2)
                top1 = float(topk.values[0].item())
                top2 = float(topk.values[1].item()) if topk.values.shape[0] > 1 else 0.0
                margin = (top1 - top2) / (abs(top1) + abs(top2) + eps)

                # Weight component from EOSS per-class weight (if available)
                weight_component = 0.0
                try:
                    w_t = getattr(self, 'weigh', {}).get(t, None)
                    if w_t is not None:
                        # use .to('cpu') to safely access single value
                        weight_component = max(0.0, float(w_t[true_cls].to(torch.device('cpu')).item()))
                except Exception:
                    weight_component = 0.0

                # Combine signals: favor high confidence, amplified by weight_component, plus margin
                score = confidence * (1.0 + weight_bonus_scale * weight_component) + margin_weight * float(margin)

                # For tail classes, give extra priority to confident specialists
                if is_tail:
                    score *= tail_priority

                # Slight epoch-based nudge to increasingly trust learned weights over time
                epoch_trust = min(max(epoch / 200.0, 0.0), 0.4)
                score = score * (1.0 + epoch_trust * 0.12)

                if score > best_score:
                    best_score = score
                    best_task = t

            if best_task is not None and best_task in out_preds:
                catout[i] = out_preds[best_task][i]
            else:
                catout[i] = out_preds[first_task_name][i]

        return torch.nn.functional.one_hot(catout, num_classes=self.num_class).float()


    def cuda(self):
        """Move internal weight tensors to CUDA."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            for t in self.tasks:
                if t in self.weigh:
                    self.weigh[t] = self.weigh[t].to(device)
        return self  # Allow chaining

    def cpu(self):
        """Move internal weight tensors to CPU."""
        device = torch.device("cpu")
        for t in self.tasks:
            if t in self.weigh:
                self.weigh[t] = self.weigh[t].to(device)
        return self  # Allow chaining