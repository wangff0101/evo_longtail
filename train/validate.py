from utils.accuracy import AverageMeter, accuracy
from progress.bar import Bar
import torch
import numpy as np
import time
import torch.nn as nn

def horizontal_flip_aug(model):
    def aug_model(inputs):
        logits = model(inputs)
        h_logits =  model(inputs.flip(3))
        return (logits + h_logits) / 2

    return aug_model

def valid_base(valloader, model, criterion, per_class_num, num_class=100, mode='Test Stats', test_aug=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if test_aug:
        model = horizontal_flip_aug(model)

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)

    all_preds = np.zeros(len(valloader.dataset))
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(valloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            indexs = data_tuple[2]

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            pred_label = outputs.max(1)[1]

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))


            # measure accuracy and record loss

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction

            all_preds[indexs] = pred_label.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
            )
            bar.next()
        bar.finish()
        # Major, Neutral, Minor

        all_targets = np.array(valloader.dataset.targets)
        pred_mask = (all_targets == all_preds).astype(float)
        for i in range(num_class):
            class_mask = np.where(all_targets == i)[0].reshape(-1)
            classwise_correct[i] += pred_mask[class_mask].sum()
            classwise_num[i] += len(class_mask)

        classwise_acc = (classwise_correct / classwise_num)

        per_class_num = torch.tensor(per_class_num)
        many_pos = torch.where(per_class_num > 100)
        med_pos = torch.where((per_class_num <= 100) & (per_class_num >= 20))
        few_pos = torch.where(per_class_num < 20)
        section_acc[0] = classwise_acc[many_pos].mean()
        section_acc[1] = classwise_acc[med_pos].mean()
        section_acc[2] = classwise_acc[few_pos].mean()

    return losses.avg, top1.avg, section_acc

# --- EOSS Validation Function ---
def valid_eoss(dataloader, model, weight_acc, criterion, N_SAMPLES_PER_CLASS, num_class, mode='EOSS Valid', epoch=0):
    """
    Validation function for EOSS.
    MODIFIED: Uses cat_targets for prediction and updates weights AFTER loop.
    Args:
        epoch (int): Current epoch number, needed for cat_targets.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter() # Loss calculation is difficult with cat_targets output
    top1 = AverageMeter()   # Overall EOSS top1 based on cat_targets
    # top5 = AverageMeter()   # Top5 calculation is difficult with cat_targets output

    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(dataloader))

    # --- Store predictions and targets for overall update --- #
    task_preds = {task_name: [] for task_name in weight_acc.tasks}
    all_targets_list = []
    eoss_preds_list = [] # Store EOSS final predictions used for calculating metrics
    # --- End storage init --- #

    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(dataloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            batch_size = inputs.size(0)
            all_targets_list.append(targets.cpu()) # Store targets on CPU

            data_time.update(time.time() - end)

            # --- Get model outputs (logits dict) --- #
            # Ensure the model itself is in eval mode
            model.eval()
            classifier_heads = model.module.fc_heads if isinstance(model, nn.DataParallel) else model.fc_heads
            # Ensure heads are in eval mode for getting predictions
            classifier_heads.eval()
            encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
            features = encoder(inputs)
            output_dict = {}
            for task_name, head in classifier_heads.items():
                if task_name in weight_acc.tasks:
                     output_dict[task_name] = head(features)
            # --- End getting outputs --- #

            if not output_dict:
                 print(f"Warning (batch {batch_idx}): Model output dict is empty. Skipping EOSS batch.")
                 continue

            # --- Store individual task predictions (as numpy) for weight update --- #
            for task_name in weight_acc.tasks:
                 if task_name in output_dict:
                     # Check for NaN/Inf in logits before argmax
                     if torch.isnan(output_dict[task_name]).any() or torch.isinf(output_dict[task_name]).any():
                         print(f"Warning: NaN/Inf logits for task '{task_name}' in batch {batch_idx}. Appending empty.")
                         task_preds[task_name].append(np.array([], dtype=np.int64))
                     else:
                         _, task_pred_labels = torch.max(output_dict[task_name], 1)
                         task_preds[task_name].append(task_pred_labels.cpu().numpy()) # Store as numpy
                 else:
                     # Append empty array if a task's output is missing
                     task_preds[task_name].append(np.array([], dtype=np.int64))
                     print(f"Warning: Missing logits for task '{task_name}' in batch {batch_idx} for storage.")
            # --- End storing predictions --- #

            # --- EOSS Prediction using cat_targets (for current epoch metrics) --- #
            eoss_combined_pred_one_hot = None
            valid_batch_for_metric = True
            try:
                 # Get EOSS predictions based on max weight task per class (using *current* weights)
                 eoss_combined_pred_one_hot = weight_acc.cat_targets(output_dict, targets, epoch)
                 if eoss_combined_pred_one_hot is None or torch.isnan(eoss_combined_pred_one_hot).any() or torch.isinf(eoss_combined_pred_one_hot).any():
                      print(f"Warning (batch {batch_idx}): Invalid output from cat_targets. Skipping metric update.")
                      valid_batch_for_metric = False
            except Exception as e:
                 print(f"Error during weight_acc.cat_targets (batch {batch_idx}): {e}. Skipping batch metrics.")
                 valid_batch_for_metric = False
            # --- End EOSS Prediction --- #

            if valid_batch_for_metric:
                 # Calculate Top-1 accuracy from one-hot EOSS predictions
                 try:
                     eoss_pred_labels = torch.argmax(eoss_combined_pred_one_hot, dim=1)
                     eoss_preds_list.append(eoss_pred_labels.cpu()) # Store EOSS predictions for final M/M/F calculation

                     # Calculate batch accuracy
                     correct_k = (eoss_pred_labels == targets).float().sum()
                     batch_acc1 = correct_k * (100.0 / batch_size) if batch_size > 0 else torch.tensor(0.0)

                     if not torch.isnan(batch_acc1) and not torch.isinf(batch_acc1):
                          top1.update(batch_acc1.item(), batch_size) # Update Top1 meter
                 except Exception as e:
                     print(f"Error calculating EOSS accuracy (batch {batch_idx}): {e}. Skipping accuracy update.")

            batch_time.update(time.time() - end)
            end = time.time()

            # Update progress bar
            bar.suffix = (
                 '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '
                 'Total: {total:} | ETA: {eta:} | top1: {top1: .4f}'
                 .format(
                      batch=batch_idx + 1, size=len(dataloader),
                      data=data_time.avg, bt=batch_time.avg,
                      total=bar.elapsed_td, eta=bar.eta_td,
                      top1=top1.avg
                 )
             )
            bar.next()
        # --- End Batch Loop --- #
        bar.finish()

    # --- Update EOSS weights AFTER the loop using aggregated data --- #
    all_targets_np = np.concatenate(all_targets_list)
    all_preds_np = {}
    valid_update_data = True
    for task_name in weight_acc.tasks:
         try:
             # Ensure list is not empty before concatenating
             if task_preds[task_name]:
                 concatenated_preds = np.concatenate(task_preds[task_name])
                 # Check if concatenated shape matches targets shape
                 if concatenated_preds.shape == all_targets_np.shape:
                     all_preds_np[task_name] = concatenated_preds
                 else:
                     print(f"Error: Shape mismatch after concatenating task '{task_name}'. Preds: {concatenated_preds.shape}, Targets: {all_targets_np.shape}. Skipping update.")
                     valid_update_data = False; break
             else:
                  print(f"Warning: No predictions collected for task '{task_name}'. Skipping EOSS update.")
                  valid_update_data = False; break
         except ValueError as e:
             print(f"Error concatenating predictions for task '{task_name}': {e}. Likely due to empty arrays from NaN/Inf logits. Skipping EOSS update.")
             valid_update_data = False; break

    if valid_update_data:
        try:
             print(f"Updating EOSS weights based on {len(all_targets_np)} validation samples.")
             weight_acc.update(all_preds_np, all_targets_np)
             weight_acc.cuda() # Move weights back to cuda after update
        except Exception as e:
             print(f"Error during weight_acc.update after validation: {e}")
    # --- End EOSS Update --- #

    # --- Final Metric Calculations --- #
    final_top1 = top1.avg / 100.0 if np.isfinite(top1.avg) else float('nan') # Return as fraction
    final_loss = float('nan') # Loss not calculated

    # Calculate Many/Medium/Few based on collected EOSS predictions
    many_acc, med_acc, few_acc = 0.0, 0.0, 0.0
    if eoss_preds_list:
        try:
             all_eoss_preds_np = np.concatenate(eoss_preds_list)
             if all_eoss_preds_np.shape == all_targets_np.shape:
                 many_acc, med_acc, few_acc = calculate_group_accuracy(all_eoss_preds_np, all_targets_np, N_SAMPLES_PER_CLASS, num_class)
             else:
                  print(f"Error: Shape mismatch for final EOSS predictions. Preds: {all_eoss_preds_np.shape}, Targets: {all_targets_np.shape}.")
        except ValueError as e:
            print(f"Error concatenating final EOSS predictions: {e}. Likely due to empty arrays.")

    print(f'{mode} Result (EOSS - cat_targets): Acc@1 {final_top1*100:.3f}% Loss {final_loss:.5f}')
    print(f'Many: {many_acc*100:.2f}% Medium: {med_acc*100:.2f}% Few: {few_acc*100:.2f}%')

    return final_loss, final_top1, (many_acc, med_acc, few_acc)

# Helper function for group accuracy (can be moved to utils)
def calculate_group_accuracy(all_preds_np, all_targets_np, N_SAMPLES_PER_CLASS, num_class):
    classwise_correct = np.zeros(num_class)
    classwise_num = np.zeros(num_class)
    pred_mask = (all_targets_np == all_preds_np)
    for i in range(num_class):
        class_mask_np = (all_targets_np == i)
        classwise_num[i] = class_mask_np.sum()
        if classwise_num[i] > 0:
            classwise_correct[i] = pred_mask[class_mask_np].sum()
            
    classwise_acc = np.divide(classwise_correct, classwise_num, 
                              out=np.zeros_like(classwise_correct, dtype=float), 
                              where=classwise_num!=0)

    if isinstance(N_SAMPLES_PER_CLASS, torch.Tensor):
        n_samples_np = N_SAMPLES_PER_CLASS.cpu().numpy()
    else:
        n_samples_np = np.array(N_SAMPLES_PER_CLASS)
    many_mask = n_samples_np > 100
    med_mask = (n_samples_np <= 100) & (n_samples_np >= 20)
    few_mask = n_samples_np < 20

    many_acc = classwise_acc[many_mask].mean() if many_mask.any() else 0.0
    med_acc = classwise_acc[med_mask].mean() if med_mask.any() else 0.0
    few_acc = classwise_acc[few_mask].mean() if few_mask.any() else 0.0
    
    return float(many_acc), float(med_acc), float(few_acc)
# --- End EOSS Validation Function ---