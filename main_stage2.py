from __future__ import print_function
import os, time
from progress.bar import Bar # Restore import
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
import importlib
# --- Add necessary imports ---
from utils.accuracy import AverageMeter, accuracy # Import AverageMeter and accuracy

# --- MOOSF Imports (only needed if is_multi_task) ---
from MultiBackward.MBACK import MBACK
from MultiBackward.ACCFun.Mul_same_task import Weight_acc # Import Weight_acc for EOSS
from losses.ce import CE as CE_Loss # Rename to avoid conflict, Corrected import name
from losses.bs import BS as BS_Loss # Corrected import name
from losses.ldam_drw import LDAM_DRW as LDAM_DRW_Loss # Added import for LDAM_DRW
from losses.ce_drw import CE_DRW as CE_DRW_Loss # Corrected class name
from losses.kps import KPSLoss as KPS_Loss # Corrected class name
from losses.bcl import BCLLoss as BCL_Loss # Corrected class name
from losses.SHIKE import SHIKELoss as SHIKE_Loss # Corrected class name
from losses.ars import ARS_LOSS as ARS_loss # Corrected class name
# --- End MOOSF Imports ---

from datasets.Cifar100LT import get_cifar100

from models.resnet import *

from train.validate import valid_base, valid_eoss # Import both validation functions

from utils.config import *
from utils.common import hms_string

from utils.logger import logger
import copy
import numpy as np # Added numpy import

# --- Helper Classes/Functions for Validation (Assume these exist or are imported) ---
# (Deleting AverageMeter, ProgressMeter, accuracy definitions)
# --- End Helper Definitions ---

args = parse_args()
reproducibility(args.seed)
args.logger = logger(args)

best_acc = 0  # best test accuracy
many_best, med_best, few_best = 0, 0, 0
# best_model = None # Removed global best_model state, save state_dict instead

# --- Define valid_eoss function ---
# (Deleting the entire valid_eoss function block)
# --- End valid_eoss function ---

def train_stage2(args, model, trainloader, valloader, testloader, N_SAMPLES_PER_CLASS):
    global best_acc, many_best, med_best, few_best # Keep track of best stats

    # --- Determine training mode ---
    is_multi_task = len(args.tasks) > 1

    # --- Loss Function Setup (Common) ---
    loss_functions = {}
    cls_num_list_tensor = torch.tensor(N_SAMPLES_PER_CLASS).float().cuda()
    if 'los' in args.tasks:
        loss_functions['los'] = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    if 'ce' in args.tasks:
        loss_functions['ce'] = CE_Loss().cuda()
    if 'bs' in args.tasks:
        loss_functions['bs'] = BS_Loss(dist=cls_num_list_tensor).cuda()
    if 'ldam' in args.tasks:
        loss_functions['ldam'] = LDAM_DRW_Loss(cls_num_list=N_SAMPLES_PER_CLASS,
                                             max_m=args.ldam_max_m,
                                             s=args.ldam_s).cuda()
    if 'cedrw' in args.tasks:
        loss_functions['cedrw'] = CE_DRW_Loss(cls_num_list=N_SAMPLES_PER_CLASS).cuda()
    if 'kps' in args.tasks:
        loss_functions['kps'] = KPS_Loss(cls_num_list=N_SAMPLES_PER_CLASS).cuda()
    if 'bcl' in args.tasks:
        loss_functions['bcl'] = BCL_Loss(cls_num_list=N_SAMPLES_PER_CLASS,
                                           temperature=args.bcl_temperature).cuda()
    if 'shike' in args.tasks:
        loss_functions['shike'] = SHIKE_Loss(args, N_SAMPLES_PER_CLASS).cuda()
    
    if "ars" in args.tasks:
        loss_functions['ars'] = ARS_loss(cls_num_list=N_SAMPLES_PER_CLASS,
                                           num_classes=args.num_class, max_m=args.ldam_max_m,
                                             s=args.ldam_s,temperature=args.bcl_temperature).cuda()
    
    if 'ldamdrw' in args.tasks: # Added instantiation for ldamdrw
        reweight_epoch = getattr(args, 'ldamdrw_reweight_epoch', 160) # Default to 160 if arg missing
        if reweight_epoch == 160 and not hasattr(args, 'ldamdrw_reweight_epoch'):
             args.logger("Warning: args.ldamdrw_reweight_epoch not found, defaulting LDAM_DRW reweight_epoch to 160.", level=2)
        loss_functions['ldamdrw'] = LDAM_DRW_Loss(cls_num_list=N_SAMPLES_PER_CLASS,
                                                  reweight_epoch=reweight_epoch,
                                                  max_m=args.ldam_max_m, # Reuse ldam args
                                                  s=args.ldam_s).cuda()     # Reuse ldam args

    # --- Optimizer Setup (Common - includes all heads even if only one is trained) ---
    params_to_optimize = []
    fc_heads_module = model.module.fc_heads if isinstance(model, nn.DataParallel) else model.fc_heads
    # Optimize ALL head parameters, even in single-task mode, as they might be used later
    # Or, only optimize the active head(s)? Let's optimize only active heads for now.
    for task_name in args.tasks:
        if task_name in fc_heads_module:
             params_to_optimize.extend(list(fc_heads_module[task_name].parameters()))
        else:
             args.logger(f"Warning: Task '{task_name}' defined but no corresponding head found.", level=2)
    params_to_optimize.extend(list(model.encoder[-1].parameters()))
    if not params_to_optimize:
        raise ValueError("No parameters found for the optimizer. Check model heads and args.tasks.")

    optimizer = optim.SGD(params_to_optimize, lr=args.finetune_lr, momentum=0.9, weight_decay=args.finetune_wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.finetune_epoch, eta_min=0.0)
    test_criterion = nn.CrossEntropyLoss() # Used by valid_base and maybe valid_eoss

    # --- MOOSF/EOSS Specific Setup ---
    weight_acc_instance = None
    mback = None
    if is_multi_task:
        args.logger("Multi-task mode detected. Initializing Weight_acc and MBACK.", level=1)
        weight_acc_instance = Weight_acc(num_class=args.num_class, tasks=args.tasks)
        weight_acc_instance.cuda() # Ensure weights are on CUDA before training loop
        mback = MBACK(optimizer, args) # MBACK takes the optimizer
    else:
        args.logger("Single-task mode detected. Using standard training loop.", level=1)
        # MBACK is not needed, standard optimizer steps will be used.

    # --- Training Loop ---
    best_model_state_dict = None
    test_accs = []
    start_time = time.time()

    losses_meter = {name: AverageMeter() for name in args.tasks} # Track loss for each task
    top1_meter = AverageMeter() # Track overall EOSS Top1 during training
    batch_time = AverageMeter() # Ensure initialized
    data_time = AverageMeter()  # Ensure initialized

    bar = Bar('Training', max=len(trainloader)) # Restore Bar usage
    end = time.time()

    for epoch in range(args.finetune_epoch):

        #model.eval() # Keep backbone frozen
        classifier_heads = model.module.fc_heads if isinstance(model, nn.DataParallel) else model.fc_heads
        classifier_heads.train() # Set only heads to train mode

        # --- Meters for logging ---
        batch_time_meter = AverageMeter() # Renamed to avoid conflict with module
        data_time_meter = AverageMeter()
        # Use a dict for losses if multi-task, single value otherwise? Simpler: use dict always.
        losses_meter = {name: AverageMeter() for name in args.tasks}
        top1_meter = AverageMeter()
        num_samples_processed = 0 # Track samples explicitly
        # --- End Meters ---

        end_time_batch = time.time() # Timing for batches

        for batch_idx, data_tuple in enumerate(trainloader):
            inputs, targets = data_tuple[0].cuda(non_blocking=True), data_tuple[1].cuda(non_blocking=True)
            batch_size = inputs.size(0)
            data_time_meter.update(time.time() - end_time_batch)

            # --- Feature Extraction (Common) ---
            #with torch.no_grad():
                #encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder
                #features = encoder(inputs)
            encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder    
            features = encoder(inputs)
            # --- Forward Pass, Loss Calculation, Backward Pass ---
            if is_multi_task:
                # --- Multi-Task Logic ---
                logits_dict = {}
                for task_name, head in classifier_heads.items():
                    if task_name in args.tasks:
                        logits_dict[task_name] = head(features)

                if not logits_dict: continue # Skip if no logits

                # --- End Debug ---

                losses_dict = {}
                valid_losses = True
                for name, criterion in loss_functions.items():
                    if name not in logits_dict: continue
                    current_logits = logits_dict[name]
                    try:
                        # Special handling for losses needing epoch or features
                        if name == 'kps' or name == 'cedrw' or name == 'ldamdrw': # Added ldamdrw
                            loss_val = criterion(current_logits, targets, epoch)
                        
                        elif  name == 'ars':
                            head = classifier_heads[name]
                            loss_val = criterion(logits=current_logits, targets=targets, epoch=epoch, reduction='mean', features=features, centers=head.weight)
                        
                        elif name == 'bcl': # Separate BCL handling
                            head = classifier_heads[name] # Get the head for centers (weights)
                            loss_val = criterion(head.weight, current_logits, features, targets) # Pass centers, logits, features, targets
                        elif name == 'shike': # SHIKE needs logits, targets, epoch
                            loss_val = criterion(current_logits, targets, epoch)
                        else: loss_val = criterion(current_logits, targets)

                        if torch.isnan(loss_val) or torch.isinf(loss_val):
                            args.logger(f"Warning: NaN/Inf loss '{name}' epoch {epoch+1}, batch {batch_idx}. Skipping.", level=3)
                            valid_losses = False; break
                        losses_dict[name] = loss_val
                    except Exception as e:
                        args.logger(f"Error calc loss '{name}': {e}", level=3); valid_losses = False; break
                
                if not valid_losses: continue

                # --- PLA Update (if enabled) ---
                if args.pla:
                     with torch.no_grad():
                         # Using first task for PLA calculation
                         first_task_name = args.tasks[0]
                         if first_task_name in logits_dict:
                             pla_logits = logits_dict[first_task_name]
                             preds_pla = torch.argmax(pla_logits, dim=1)
                             correct_per_class_pla = torch.zeros(args.num_class).cuda()
                             total_per_class_pla = torch.zeros(args.num_class).cuda()
                             for i in range(args.num_class):
                                 class_mask_pla = (targets == i)
                                 total_per_class_pla[i] = class_mask_pla.sum()
                                 correct_per_class_pla[i] = (preds_pla[class_mask_pla] == i).sum()
                             batch_acc_per_class_pla = correct_per_class_pla / (total_per_class_pla + 1e-6)
                             mback.pla_update([batch_acc_per_class_pla] * len(args.tasks)) # Pass list
                         else: args.logger("Warning: First task logits not found for PLA.", level=3)
                # --- End PLA ---

                # --- Corrected Backward and Step ---
                optimizer.zero_grad() # Zero gradients BEFORE backward pass
                try:
                    # Calculates gradients using PCG (if pcg=True) and sets p.grad
                    mback.backward(losses_dict)
                    # Perform optimizer step based on calculated gradients
                    optimizer.step() 
                except Exception as e:
                     import traceback
                     args.logger(f"Error during MBACK backward or optimizer step: {e}", level=3)
                     args.logger(f"Traceback: {traceback.format_exc()}", level=3)
                     continue
                # --- End Corrected Backward and Step ---

                # Update loss meters
                for name, loss_val in losses_dict.items():
                    if name in losses_meter: losses_meter[name].update(loss_val.item(), batch_size)

                # Accuracy for logging (use EOSS combined output)
                with torch.no_grad():
                     # Ensure device consistency before accuracy calculation
                     targets_device = targets.device
                     combined_probs = weight_acc_instance.cat_out(logits_dict).to(targets_device)
                     prec1_train, _ = accuracy(combined_probs, targets, topk=(1, 5))
                     if not torch.isnan(prec1_train) and not torch.isinf(prec1_train):
                         top1_meter.update(prec1_train.item(), batch_size)
                # --- End Multi-Task Logic ---

            else:
                # --- Single-Task Logic ---
                task_name = args.tasks[0]
                criterion = loss_functions[task_name]
                head = classifier_heads[task_name]

                outputs = head(features) # Direct output from the single head

                # Calculate loss
                try:
                    # Special handling for losses needing epoch or features
                    if task_name == 'kps' or task_name == 'cedrw' or task_name == 'ldamdrw': # Added ldamdrw
                        loss = criterion(outputs, targets, epoch)
                    elif task_name == 'bcl': # Separate BCL handling
                        head = classifier_heads[task_name] # Get the head for centers (weights)
                        loss = criterion(head.weight, outputs, features, targets) # Pass centers, logits, features, targets
                    elif task_name == 'shike': # SHIKE needs logits, targets, epoch
                        loss = criterion(outputs, targets, epoch)
                    else: loss = criterion(outputs, targets)

                    if torch.isnan(loss) or torch.isinf(loss):
                         args.logger(f"Warning: NaN/Inf loss '{task_name}' epoch {epoch+1}, batch {batch_idx}. Skipping.", level=3)
                         continue
                except Exception as e:
                    args.logger(f"Error calc loss '{task_name}': {e}", level=3); continue
                
                # Standard backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update loss meter
                losses_meter[task_name].update(loss.item(), batch_size)

                # Accuracy for logging
                with torch.no_grad():
                    prec1_train, _ = accuracy(outputs, targets, topk=(1, 5))
                    if not torch.isnan(prec1_train) and not torch.isinf(prec1_train):
                        top1_meter.update(prec1_train.item(), batch_size)
                # --- End Single-Task Logic ---

            num_samples_processed += batch_size
            batch_time_meter.update(time.time() - end_time_batch)
            end_time_batch = time.time() # Reset timer for next batch

        # --- End of Epoch ---
        if num_samples_processed == 0:
             args.logger(f"Warning: Epoch {epoch+1} processed 0 samples.", level=2)
             continue

        # --- Validation ---
        if is_multi_task:
            # Use EOSS validation on validation set
            val_loss, val_acc, val_cls = valid_eoss(valloader, model, weight_acc_instance, test_criterion, N_SAMPLES_PER_CLASS,
                                                    num_class=args.num_class, mode='Validation EOSS')
        else:
            # Use standard validation on validation set
            val_loss, val_acc, val_cls = valid_base(valloader, model, test_criterion, N_SAMPLES_PER_CLASS,
                                                    num_class=args.num_class, mode='Validation Base', test_aug=False)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # --- MODIFIED CHECK: Only check accuracy for NaN/Inf --- 
        if np.isnan(val_acc) or np.isinf(val_acc):
            args.logger(f"Warning: NaN/Inf detected in validation accuracy epoch {epoch+1}. Stopping early.", level=2)
            break
        # --- END MODIFIED CHECK --- 

        # Correct handling: valid_base returns %, valid_eoss returns [0,1]
        current_val_acc_percent = val_acc if not is_multi_task else val_acc * 100.0

        # Use percentage for comparison and storage
        if best_acc <= current_val_acc_percent: 
            best_acc = current_val_acc_percent
            # val_cls is always [0,1] from both functions
            many_best = val_cls[0] * 100.0
            med_best = val_cls[1] * 100.0
            few_best = val_cls[2] * 100.0
            best_model_state_dict = copy.deepcopy(model.state_dict())

        test_accs.append(current_val_acc_percent)

        # --- Logging ---
        args.logger(f'Epoch: [{epoch + 1} | {args.finetune_epoch}]', level=1)

        # Log training losses and accuracy
        train_loss_log = "\t".join([f'{name}: {meter.avg:.4f}' for name, meter in losses_meter.items()])
        args.logger(f'[Train]\tLosses: {train_loss_log}\tAcc:\t{top1_meter.avg:.4f}', level=2)
        # Log validation results (use percentage)
        args.logger(f'[Test ]\tLoss:	{val_loss:.4f}\tAcc:	{current_val_acc_percent:.4f}', level=2)
        args.logger(f'[Stats]\tMany:	{val_cls[0]*100:.2f}\tMedium:	{val_cls[1]*100:.2f}\tFew:\t{val_cls[2]*100:.2f}', level=2)
        # Log best results so far (already percentage)
        args.logger(
            f'[Best ]\tAcc:	{best_acc:.4f}\tMany:	{many_best:.2f}\tMedium:	{med_best:.2f}\tFew:	{few_best:.2f}',
            level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)

        # --- Add EOSS Weights Logging and Individual Head Validation ---
        if is_multi_task and weight_acc_instance is not None:
            # Correctly log the EOSS weight dictionary
            weights_log_str = "[EOSS ]\tWeights:\n"
            try:
                for task_name, weight_tensor in weight_acc_instance.weigh.items():
                     weights_log_str += f"  '{task_name}': shape={weight_tensor.shape}, example_vals={weight_tensor.detach().cpu().numpy()[:5]}...\n" # Show shape and first few values
            except Exception as log_e:
                 weights_log_str += f"  Error logging weights: {log_e}\n"
            args.logger(weights_log_str, level=3)

            # --- Individual Head Validation --- # Only run on the last epoch
            if epoch == args.finetune_epoch - 1:
                args.logger(f'--- Individual Head Validation (Last Epoch) ---', level=2)
                head_val_accs = {}
                model.eval() # Ensure model is in eval mode
                encoder = model.module.encoder if isinstance(model, nn.DataParallel) else model.encoder # Get encoder once
                with torch.no_grad():
                    for task_name, head in classifier_heads.items():
                        if task_name in args.tasks:
                            # --- Manual Validation Logic for this Head ---
                            head_losses = AverageMeter()
                            head_top1 = AverageMeter()
                            head_classwise_correct = torch.zeros(args.num_class).cuda() # Use cuda directly
                            head_classwise_num = torch.zeros(args.num_class).cuda()
                            temp_criterion = torch.nn.CrossEntropyLoss().cuda() # Use simple CE for temp validation

                            # Inner loop for validation data - MODIFIED: Only process first batch
                            for batch_idx_val, data_tuple_val in enumerate(testloader):
                                inputs_val = data_tuple_val[0].cuda(non_blocking=True)
                                targets_val = data_tuple_val[1].cuda(non_blocking=True)

                                # Get features and head output
                                features_val = encoder(inputs_val)
                                outputs_val = head(features_val) # Use the specific head

                                # Calculate loss and accuracy for this head
                                loss_val = temp_criterion(outputs_val, targets_val)
                                prec1_val = accuracy(outputs_val, targets_val, topk=(1,))[0] # Returns list like [tensor(..)], get first item

                                head_losses.update(loss_val.item(), inputs_val.size(0))
                                if not torch.isnan(prec1_val) and not torch.isinf(prec1_val):
                                    head_top1.update(prec1_val.item(), inputs_val.size(0))

                                # Update classwise stats
                                _, pred_label_val = torch.max(outputs_val, 1)
                                correct_mask = (pred_label_val == targets_val)
                                for i in range(args.num_class):
                                    class_mask_val = (targets_val == i)
                                    head_classwise_num[i] += class_mask_val.sum()
                                    head_classwise_correct[i] += (correct_mask & class_mask_val).sum()

                                # --- ADD BREAK --- #
                                break # Only process the first batch for sanity check
                            # --- END BREAK --- #

                            # Calculate Many/Medium/Few for this head (based on first batch)
                            classwise_acc = head_classwise_correct / (head_classwise_num + 1e-6) # Add epsilon for safety
                            
                            # Convert N_SAMPLES_PER_CLASS to tensor if it's not already
                            if isinstance(N_SAMPLES_PER_CLASS, list):
                                n_samples_tensor = torch.tensor(N_SAMPLES_PER_CLASS).cuda()
                            elif isinstance(N_SAMPLES_PER_CLASS, torch.Tensor):
                                n_samples_tensor = N_SAMPLES_PER_CLASS.cuda() # Ensure it's on CUDA
                            else: # Handle numpy array or other types if necessary
                                n_samples_tensor = torch.from_numpy(np.array(N_SAMPLES_PER_CLASS)).cuda()

                            many_pos = torch.where(n_samples_tensor > 100)[0]
                            med_pos = torch.where((n_samples_tensor <= 100) & (n_samples_tensor >= 20))[0]
                            few_pos = torch.where(n_samples_tensor < 20)[0]
                            
                            # Ensure indices are valid before accessing classwise_acc
                            many_acc_val = classwise_acc[many_pos].mean().item() if len(many_pos) > 0 else 0.0
                            med_acc_val = classwise_acc[med_pos].mean().item() if len(med_pos) > 0 else 0.0
                            few_acc_val = classwise_acc[few_pos].mean().item() if len(few_pos) > 0 else 0.0

                            head_val_accs[task_name] = head_top1.avg # Store overall head accuracy (%)
                            
                            # Log results for this head - MODIFIED: Indicate it's a sanity check
                            args.logger(f'[{task_name} Head Sanity Check (1 batch)]\tAcc:\t{head_top1.avg:.4f}%\tLoss: {head_losses.avg:.4f}', level=3)
                            # --- End Manual Validation Logic ---
                args.logger(f'--- End Individual Head Validation ---', level=2)
            # --- End Individual Head Validation ---
        # --- End Added Logging/Validation ---

        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress - Restore Bar usage
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss(avg): {loss_avg:.4f} | Acc(avg): {acc_avg:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss_avg=losses_meter[list(losses_meter.keys())[0]].avg if losses_meter else 0.0,
                    acc_avg=top1_meter.avg
                    )
        bar.next()
    bar.finish() # Restore Bar usage

    end_time_train = time.time() # Training end time

    # Save the best model state_dict
    loss_suffix = "_".join(args.losses)
    mode_suffix = "MT" if is_multi_task else "ST" # Multi-Task or Single-Task
    file_name = os.path.join(args.out, f'best_model_stage2_{mode_suffix}_{loss_suffix}.pth')
    if best_model_state_dict:
         torch.save(best_model_state_dict, file_name)
         args.logger(f"Saved best model state_dict to {file_name}", level=1)
    else:
         args.logger("No best model state_dict recorded.", level=1)


    # Print the final results
    args.logger(f'Finish Training Stage 2 ({mode_suffix} - Losses: {args.losses})...', level=1)
    args.logger(f'Final performance (based on {"EOSS" if is_multi_task else "Base"} validation)...', level=1)
    args.logger(f'best bAcc (test):\t{best_acc:.4f}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best:.2f}\tMed:\t{med_best:.2f}\tFew:\t{few_best:.2f}', level=2)
    args.logger(f'Training Time: {hms_string(end_time_train - start_time)}', level=1)

def load_model(args, model, testloader, N_SAMPLES_PER_CLASS):
    if args.pretrained_pth is not None:
        pth = args.pretrained_pth
    else:
        # Construct default path, assuming stage 1 model is needed
        pth_dir = os.path.dirname(args.out).replace(f'_{args.cur_stage}', '_stage1') # Try to infer stage1 output dir
        pth = os.path.join(pth_dir, 'best_model_stage1.pth')
        print(f"Defaulting pretrained path to: {pth}")
        if not os.path.exists(pth):
             print(f"Warning: Inferred path {pth} not found. Falling back to original default logic.")
             pth = f'pretrained/cifar100/IR={args.imb_ratio}/best_model_stage1.pth' # Original fallback

    print(f"Loading model from: {pth}")
    loaded_data = torch.load(pth, map_location=lambda storage, loc: storage)
    old_state_dict = None

    if isinstance(loaded_data, dict):
        print("Loaded data is a state_dict.")
        old_state_dict = loaded_data
    elif isinstance(loaded_data, nn.Module):
        print("Loaded data is a full model object. Extracting its state_dict.")
        old_state_dict = loaded_data.state_dict()
    else:
        raise TypeError(f"Loaded file {pth} contains unexpected data type: {type(loaded_data)}")

    is_new_dp = isinstance(model, nn.DataParallel)
    new_model_internal = model.module if is_new_dp else model
    new_state_dict_internal = new_model_internal.state_dict()

    if old_state_dict:
        base_prefix_map = {
            'conv1': 'encoder.0', 'bn1': 'encoder.1',
            'layer1': 'encoder.3', 'layer2': 'encoder.4',
            'layer3': 'encoder.5', 'layer4': 'encoder.6',
        }

        print("Applying mapping/matching from loaded state_dict keys...")
        mapped_state_dict = {}
        unmapped_old_keys = []
        shape_mismatches = []
        mapped_key_not_found_in_new = []
        skipped_fc_keys = []
        direct_match_keys = []
        los_head_mapped = False # Track if default head was mapped to LOS head
        old_fc_mapped_to_task = False # Track if old single fc was mapped
        prefix_mapped_backbone_keys = []

        # Use args.tasks (which defaults to args.losses) to check if LOS is requested
        tasks_to_run = args.tasks

        for old_key, old_value in old_state_dict.items():
            key_to_process = old_key.replace('module.', '', 1)
            mapped = False

            # --- Check if it's the OLD single fc layer from resnet.py --- #
            is_old_fc_key = key_to_process.startswith('fc.')
            if is_old_fc_key and len(tasks_to_run) == 1: # Only map if exactly one task is specified
                target_task_name = tasks_to_run[0]
                suffix = key_to_process.split('.')[-1] # weight or bias
                new_head_key_internal = f"fc_heads.{target_task_name}.{suffix}"

                if new_head_key_internal in new_state_dict_internal:
                    if new_state_dict_internal[new_head_key_internal].shape == old_value.shape:
                        print(f"  Info: Mapping old single 'fc.{suffix}' to Stage 2 '{target_task_name}' head '{new_head_key_internal}'.")
                        mapped_state_dict[new_head_key_internal] = old_value
                        old_fc_mapped_to_task = True # Mark as mapped
                        mapped = True
                    else:
                        shape_mismatches.append(f"Old single fc key '{old_key}' mapped to '{new_head_key_internal}', shape mismatch: {old_value.shape} vs {new_state_dict_internal[new_head_key_internal].shape}")
                else:
                    mapped_key_not_found_in_new.append(f"Old single fc key '{old_key}' intended for '{new_head_key_internal}', but target key not found.")
                # Whether mapped or not, continue to next old_key as we've handled this old fc key
                continue
            elif is_old_fc_key: # If old fc key but not single task, skip it
                skipped_fc_keys.append(old_key)
                continue
            # --- End OLD single fc mapping --- #

            # --- Check if it's the default head from Stage 1 (using resnet_copy.py) --- 
            is_default_head_key = key_to_process.startswith('fc_heads.default.')
            
            if is_default_head_key and 'los' in tasks_to_run:
                # Try to map Stage 1 default head to Stage 2 LOS head
                suffix = key_to_process.split('.')[-1] # weight or bias
                new_los_key_internal = f"fc_heads.los.{suffix}"
                
                if new_los_key_internal in new_state_dict_internal:
                    if new_state_dict_internal[new_los_key_internal].shape == old_value.shape:
                        print(f"  Info: Mapping Stage 1 'fc_heads.default.{suffix}' to Stage 2 'fc_heads.los.{suffix}'.")
                        mapped_state_dict[new_los_key_internal] = old_value
                        los_head_mapped = True
                        mapped = True
                    else:
                        shape_mismatches.append(f"Default head key '{old_key}' mapped to '{new_los_key_internal}', shape mismatch: {old_value.shape} vs {new_state_dict_internal[new_los_key_internal].shape}")
                        # Don't map if shapes mismatch
                else:
                    mapped_key_not_found_in_new.append(f"Default head key '{old_key}' intended for '{new_los_key_internal}', but target key not found.")
                # Whether mapped or not, continue to next old_key as we've handled this default head key
                continue 
            elif is_default_head_key: 
                 # If it's the default head but 'los' is not a task, skip it
                 skipped_fc_keys.append(old_key)
                 continue
            
            # --- Skip any other old fc layers (e.g., potential 'fc.' if structure was different) ---
            if key_to_process.startswith('fc.'):
                 skipped_fc_keys.append(old_key)
                 continue

            # --- Try mapping backbone layers --- (No change needed here)
            if not mapped:
                new_key_internal = None
                for old_prefix, new_encoder_prefix in base_prefix_map.items():
                    if key_to_process.startswith(old_prefix + '.') or key_to_process == old_prefix:
                        suffix = key_to_process[len(old_prefix):]
                        new_key_internal = new_encoder_prefix + suffix
                        mapped = True
                        break

                if mapped and new_key_internal is not None:
                    if new_key_internal in new_state_dict_internal:
                        if new_state_dict_internal[new_key_internal].shape == old_value.shape:
                            mapped_state_dict[new_key_internal] = old_value
                            prefix_mapped_backbone_keys.append(old_key)
                        else:
                            shape_mismatches.append(f"Old key '{old_key}' (prefix map) to '{new_key_internal}', shape mismatch: {old_value.shape} vs {new_state_dict_internal[new_key_internal].shape}")
                    else:
                        mapped_key_not_found_in_new.append(f"Old key '{old_key}' (prefix map) to '{new_key_internal}', but key not found in new internal state_dict")
            
            # --- Fallback: Direct key matching --- (No change needed here)
            if not mapped:
                 if key_to_process in new_state_dict_internal:
                      if new_state_dict_internal[key_to_process].shape == old_value.shape:
                           mapped_state_dict[key_to_process] = old_value
                           direct_match_keys.append(key_to_process)
                           mapped = True
                      else:
                           shape_mismatches.append(f"Old key '{old_key}' (direct match) to '{key_to_process}', shape mismatch: {old_value.shape} vs {new_state_dict_internal[key_to_process].shape}")

            # --- Record unmapped keys --- 
            if not mapped:
                 # Only record as unmapped if it wasn't explicitly skipped above
                 if old_key not in skipped_fc_keys:
                      unmapped_old_keys.append(old_key)

        # --- Loading and Logging Summary --- 
        print("\nAttempting to load processed state dict...")
        # Try strict=False first, as we *expect* missing keys for the new heads (ce, bs, etc.)
        missing_keys, unexpected_keys = new_model_internal.load_state_dict(mapped_state_dict, strict=False)
        print(f"State dict loaded into {'DataParallel model' if is_new_dp else 'model'} using strict=False.")

        # --- Updated Logging Summary --- 
        if old_fc_mapped_to_task:
             print(f"  Info: Successfully mapped Stage 1 single 'fc.*' to Stage 2 '{tasks_to_run[0]}' head.")
        if los_head_mapped:
             print("  Info: Successfully mapped Stage 1 'fc_heads.default.*' to Stage 2 'fc_heads.los.*'.")
        elif 'los' in tasks_to_run and not old_fc_mapped_to_task: # Only warn if neither mapping worked
              print("  Warning: 'los' task was requested, but failed to map Stage 1 'fc_heads.default.*' to it (check logs above for shape mismatch or missing key). Mapping from old 'fc.*' was also not attempted or failed.")
 
        if prefix_mapped_backbone_keys:
             print(f"  Info: Mapped {len(prefix_mapped_backbone_keys)} backbone keys using prefix rules (e.g., layer1 -> encoder.3).")
        if direct_match_keys:
             print(f"  Info: Matched {len(direct_match_keys)} keys directly.")

        # Analyze missing_keys (keys in the *model* that were NOT loaded)
        expected_missing_heads = []
        truly_missing_backbone = []
        target_task_name_single = tasks_to_run[0] if len(tasks_to_run) == 1 else None

        if missing_keys:
            for k in missing_keys:
                is_expected_head = False
                for task_name in tasks_to_run:
                    # A key is expected missing if:
                    # 1. It belongs to a task head OTHER than 'los' (if 'los' was mapped from default)
                    # 2. OR It belongs to a task head OTHER than the single target task (if mapped from old fc)
                    # 3. OR If 'los' wasn't mapped from default AND old 'fc' wasn't mapped to 'los'
                    mapped_from_default = los_head_mapped and task_name == 'los'
                    mapped_from_old_fc = old_fc_mapped_to_task and task_name == target_task_name_single

                    if k.startswith(f'fc_heads.{task_name}.') and not mapped_from_default and not mapped_from_old_fc:
                         is_expected_head = True; break
                
                if is_expected_head:
                     expected_missing_heads.append(k)

        # Analyze unexpected_keys (keys in mapped_state_dict that the model doesn't have)
        if unexpected_keys:
             print(f"  WARNING: Unexpected keys loaded (in mapped state_dict but not in Model): {unexpected_keys}")

        # Report keys from the old state dict that were skipped or unmapped
        final_skipped_or_unmapped = skipped_fc_keys + unmapped_old_keys
        if final_skipped_or_unmapped:
             # Refine the list: remove the original default head keys if they were successfully mapped to los
             keys_to_report = final_skipped_or_unmapped
             if los_head_mapped:
                 keys_to_report = [k for k in keys_to_report if not k.replace('module.','').startswith('fc_heads.default.')]
             # Also remove the old fc keys if they were successfully mapped
             if old_fc_mapped_to_task:
                 keys_to_report = [k for k in keys_to_report if not k.replace('module.','').startswith('fc.')]

             if keys_to_report:
                 print(f"  Info: Skipped/Unused keys from loaded state_dict: {keys_to_report}")

        # Report warnings collected during mapping
        if shape_mismatches:
             print(f"  WARNING: Shape mismatches encountered during mapping: {shape_mismatches}")
        if mapped_key_not_found_in_new:
             print(f"  WARNING: Mapped keys not found in new model structure: {mapped_key_not_found_in_new}")
        # --- End Updated Logging Summary ---

        # --- Add Validation for Loaded Stage 1 Head ('los') --- #
        # Modify validation check: Run if EITHER the default head was mapped to 'los' OR the old fc was mapped to the single task
        should_validate_loaded_head = (los_head_mapped and 'los' in tasks_to_run) or (old_fc_mapped_to_task and len(tasks_to_run) == 1)
        if should_validate_loaded_head:
            validation_task_name = 'los' if los_head_mapped else tasks_to_run[0]
            # --- FIX: Manual Validation instead of valid_base --- #
            args.logger(f"Running full validation for the loaded Stage 1 head ('{validation_task_name}')...".format(), level=1)
            model.eval() # Ensure model is in eval mode
            model_to_validate = model.module if isinstance(model, nn.DataParallel) else model
            loaded_head = model_to_validate.fc_heads[validation_task_name]
            encoder = model_to_validate.encoder
            
            # Manual validation setup
            s1_losses = AverageMeter()
            s1_top1 = AverageMeter()
            s1_classwise_correct = torch.zeros(args.num_class).cuda()
            s1_classwise_num = torch.zeros(args.num_class).cuda()
            temp_criterion = nn.CrossEntropyLoss().cuda()

            with torch.no_grad():
                 # Loop through entire testloader for this head
                 for batch_idx_s1, data_tuple_s1 in enumerate(testloader):
                     inputs_s1 = data_tuple_s1[0].cuda(non_blocking=True)
                     targets_s1 = data_tuple_s1[1].cuda(non_blocking=True)

                     # Get features and los_head output
                     features_s1 = encoder(inputs_s1)
                     outputs_s1 = loaded_head(features_s1)

                     # Calculate loss and accuracy
                     loss_s1 = temp_criterion(outputs_s1, targets_s1)
                     # Extract single value from accuracy list
                     prec1_s1 = accuracy(outputs_s1, targets_s1, topk=(1,))[0] 

                     s1_losses.update(loss_s1.item(), inputs_s1.size(0))
                     if not torch.isnan(prec1_s1) and not torch.isinf(prec1_s1):
                         s1_top1.update(prec1_s1.item(), inputs_s1.size(0))

                     # Update classwise stats
                     _, pred_label_s1 = torch.max(outputs_s1, 1)
                     correct_mask_s1 = (pred_label_s1 == targets_s1)
                     for i in range(args.num_class):
                         class_mask_s1 = (targets_s1 == i)
                         s1_classwise_num[i] += class_mask_s1.sum()
                         s1_classwise_correct[i] += (correct_mask_s1 & class_mask_s1).sum()

            # Calculate Many/Medium/Few for this head
            classwise_acc_s1 = s1_classwise_correct / (s1_classwise_num + 1e-6)
            if isinstance(N_SAMPLES_PER_CLASS, list):
                 n_samples_tensor_s1 = torch.tensor(N_SAMPLES_PER_CLASS).cuda()
            elif isinstance(N_SAMPLES_PER_CLASS, torch.Tensor):
                 n_samples_tensor_s1 = N_SAMPLES_PER_CLASS.cuda()
            else: 
                 n_samples_tensor_s1 = torch.from_numpy(np.array(N_SAMPLES_PER_CLASS)).cuda()

            many_pos_s1 = torch.where(n_samples_tensor_s1 > 100)[0]
            med_pos_s1 = torch.where((n_samples_tensor_s1 <= 100) & (n_samples_tensor_s1 >= 20))[0]
            few_pos_s1 = torch.where(n_samples_tensor_s1 < 20)[0]
            
            many_acc_s1 = classwise_acc_s1[many_pos_s1].mean().item() if len(many_pos_s1) > 0 else 0.0
            med_acc_s1 = classwise_acc_s1[med_pos_s1].mean().item() if len(med_pos_s1) > 0 else 0.0
            few_acc_s1 = classwise_acc_s1[few_pos_s1].mean().item() if len(few_pos_s1) > 0 else 0.0
            
            # Log results
            args.logger(f"Loaded Stage 1 Head ('{validation_task_name}') Performance: ", level=1)
            args.logger(f'[Validation] Loss: {s1_losses.avg:.4f}\tAcc: {s1_top1.avg:.4f}% ', level=2) 
            args.logger(f'[Stats] Many: {many_acc_s1*100:.2f}%\tMedium: {med_acc_s1*100:.2f}%\tFew: {few_acc_s1*100:.2f}%', level=2)
            # --- END FIX --- #
        # --- End Stage 1 Head Validation --- #

    else:
         print("Error: Could not extract state_dict from loaded file.")

    # Ensure the model is on the correct device
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    else:
        model.cpu()

    # Evaluate the loaded model before starting stage 2 training
    # --- Modification: Only run initial validation for single-task mode --- 
    # Let's keep the original logic for now. The validation will use EOSS 
    # which now combines the loaded LOS head with random CE/BS heads.
    # The result will be the true starting point for multi-task training.
    args.logger(f'Running initial validation after loading model (using EOSS if multi-task)...', level=1)
    model.eval() 
    test_criterion = nn.CrossEntropyLoss()
    try:
        if len(args.tasks) > 1:
            # Need to initialize Weight_acc temporarily for validation
            from MultiBackward.ACCFun.Mul_same_task import Weight_acc # Local import
            temp_weight_acc = Weight_acc(num_class=args.num_class, tasks=args.tasks)
            temp_weight_acc.cuda() # Ensure temp instance is on CUDA for initial validation
            # Pass it to valid_eoss. Note: this won't reflect learned weights yet.
            # Pass epoch=0 for initial validation if needed by cat_targets
            test_loss, test_acc, test_cls = valid_eoss(testloader, model, temp_weight_acc, test_criterion, N_SAMPLES_PER_CLASS,
                                                     num_class=args.num_class, mode='Loaded Model Validation (EOSS Init)', epoch=0) 
            # Convert test_acc from [0,1] to percentage for logging consistency below
            test_acc_percent = test_acc * 100.0 
        else:
            # Single task mode, use valid_base
            test_loss, test_acc_percent, test_cls = valid_base(testloader, model, test_criterion, N_SAMPLES_PER_CLASS,
                                                       num_class=args.num_class, mode='Loaded Model Validation (Base)', test_aug=False)
            # test_cls from valid_base is already [0,1], no need to change
        
        args.logger(f'Loaded model performance (before Stage 2 training)... ', level=1)
        args.logger(f'[Validation] Loss: {test_loss:.4f}\tAcc: {test_acc_percent:.4f}% ', level=2)
        args.logger(f'[Stats] Many: {test_cls[0]*100:.2f}%\tMedium: {test_cls[1]*100:.2f}%\tFew: {test_cls[2]*100:.2f}%', level=2)
    
    except Exception as e:
        args.logger(f"Warning: Initial validation failed after loading. Error: {e}", level=2)
        # Continue without validation results if it fails for some reason
    
    # Important: Return model to appropriate mode if subsequent action is expected
    # train_stage2 function handles setting eval/train modes

    return model

def main():
    print(f'==> Preparing imbalanced CIFAR-100')

    trainset, valset, testset = get_cifar100(
        os.path.join(args.data_dir, 'cifar100/'), args, 
        val_ratio=args.val_ratio, 
        val_from_train_ratio=args.val_from_train_ratio,
        balanced_val=args.balanced_val,
        val_samples_per_class=args.val_samples_per_class
    )
    N_SAMPLES_PER_CLASS = trainset.img_num_list

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=False, pin_memory=True, sampler=None)
    valloader = data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                 pin_memory=True)

    # Model
    print("==> creating {}{}".format(args.network, f' with heads: {args.tasks}' if args.tasks and len(args.tasks)>1 else ''))
    # Pass args.tasks (which defaults to args.losses) to the model constructor
    model = resnet34(num_classes=args.num_class, pool_size=4, task_list=args.tasks).cuda()

    # Potentially wrap model with DataParallel AFTER multi-head creation
    if args.gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = load_model(args, model, testloader, N_SAMPLES_PER_CLASS)
    train_stage2(args, model, trainloader, valloader, testloader, N_SAMPLES_PER_CLASS)

if __name__ == '__main__':
    main()