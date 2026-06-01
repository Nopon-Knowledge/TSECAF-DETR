"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
import json
import datetime
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import Subset

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def _get_label2category(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    for _ in range(10):
        if dataset is None:
            return None
        if hasattr(dataset, "label2category"):
            return dataset.label2category
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
            continue
        break
    return None


def _remap_results_category_ids(results, label2category):
    remapped = []
    for output in results:
        labels = output.get("labels")
        if labels is None or labels.numel() == 0:
            remapped.append(output)
            continue

        mapped = torch.as_tensor(
            [label2category[int(x)] for x in labels.flatten().tolist()],
            device=labels.device,
            dtype=labels.dtype,
        ).reshape(labels.shape)

        remapped.append({**output, "labels": mapped})

    return remapped


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)
    iter_log_path = kwargs.get('iter_log_path', None)
    iter_log_freq = int(kwargs.get('iter_log_freq', print_freq))
    num_iters = len(data_loader)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        if iter_log_path and dist_utils.is_main_process():
            should_dump = (i + 1) % max(iter_log_freq, 1) == 0 or (i + 1) == num_iters
            if should_dump:
                iter_stats = {
                    "time": datetime.datetime.now().isoformat(timespec='seconds'),
                    "epoch": epoch,
                    "iter": i,
                    "iter_in_epoch": i + 1,
                    "iters_per_epoch": num_iters,
                    "global_step": global_step,
                    "lr": [float(pg["lr"]) for pg in optimizer.param_groups],
                    "loss_total": _to_float(loss_value),
                }
                for k, v in loss_dict_reduced.items():
                    iter_stats[f"loss_{k}"] = _to_float(v)

                if torch.cuda.is_available():
                    iter_stats["max_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, 2)

                with open(iter_log_path, "a") as f:
                    f.write(json.dumps(iter_stats) + "\n")
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types
    label2category = None if getattr(postprocessor, "remap_mscoco_category", False) else _get_label2category(data_loader)

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)
        if label2category is not None:
            results = _remap_results_category_ids(results, label2category)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator

