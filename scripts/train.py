#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import time
import logging
from pathlib import Path
import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
import torchvision
from tqdm import tqdm

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from utils.general import (
    check_img_size, check_requirements, make_divisible, increment_path, 
    setup_seed, compute_loss, box_iou, non_max_suppression, xywh2xyxy, 
    plot_images, plot_labels, plot_results, strip_optimizer, labels_to_image_weights
)
from utils.datasets import LoadImagesAndLabels, create_dataloader
from utils.torch_utils import (
    ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first
)
from ultralytics import Model


logger = logging.getLogger(__name__)

def train(opt, device):
    # Set logging level
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    
    # Create save directory
    save_dir = Path(increment_path(Path(opt.save_dir) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and data paths
    with open(opt.config) as f:
        cfg = yaml.safe_load(f)
    
    # Configure directories
    train_path = cfg['train']
    val_path = cfg['val']
    nc = int(cfg['nc'])
    names = cfg['names']
    
    # Setup training parameters from config
    img_size = cfg.get('img_size', [640, 640])
    batch_size = cfg.get('batch_size', 16)
    epochs = cfg.get('epochs', 100)
    workers = cfg.get('workers', 4)
    
    # Setup logging
    if TENSORBOARD_AVAILABLE:
        tb_writer = SummaryWriter(str(save_dir / 'runs'))
    
    # Load YOLOv11 model
    model_yaml = Path(opt.model).name
    model = Model(opt.model, nc=nc)
    model = model.to(device)
    
    # Optimizer setup
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(pg0, lr=float(cfg.get('lr0', 0.01)), betas=(cfg.get('momentum', 0.937), 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay': float(cfg.get('weight_decay', 0.0005))})
    optimizer.add_param_group({'params': pg2})  # add biases
    logger.info(f'Optimizer: {type(optimizer).__name__} with parameter groups '
                f'{len(pg0)} weight, {len(pg1)} weight (with decay), {len(pg2)} bias')
    
    # Learning rate scheduler
    lf = lambda x: (1 - x / epochs) * (1.0 - cfg.get('lrf', 0.01)) + cfg.get('lrf', 0.01)  # linear
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Resume training if specified
    start_epoch = 0
    best_fitness = 0.0
    if opt.resume:
        if os.path.isfile(opt.resume):
            ckpt = torch.load(opt.resume, map_location=device)
            
            # Load model
            exclude = ['anchor']  # exclude keys
            state_dict = ckpt['model'].float().state_dict()
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            model.load_state_dict(state_dict, strict=False)
            
            # Load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
            
            # Load results
            if ckpt.get('training_results') is not None:
                with open(save_dir / 'results.txt', 'w') as file:
                    file.write(ckpt['training_results'])
            
            # Load epoch and best fitness
            start_epoch = ckpt['epoch'] + 1
            if ckpt.get('best_fitness') is not None:
                best_fitness = ckpt['best_fitness']
            
            del ckpt, state_dict
            logger.info(f'Resumed training from {opt.resume}')
        else:
            logger.info(f'No checkpoint found at {opt.resume}, starting from scratch')
    
    # Configure data loaders
    train_loader, dataset = create_dataloader(
        train_path, img_size, batch_size, gs=32, 
        augment=True,
        hyp=cfg,
        rect=False,
        cache=opt.cache_images,
        stride=32,
        pad=0.0,
        image_weights=opt.image_weights,
        prefix='train: ',
        workers=workers
    )
    
    val_loader, val_dataset = create_dataloader(
        val_path, img_size, batch_size, gs=32, 
        augment=False,
        hyp=cfg,
        rect=True,
        cache=opt.cache_images,
        stride=32,
        pad=0.5,
        image_weights=False,
        prefix='val: ',
        workers=workers
    )
    
    # Model parameters
    hyp = {}
    for k, v in cfg.items():
        if k.startswith('box_') or k.startswith('cls_') or k.startswith('obj_'):
            hyp[k] = v
    
    # Initialize EMA (Exponential Moving Average)
    ema = ModelEMA(model)
    
    # Mixed precision training with AMP
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    # Start training
    logger.info(f'Image sizes {img_size} train, {img_size} val')
    logger.info(f'Using {workers} dataloader workers')
    logger.info(f'Starting training for {epochs} epochs...')
    
    # Create output directories
    for d in ['labels', 'images', 'weights']:
        (save_dir / d).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        
        # Update image weights (optional)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - cfg.get('flipud', 0.0))  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        
        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{epochs - 1}')
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + len(train_loader) * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            
            # Warmup
            if ni <= cfg.get('warmup_iterations', 500):
                xi = [0, cfg.get('warmup_iterations', 500)]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [cfg.get('warmup_momentum', 0.8), cfg.get('momentum', 0.937)])
            
            # Multi-scale training
            if opt.multi_scale:
                sz = random.randrange(img_size[0] * 0.5, img_size[0] * 1.5 + 32) // 32 * 32  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            
            # Forward pass
            with amp.autocast(enabled=device.type != 'cpu'):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets, model, hyp)  # loss scaled by batch_size
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimize
            if ni % opt.accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
            
            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(
                f'Epoch {epoch}/{epochs-1} | '
                f'Mem: {mem} | '
                f'Box: {mloss[0]:.4f} | '
                f'Obj: {mloss[1]:.4f} | '
                f'Cls: {mloss[2]:.4f} | '
                f'Total: {mloss[3]:.4f}'
            )
            
            # Log batch results
            if TENSORBOARD_AVAILABLE and (ni % 10 == 0):
                # Log scalar values
                tb_writer.add_scalar('train/box_loss', mloss[0], ni)
                tb_writer.add_scalar('train/obj_loss', mloss[1], ni)
                tb_writer.add_scalar('train/cls_loss', mloss[2], ni)
                tb_writer.add_scalar('train/total_loss', mloss[3], ni)
                tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], ni)
                
                # Log images (first 3 batches only)
                if ni < 30:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    plot_images(imgs, targets, paths, f)
                    tb_writer.add_image(f'train_batch{ni}', torchvision.io.read_image(str(f)) / 255.0, ni)
        
        # Scheduler step
        scheduler.step()
        
        # Validation
        results, maps = validate(
            model=ema.ema if ema else model,
            dataloader=val_loader,
            save_dir=save_dir,
            batch_size=batch_size,
            conf_thres=cfg.get('conf_thresh', 0.001),
            iou_thres=cfg.get('iou_thresh', 0.6),
            augment=False
        )
        
        # Log validation results
        if TENSORBOARD_AVAILABLE:
            tb_writer.add_scalar('val/precision', results[0], epoch)
            tb_writer.add_scalar('val/recall', results[1], epoch)
            tb_writer.add_scalar('val/mAP_0.5', results[2], epoch)
            tb_writer.add_scalar('val/mAP_0.5:0.95', results[3], epoch)
            
            # Log validation batch images
            if epoch < 3:
                f = save_dir / f'val_batch{epoch}.jpg'
                if f.exists():
                    tb_writer.add_image(f'val_batch{epoch}', torchvision.io.read_image(str(f)) / 255.0, epoch)
        
        # Save model
        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'model': model.state_dict() if epoch == epochs - 1 else deepcopy(ema.ema).half() if ema else deepcopy(model).half(),
            'optimizer': optimizer.state_dict(),
            'training_results': None
        }
        
        # Save last, best and delete
        torch.save(ckpt, save_dir / 'last.pt')
        if results[2] > best_fitness:  # mAP@0.5
            best_fitness = results[2]
            torch.save(ckpt, save_dir / 'best.pt')
        
        if epoch == 0:
            torch.save(ckpt, save_dir / 'epoch_000.pt')
        elif epoch % 10 == 0:  # save every 10 epochs
            torch.save(ckpt, save_dir / f'epoch_{epoch:03d}.pt')
        
        # Delete checkpoint
        del ckpt
        
        # Plot training results
        if epoch > 0:
            plot_results(save_dir=save_dir)
        
        # Print epoch results
        logger.info(f'Epoch {epoch}/{epochs-1} results:')
        logger.info(f'Precision: {results[0]:.4f}')
        logger.info(f'Recall: {results[1]:.4f}')
        logger.info(f'mAP@0.5: {results[2]:.4f}')
        logger.info(f'mAP@0.5:0.95: {results[3]:.4f}')
    
    # Training complete
    if TENSORBOARD_AVAILABLE:
        tb_writer.close()
    
    # Strip optimizer from 'best.pt' to reduce file size
    strip_optimizer(save_dir / 'best.pt')
    torch.save(model, save_dir / 'best_full.pt')
    
    # Copy best.pt to weights/best.pt
    weights_dir = Path('weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, weights_dir / 'best.pt')
    
    logger.info(f'\nTraining complete. Results saved to {save_dir}')
    logger.info(f'Best model saved to {weights_dir / "best.pt"}')


def validate(model, dataloader, save_dir, batch_size, conf_thres=0.001, iou_thres=0.6, augment=False):
    """
    Run validation on the validation dataset
    """
    device = next(model.parameters()).device
    nc = 1  # number of classes (only punch)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    seen = 0
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    
    pbar = tqdm(dataloader, desc=s)
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        # Disable gradient calculations
        with torch.no_grad():
            # Run model
            out, train_out = model(img, augment=augment)  # inference and training outputs
            
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls
            
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if nc > 1 else []  # for autolabelling
            
            # Apply NMS
            pred = non_max_suppression(
                out, conf_thres=conf_thres, iou_thres=iou_thres, 
                labels=lb, multi_label=True, agnostic=False
            )
            
            # Statistics per image
            for si, pred_i in enumerate(pred):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                
                seen += 1
                
                if len(pred_i) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                
                # Predictions
                predn = pred_i.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # to original shape
                
                # Assign all predictions as correct
                correct = torch.zeros(pred_i.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]  # target classes
                    
                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred_i[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                        
                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], labels[ti, 1:5]).max(1)  # best ious, indices
                            
                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                
                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred_i[:, 4].cpu(), pred_i[:, 5].cpu(), tcls))
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        
        # Print results
        pbar.close()
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, stats[3].shape[0] if len(stats) > 3 else 0, mp, mr, map50, map))
    
    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=None):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness confidence score (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = len(unique_classes)
    
    # Create PR curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects
        
        if n_p == 0 or n_gt == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (n_gt + 1e-16)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        
        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
        
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    
    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    
    # Plot the precision-recall curve
    if plot:
        plot_pr_curve(px, py, ap, save_dir, names)
    
    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    
    return ap, mpre, mrec


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def plot_pr_curve(px, py, ap, save_dir='.', names=()):
    """
    Plots the precision-recall curve
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    
    if 0 < len(names) < 21:  # show mAP in legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)
    
    ax.plot(px, py.mean(1), 'k-', linewidth=3, label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir) / 'precision_recall_curve.png', dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/punch.yaml', help='data config file path')
    parser.add_argument('--model', type=str, default='models/yolov11_punch.yaml', help='model config file path')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='directory to save results')
    parser.add_argument('--name', default='exp', help='save to save_dir/name')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--multi-scale', action='store_true', help='train with multi-scale augmentation')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--exist-ok', action='store_true', help='existing dir/file ok, do not increment')
    parser.add_argument('--resume', type=str, default='', help='resume from given path')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    opt = parser.parse_args()
    
    # Check requirements
    check_requirements(exclude=('pycocotools', 'thop'))
    
    # Select device
    device = select_device(opt.device)
    
    # Set random seed
    setup_seed(0)
    
    # Start training
    train(opt, device)


if __name__ == '__main__':
    main()