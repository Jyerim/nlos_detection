# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import util.misc as utils
from datasets.nlos_eval import NLOSEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
# CLASSES = ['N/A','human','fire_extinguisher','dog']        # 2021
CLASSES = ['N/A','bag','glass','picket','traffic_light']

scaler = torch.cuda.amp.GradScaler()

def plot_check_nlos(img,targets,preds,output_dir,mode,epoch):
    
    img = img.cpu().numpy()
    img = 255.0*(img - np.min(img)+1e-10)/(1e-10 + np.max(img)- np.min(img))
    img = img.transpose(1,2,0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,c = img.shape

    group_id, bboxes, labels = targets['image_group_id'].cpu(), targets['boxes'].cpu(), targets['labels'].cpu()
    gt_bboxes = rescale_bboxes(bboxes,(w,h))
    if mode == 'Training':
        pred_logits, pred_bboxes = preds['pred_logits'][-1].cpu(), preds['pred_boxes'][-1].cpu()
        prob = F.softmax(pred_logits,-1)
        scores, pred_labels = prob[:,:-1].max(-1)
        topk = scores.topk(3)[0][-1]
        keep = scores > topk
        pred_bboxes = rescale_bboxes(pred_bboxes[keep],(w,h))
        scores = scores[keep]
        pred_labels = pred_labels[keep]
    else:
        scores, pred_labels,pred_bboxes = preds['scores'].cpu(), preds['labels'].cpu(), preds['boxes'].cpu()
    
    img2 = img.copy()
    for gt_label, (gt_x,gt_y,gt_x2,gt_y2) in zip(labels.tolist(),gt_bboxes.tolist()):
        img = cv2.rectangle(img,(int(gt_x),int(gt_y)),(int(gt_x2),int(gt_y2)),(255, 255, 255),3)
        text = f'{CLASSES[gt_label]}'
        cv2.putText(img,text,(int(gt_x),int(gt_y)), cv2.FONT_HERSHEY_PLAIN, 2.0,(255, 255, 255), 2)
    
    for pred_label, p, (pred_x,pred_y,pred_x2,pred_y2) in zip(pred_labels.tolist(),scores.tolist(), pred_bboxes.tolist()):
        img2 = cv2.rectangle(img2,(int(pred_x),int(pred_y)),(int(pred_x2),int(pred_y2)),(255, 255, 255),3)
        text = f'{CLASSES[pred_label]}: {p:0.2f}'
        cv2.putText(img2,text,(int(pred_x),int(pred_y)), cv2.FONT_HERSHEY_PLAIN, 2.0,(255, 255, 255), 2)
    cv2.putText(img,"GT",(0,20), cv2.FONT_HERSHEY_PLAIN, 2.0,(255, 255, 255), 2)
    cv2.putText(img2,"Pred",(0,20), cv2.FONT_HERSHEY_PLAIN, 2.0,(255, 255, 255), 2)
    out = np.hstack((img,img2))
    if mode == 'Training':
        save_folder = os.path.join(output_dir,'Train')
    else:
        save_folder = os.path.join(output_dir,'Val')
    os.makedirs(save_folder,exist_ok=True)
    save_path = os.path.join(save_folder,'{}_{}.png'.format(str(epoch),str(group_id.item())))
    cv2.imwrite(save_path,out)

def log_tensorboard(self, iter):
        for name, meter in self.meters.items():
            # print("meter type: ", meter.value)
            # exit()
            self.writer.add_scalar(name, meter.value, iter)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None, output_dir=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for rgb_images, samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        rgb_images = rgb_images.to(device)
        samples = samples.to(device)
        # print("samples.tensors: ", samples.tensors.shape)
        # print("sampels.mask: ", samples.mask.shape)
        # print("target key: ", targets[0].keys())
        # exit()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # [{'boxes':[],'labels':[],'orig_size':[]}*batch]
        
        # with torch.cuda.amp.autocast():
        outputs = model(samples) # {'pred_logits':[b,n_proposals,n_class+1],'pred_boxes':[b,n_proposals,4-cord]}
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # scaler.scale(losses).backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # plot_check_nlos(rgb_images.tensors[-1],targets[-1],outputs,output_dir,'Training',epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, nlos_api, device, output_dir,epoch=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    nlos_evaluator = NLOSEvaluator(nlos_api, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for rgb_images, samples, targets in metric_logger.log_every(data_loader, 10, header):
        rgb_images = rgb_images.to(device)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # with torch.cuda.amp.autocast():
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_group_id'].item(): output for target, output in zip(targets, results)}

        if nlos_evaluator is not None:
            nlos_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_group_id = target["image_group_id"].item()
                file_name = f"{image_group_id:012d}.png"
                res_pano[i]["image_group_id"] = image_group_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
    # plot_check_nlos(rgb_images.tensors[-1],targets[-1],results[-1],output_dir,'Validation',epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if nlos_evaluator is not None:
        nlos_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if nlos_evaluator is not None:
        nlos_evaluator.accumulate()
        nlos_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if nlos_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['nlos_eval_bbox'] = nlos_evaluator.nlos_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['nlos_eval_masks'] = nlos_evaluator.nlos_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    nlos_evaluator
    return stats, nlos_evaluator
