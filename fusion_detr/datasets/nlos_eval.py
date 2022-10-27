# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/nlos_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pynlostools.nloseval import NLOSeval
from pynlostools.nlos import NLOS
import pynlostools.mask as mask_util

from util.misc import all_gather


class NLOSEvaluator(object):
    def __init__(self, nlos_api, iou_types):
        assert isinstance(iou_types, (list, tuple))

        self.nlos_api = nlos_api

        self.iou_types = iou_types
        self.nlos_eval = {}
        for iou_type in iou_types:
            self.nlos_eval[iou_type] = NLOSeval(self.nlos_api, iouType=iou_type)

        self.img_groupIds = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_groupIds = list(np.unique(list(predictions.keys())))
        self.img_groupIds.extend(img_groupIds)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    nlos_dt = self.nlos_api.loadRes(results) if results else NLOS()

            nlos_eval = self.nlos_eval[iou_type]

            nlos_eval.nlosDt = nlos_dt
            nlos_eval.params.img_groupIds = list(img_groupIds)
            img_groupIds, eval_imgs = evaluate(nlos_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_nlos_eval(self.nlos_eval[iou_type], self.img_groupIds, self.eval_imgs[iou_type])

    def accumulate(self):
        for nlos_eval in self.nlos_eval.values():
            nlos_eval.accumulate()

    def summarize(self):
        for iou_type, nlos_eval in self.nlos_eval.items():
            print("IoU metric: {}".format(iou_type))
            nlos_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_nlos_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_nlos_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_nlos_keypoints(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_nlos_detection(self, predictions):
        nlos_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            nlos_results.extend(
                [
                    {
                        "image_group_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return nlos_results

    def prepare_for_nlos_segmentation(self, predictions):
        nlos_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            nlos_results.extend(
                [
                    {
                        "image_group_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return nlos_results

    def prepare_for_nlos_keypoints(self, predictions):
        nlos_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            nlos_results.extend(
                [
                    {
                        "image_group_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return nlos_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_groupIds, eval_imgs):
    all_img_groupIds = all_gather(img_groupIds)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_groupIds = []
    for p in all_img_groupIds:
        merged_img_groupIds.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_groupIds = np.array(merged_img_groupIds)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_groupIds, idx = np.unique(merged_img_groupIds, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_groupIds, merged_eval_imgs


def create_common_nlos_eval(nlos_eval, img_groupIds, eval_imgs):
    img_groupIds, eval_imgs = merge(img_groupIds, eval_imgs)
    img_groupIds = list(img_groupIds)
    eval_imgs = list(eval_imgs.flatten())

    nlos_eval.evalImgs = eval_imgs
    nlos_eval.params.img_groupIds = img_groupIds
    nlos_eval._paramsEval = copy.deepcopy(nlos_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.img_groupIds = list(np.unique(p.img_groupIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (img_groupId, catId): computeIoU(img_groupId, catId)
        for img_groupId in p.img_groupIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(img_groupId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for img_groupId in p.img_groupIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.img_groupIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.img_groupIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
