# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F

from examples.torch.common.example_logger import logger
from nncf.torch.utils import get_world_size
from nncf.torch.utils import is_dist_avail_and_initialized
from nncf.torch.utils import is_main_process


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Timer:
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        return self.diff


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_detections(box_list, dataset, use_07=True):
    cachedir = os.path.join("cache", "annotations_cache")
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    logger.info("VOC07 metric? {}".format("Yes" if use_07_metric else "No"))
    for cls_ind, cls in enumerate(dataset.classes):  # for each class
        class_boxes = box_list[box_list[:, 1] == cls_ind + 1]
        ap, _, _ = voc_eval(  # calculate rec, prec, ap
            class_boxes, dataset, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric
        )
        aps += [ap]
        logger.info("AP for {} = {:.4f}".format(cls, ap))
    mAp = np.mean(aps)
    logger.info("Mean AP = {:.4f}".format(mAp))
    return mAp


def voc_ap(rec, prec, use_07_metric=True):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(class_detections, dataset, classname, cachedir, ovthresh=0.5, use_07_metric=True):
    # cachedir caches the annotations in a pickle file
    # first load gt
    gt, imagenames = load_detection_annotations(cachedir, dataset)
    image_bboxes, npos = extract_gt_bboxes(classname, dataset, gt, imagenames)

    image_names = dataset.get_img_names()
    image_ids = [image_names[int(i)] for i in class_detections[:, 0]]
    BB = class_detections[:, 3:]
    confidence = class_detections[:, 2]

    # sort by confidence
    sorted_ind = np.argsort(confidence)[::-1]
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = image_bboxes[image_ids[d]]
        bb = BB[d, :].astype(float)
        matched_iou = -np.inf
        BBGT = R["bbox"].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            matched_ind, matched_iou = match_bbox(BBGT, bb)

        if matched_iou > ovthresh:
            if not R["difficult"][matched_ind]:
                if not R["det"][matched_ind]:
                    tp[d] = 1.0
                    R["det"][matched_ind] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    return compute_detection_metrics(fp, tp, npos, use_07_metric)


def extract_gt_bboxes(classname, dataset, gt, imagenames):
    # extract gt objects for this class
    class_gt = {}
    npos = 0
    for imagename in imagenames:
        img_gt_objects_for_class = [rec for rec in gt[imagename] if dataset.classes[rec["label_idx"]] == classname]
        bbox = np.asarray([x["bbox"] for x in img_gt_objects_for_class])
        difficult = []
        for x in img_gt_objects_for_class:
            if "difficult" in x:
                difficult.append(x["difficult"])
            else:
                difficult.append(False)
        difficult = np.array(difficult).astype(bool)
        det = [False] * len(img_gt_objects_for_class)
        npos = npos + sum(~difficult)
        class_gt[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    return class_gt, npos


def load_detection_annotations(cachedir, dataset):
    cachefile = os.path.join(cachedir, "annots_{}.json".format(dataset.name))
    imagenames = dataset.get_img_names()
    if is_main_process() and not os.path.isfile(cachefile):
        # load annots
        gt = {}
        for i, imagename in enumerate(imagenames):
            _, gt[imagename] = dataset.pull_anno(i)

            if i % 100 == 0:
                logger.info("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
        # save
        logger.info("Saving cached annotations to {:s}".format(cachefile))
        pathlib.Path(cachedir).mkdir(parents=True, exist_ok=True)
        with open(cachefile, "w", encoding="utf8") as f:
            json.dump(gt, f)
    if is_dist_avail_and_initialized():
        dist.barrier()
    with open(cachefile, "r", encoding="utf8") as f:
        gt = json.load(f)
    return gt, imagenames


def compute_detection_metrics(fp, tp, n_positives, use_07_metric=True):
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(n_positives)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap, prec, rec


def match_bbox(gt_boxes, bbox):
    ixmin = np.maximum(gt_boxes[:, 0], bbox[0])
    iymin = np.maximum(gt_boxes[:, 1], bbox[1])
    ixmax = np.minimum(gt_boxes[:, 2], bbox[2])
    iymax = np.minimum(gt_boxes[:, 3], bbox[3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inters = iw * ih
    uni = (
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        + (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        - inters
    )
    overlaps = inters / uni
    matched_ind = np.argmax(overlaps)
    matched_iou = np.max(overlaps)
    return matched_ind, matched_iou


def gather_detections(all_detections, samples_per_rank):
    world_size = get_world_size()
    result = [torch.zeros(samples_per_rank, *all_detections.shape[1:]).cuda() for _ in range(world_size)]
    all_detections = F.pad(all_detections, [0, 0, 0, 0, 0, samples_per_rank - all_detections.size(0)])
    dist.all_gather(result, all_detections.cuda())
    return torch.cat(result)


def convert_detections(all_detection):
    """Returns (num_dets x 7) array with detections (img_ing, label, conf, x0, y0, x1, y1)"""
    all_boxes = []
    for img_ind, dets in enumerate(all_detection):
        # remove predictions with zero confidence
        mask = dets[:, 2].gt(0.0).expand(7, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 7).cpu()

        if dets.size() == (0,):
            continue

        boxes = np.c_[np.full(dets.size(0), img_ind), dets[:, 1:].numpy()]
        all_boxes.append(boxes)
    return np.vstack(all_boxes)


def predict_detections(data_loader, device, net):
    num_batches = len(data_loader)
    all_detections = []
    timer = Timer()
    for batch_ind, (ims, _gts, hs, ws) in enumerate(data_loader):
        x = ims.to(device)
        hs = x.new_tensor(hs).view(-1, 1)
        ws = x.new_tensor(ws).view(-1, 1)

        timer.tic()
        batch_detections = net(x)
        top_k = batch_detections.size(2)
        batch_detections = batch_detections.view(-1, top_k, 7)
        detect_time = timer.toc(average=False)

        batch_detections[..., 3] *= ws
        batch_detections[..., 5] *= ws
        batch_detections[..., 4] *= hs
        batch_detections[..., 6] *= hs

        all_detections.append(batch_detections.cpu())
        logger.info("Detect for batch: {:d}/{:d} {:.3f}s".format(batch_ind + 1, num_batches, detect_time))
    if all_detections:
        return torch.cat(all_detections)
    return None  # No predictions


def eval_net_loss(data_loader, device, net, criterion):
    batch_loss_l = AverageMeter()
    batch_loss_c = AverageMeter()
    batch_loss = AverageMeter()
    t_elapsed = AverageMeter()

    num_batches = len(data_loader)

    # Assume 10 lines of reporting
    print_freq = num_batches // 10
    print_freq = 1 if print_freq == 0 else print_freq

    # all_detections = []
    timer = Timer()
    for batch_ind, (ims, _gts, _, _) in enumerate(data_loader):
        images = ims.to(device)
        targets = [anno.requires_grad_(False).to(device) for anno in _gts]

        # forward
        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        batch_loss_l.update(loss_l.item(), images.size(0))
        batch_loss_c.update(loss_c.item(), images.size(0))
        batch_loss.update(loss.item(), images.size(0))

        timer.tic()
        t_elapsed.update(timer.toc(average=False))

        if batch_ind % print_freq == 0:
            logger.info(
                "Loss_inference: [{}/{}] || Time: {elapsed.val:.4f}s ({elapsed.avg:.4f}s)"
                " || Conf Loss: {conf_loss.val:.3f} ({conf_loss.avg:.3f})"
                " || Loc Loss: {loc_loss.val:.3f} ({loc_loss.avg:.3f})"
                " || Model Loss: {model_loss.val:.3f} ({model_loss.avg:.3f})".format(
                    batch_ind,
                    num_batches,
                    elapsed=t_elapsed,
                    conf_loss=batch_loss_c,
                    loc_loss=batch_loss_l,
                    model_loss=batch_loss,
                )
            )

    model_loss = batch_loss_l.avg + batch_loss_c.avg
    return model_loss


def test_net(net, device, data_loader, distributed=False, loss_inference=False, criterion=None):
    """Test a Fast R-CNN network on an image database."""

    # Put BN layers into evaluation mode
    bn_modules_training_flags = {}

    def put_bn_modules_in_eval_mode(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_modules_training_flags[module] = module.training
            module.eval()

    def restore_bn_module_mode(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.training = bn_modules_training_flags[module]

    net.apply(put_bn_modules_in_eval_mode)

    if loss_inference is True:
        logger.info("Testing... loss function will be evaluated instead of detection mAP")
        if distributed:
            raise NotImplementedError
        if criterion is None:
            raise ValueError("Missing loss inference function (criterion)")
        output = eval_net_loss(data_loader, device, net, criterion)
        net.apply(restore_bn_module_mode)
        return output

    logger.info("Testing...")
    num_images = len(data_loader.dataset)
    batch_detections = predict_detections(data_loader, device, net)
    if distributed:
        batch_detections = gather_detections(batch_detections, data_loader.sampler.samples_per_rank)
    batch_detections = batch_detections[:num_images]
    all_boxes = convert_detections(batch_detections)

    logger.info("Evaluating detections")
    output = evaluate_detections(all_boxes, data_loader.dataset)
    net.apply(restore_bn_module_mode)
    return output
