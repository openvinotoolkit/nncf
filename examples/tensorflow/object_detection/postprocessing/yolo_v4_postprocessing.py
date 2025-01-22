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

import copy

import numpy as np
from scipy.special import expit
from scipy.special import softmax


def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    """Decode final layer features to bounding box parameters."""
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_size = np.shape(prediction)[1:3]

    prediction = np.reshape(prediction, (batch_size, grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # generate x_y_offset grid map
    grid_y = np.arange(grid_size[0])
    grid_x = np.arange(grid_size[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
    anchors = np.expand_dims(anchors, 0)

    if scale_x_y:
        # Eliminate grid sensitivity trick involved in YOLOv4
        #
        # Reference Paper & code:
        #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
        #     https://arxiv.org/abs/2004.10934
        #     https://github.com/opencv/opencv/issues/17148
        #
        box_xy_tmp = expit(prediction[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_size)[::-1]
    else:
        box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

    # Sigmoid objectness scores
    objectness = expit(prediction[..., 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

    if use_softmax:
        # Softmax class scores
        class_scores = softmax(prediction[..., 5:], axis=-1)
    else:
        # Sigmoid class scores
        class_scores = expit(prediction[..., 5:])

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo3_decode(predictions, anchors, num_classes, input_dims, elim_grid_sense=False):
    """
    YOLOv3 Head to process predictions from YOLOv3 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]

    results = []
    for i, prediction in enumerate(predictions):
        results.append(
            yolo_decode(
                prediction, anchors[anchor_mask[i]], num_classes, input_dims, scale_x_y=scale_x_y[i], use_softmax=False
            )
        )

    return np.concatenate(results, axis=1)


def yolo_correct_boxes(predictions, img_shape, model_image_size):
    """Rescale predicition boxes back to original image shape"""
    box_xy = predictions[..., :2]
    box_wh = predictions[..., 2:4]
    objectness = np.expand_dims(predictions[..., 4], -1)
    class_scores = predictions[..., 5:]

    # model_image_size & image_shape should be (height, width) format
    model_image_size = np.array(model_image_size, dtype="float32")
    image_shape = np.array(img_shape, dtype="float32")

    new_shape = np.round(image_shape * np.min(model_image_size / image_shape))
    offset = (model_image_size - new_shape) / 2.0 / model_image_size
    scale = model_image_size / new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def box_iou(boxes):
    """
    Calculate IoU value of 1st box with other boxes of a box array

    :param boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates
    :return iou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)
    return iou


def box_diou(boxes):
    """
    Calculate DIoU value of 1st box with other boxes of a box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    :param boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates
    :return diou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)

    # box center distance
    x_center = x + w / 2
    y_center = y + h / 2
    center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(y_center[1:] - y_center[0], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[1:], x[0])
    enclose_ymin = np.minimum(y[1:], y[0])
    enclose_xmax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_ymax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

    return diou


def box_iou_matrix(boxes1, boxes2):
    """
    Calculate IoU matrix for two box array.
    Both sets of boxes are expected to be in (x, y, w, h) format.
    Reference implementation:
        https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    :param boxes1: (np.array[N, 4])
    :param boxes2: (np.array[M, 4])
    :return iou: (np.array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xN
        return box[2] * box[3]

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    inter_min = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    inter_max = np.minimum(boxes1[:, None, :2] + boxes1[:, None, 2:], boxes2[:, :2] + boxes2[:, 2:])  # [N,M,2]
    inter = np.maximum(inter_max - inter_min, 0).prod(axis=-1)  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    return iou


def box_diou_matrix(boxes1, boxes2):
    """
    Calculate DIoU matrix for two box array.
    Both sets of boxes are expected to be in (x, y, w, h) format.

    :param boxes1: (np.array[N, 4])
    :param boxes2: (np.array[M, 4])
    :return diou: (np.array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    iou = box_iou_matrix(boxes1, boxes2)

    # box center distance
    center_distance = (boxes1[:, None, :2] + boxes1[:, None, 2:] / 2) - (boxes2[:, :2] + boxes2[:, 2:] / 2)  # [N,M,2]
    center_distance = np.power(center_distance[..., 0], 2) + np.power(center_distance[..., 1], 2)  # [N,M]

    # get enclosed area
    enclose_min = np.minimum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    enclose_max = np.maximum(boxes1[:, None, :2] + boxes1[:, None, 2:], boxes2[:, :2] + boxes2[:, 2:])  # [N,M,2]

    enclose_wh = np.maximum(enclose_max - enclose_min, 0)  # [N,M,2]
    enclose_wh = np.maximum(enclose_max - enclose_min, 0)  # [N,M,2]

    # get enclosed diagonal distance matrix
    enclose_diagonal = np.power(enclose_wh[..., 0], 2) + np.power(enclose_wh[..., 1], 2)  # [N,M]

    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * np.true_divide(center_distance, enclose_diagonal + np.finfo(float).eps)

    return diou


def fast_cluster_nms_boxes(boxes, classes, scores, iou_threshold):
    """
    Fast NMS/Cluster NMS/Matrix NMS bbox post process
    Reference Paper:
        1. "YOLACT: Real-time Instance Segmentation"
           https://arxiv.org/abs/1904.02689

        2. "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation"
           https://arxiv.org/abs/2005.03572

        3. "SOLOv2: Dynamic, Faster and Stronger"
           https://arxiv.org/abs/2003.10152

        4. Blogpost on zhihu:
           https://zhuanlan.zhihu.com/p/157900024

    :param boxes:   bbox numpy array, shape=(N, 4), xywh
             x,y are top left coordinates
    :param classes: bbox class index numpy array, shape=(N, 1)
    :param scores:  bbox score numpy array, shape=(N, 1)
    :param iou_threshold:
    :return nboxes:   NMSed bbox numpy array, shape=(N, 4), xywh
              x,y are top left coordinates
    :return nclasses: NMSed bbox class index numpy array, shape=(N, 1)
    :return nscores:  NMSed bbox score numpy array, shape=(N, 1)
    """
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        # ascend sort boxes according to scores
        sorted_indices = np.argsort(s_nms)
        sorted_indices = sorted_indices[::-1]
        b_nms = b_nms[sorted_indices]
        c_nms = c_nms[sorted_indices]
        s_nms = s_nms[sorted_indices]

        # number of boxes for one class
        num_boxes = b_nms.shape[0]

        # get IoU/DIoU matrix (upper triangular matrix)
        iou_matrix = box_diou_matrix(b_nms, b_nms)

        iou_matrix = np.triu(iou_matrix, k=1)
        max_iou = np.max(iou_matrix, axis=0)
        updated_iou_matrix = copy.deepcopy(iou_matrix)

        # Cluster loop
        for _ in range(200):
            prev_iou_matrix = copy.deepcopy(updated_iou_matrix)
            max_iou = np.max(prev_iou_matrix, axis=0)
            keep_diag = np.diag((max_iou < iou_threshold).astype(np.float32))
            updated_iou_matrix = np.dot(keep_diag, iou_matrix)
            if (prev_iou_matrix == updated_iou_matrix).all():
                break

        # filter low score box with iou_threshold
        keep_mask = max_iou < iou_threshold

        # generate weights matrix with box score and final IoU matrix
        weights = (updated_iou_matrix * (updated_iou_matrix > iou_threshold).astype(np.float32) + np.eye(num_boxes)) * (
            s_nms.reshape((1, num_boxes))
        )

        # convert box format to (xmin,ymin,xmax,ymax) for weighted average,
        # and expand to NxN array
        xmin_expand = np.tile(b_nms[:, 0], (num_boxes, 1))  # (num_boxes)x(num_boxes)
        ymin_expand = np.tile(b_nms[:, 1], (num_boxes, 1))  # (num_boxes)x(num_boxes)
        xmax_expand = np.tile(b_nms[:, 0] + b_nms[:, 2], (num_boxes, 1))  # (num_boxes)x(num_boxes)
        ymax_expand = np.tile(b_nms[:, 1] + b_nms[:, 3], (num_boxes, 1))  # (num_boxes)x(num_boxes)

        # apply weighted average to all the candidate boxes
        weightsum = weights.sum(axis=1)
        xmin_expand = np.true_divide((xmin_expand * weights).sum(axis=1), weightsum)
        ymin_expand = np.true_divide((ymin_expand * weights).sum(axis=1), weightsum)
        xmax_expand = np.true_divide((xmax_expand * weights).sum(axis=1), weightsum)
        ymax_expand = np.true_divide((ymax_expand * weights).sum(axis=1), weightsum)

        # stack the weighted average boxes and convert back to (x,y,w,h)
        b_nms = np.stack([xmin_expand, ymin_expand, xmax_expand - xmin_expand, ymax_expand - ymin_expand], axis=1)

        # keep NMSed boxes
        b_nms = b_nms[keep_mask]
        c_nms = c_nms[keep_mask]
        s_nms = s_nms[keep_mask]

        # merge NMSed boxes to final result
        if len(nboxes) == 0:
            nboxes = np.asarray(copy.deepcopy(b_nms))
            nclasses = np.asarray(copy.deepcopy(c_nms))
            nscores = np.asarray(copy.deepcopy(s_nms))
        else:
            nboxes = np.append(nboxes, copy.deepcopy(b_nms), axis=0)
            nclasses = np.append(nclasses, copy.deepcopy(c_nms), axis=0)
            nscores = np.append(nscores, copy.deepcopy(s_nms), axis=0)

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def nms_boxes(boxes, classes, scores, iou_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and first line
            b_nms[[i, 0], :] = b_nms[[0, i], :]
            c_nms[[i, 0]] = c_nms[[0, i]]
            s_nms[[i, 0]] = s_nms[[0, i]]

            iou = box_diou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]

            # normal Hard-NMS
            keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def filter_boxes(boxes, classes, scores, max_boxes):
    """
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    """
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def yolo_handle_predictions(predictions, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    # filter boxes with confidence threshold
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # Boxes, Classes and Scores returned from NMS
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold)

    boxes, classes, scores = [], [], []
    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype("int32")
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

    return boxes, classes, scores


def yolo_adjust_boxes(boxes, img_shape):
    """
    change box format from (x,y,w,h) top left coordinate to
    (xmin, ymin, xmax, ymax) format
    """
    if boxes is None or len(boxes) == 0:
        return np.array([])

    image_shape = np.array(img_shape, dtype="float32")
    height, width = image_shape

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype("int32"))
        xmin = max(0, np.floor(xmin + 0.5).astype("int32"))
        ymax = min(height, np.floor(ymax + 0.5).astype("int32"))
        xmax = min(width, np.floor(xmax + 0.5).astype("int32"))
        adjusted_boxes.append([xmin, ymin, xmax, ymax])

    return np.array(adjusted_boxes, dtype=np.int32)


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path, encoding="utf8") as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1, 2)


def yolo3_postprocess_np(
    yolo_outputs,
    image_shape,
    anchors,
    num_classes,
    model_image_size,
    max_boxes=100,
    confidence=0.1,
    iou_threshold=0.4,
    elim_grid_sense=False,
):
    predictions = yolo3_decode(
        yolo_outputs, anchors, num_classes, input_dims=model_image_size, elim_grid_sense=elim_grid_sense
    )
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = yolo_handle_predictions(
        predictions, max_boxes=max_boxes, confidence=confidence, iou_threshold=iou_threshold
    )
    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores


def convert_coco_category(category_id):
    """
    Convert continuous coco class id to discontinuous coco category id (0..79 --> 0..90)
    """
    match = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    category_id = match[category_id]
    return category_id


def convert_coordinate(box):
    xmin, ymin, xmax, ymax = box
    return [ymin, xmin, ymax, xmax]


def postprocess_yolo_v4_np(
    image_info, out1, out2, out3, anchors, num_classes, input_shape, conf_threshold, elim_grid_sense
):
    image_info = image_info.numpy()
    out1 = out1.numpy()
    out2 = out2.numpy()
    out3 = out3.numpy()
    anchors = anchors.numpy()
    num_classes = num_classes.numpy()
    input_shape = input_shape.numpy()
    conf_threshold = conf_threshold.numpy()
    elim_grid_sense = elim_grid_sense.numpy()

    anchors = np.array(anchors).reshape(-1, 2)

    batch_size = image_info.shape[0]
    boxes = []
    classes = []
    scores = []
    valid_detections = []
    for i in range(batch_size):
        prediction = [out1[i][None, ...], out2[i][None, ...], out3[i][None, ...]]
        image_shape = image_info[i][0]
        pred_boxes, pred_classes, pred_scores = yolo3_postprocess_np(
            prediction,
            image_shape,
            anchors,
            num_classes,
            input_shape,
            max_boxes=100,
            confidence=conf_threshold,
            elim_grid_sense=elim_grid_sense,
        )

        num_detections = pred_boxes.shape[0]
        pred_boxes = np.array(list(map(convert_coordinate, pred_boxes)))
        pred_classes = np.array(list(map(convert_coco_category, pred_classes)))

        if 0 < num_detections < 100:
            pred_boxes = np.vstack((pred_boxes, np.zeros((100 - num_detections, 4))))
            pred_classes = np.hstack((pred_classes, np.ones(100 - num_detections)))
            pred_scores = np.hstack((pred_scores, np.zeros(100 - num_detections)))
        elif num_detections == 0:
            pred_boxes = np.zeros((100, 4))
            pred_classes = np.ones(100)
            pred_scores = np.zeros(100)

        boxes.append(pred_boxes)
        classes.append(pred_classes)
        scores.append(pred_scores)
        valid_detections.append(num_detections)

    boxes = np.stack(boxes, axis=0)
    classes = np.stack(classes, axis=0)
    scores = np.stack(scores, axis=0)
    valid_detections = np.array(valid_detections)

    return boxes, classes, scores, valid_detections
