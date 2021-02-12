#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""training data generation functions."""
import numpy as np
import random, math
from PIL import Image
from tensorflow.keras.utils import Sequence
from common.data_utils import normalize_image, letterbox_resize, random_resize_crop_pad, reshape_boxes, random_hsv_distort, random_horizontal_flip, random_vertical_flip, random_grayscale, random_brightness, random_chroma, random_contrast, random_sharpness, random_blur, random_rotate, random_mosaic_augment # random_motion_blur
from common.utils import get_multiscale_list


def get_ground_truth_data(annotation_line, input_shape, augment=True, max_boxes=100):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    image_size = image.size
    model_input_size = tuple(reversed(input_shape))
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not augment:
        new_image, padding_size, offset = letterbox_resize(image, target_size=model_input_size, return_padding_info=True)
        image_data = np.array(new_image)
        image_data = normalize_image(image_data)

        # reshape boxes
        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size, offset=offset)
        if len(boxes)>max_boxes:
            boxes = boxes[:max_boxes]

        # fill in box data
        box_data = np.zeros((max_boxes,5))
        if len(boxes)>0:
            box_data[:len(boxes)] = boxes

        return image_data, box_data

    # random resize image and crop|padding to target size
    image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)

    # random horizontal flip image
    image, horizontal_flip = random_horizontal_flip(image)

    # random adjust brightness
    image = random_brightness(image)

    # random adjust color level
    image = random_chroma(image)

    # random adjust contrast
    image = random_contrast(image)

    # random adjust sharpness
    image = random_sharpness(image)

    # random convert image to grayscale
    image = random_grayscale(image)

    # random do normal blur to image
    #image = random_blur(image)

    # random do motion blur to image
    #image = random_motion_blur(image, prob=0.2)

    # random vertical flip image
    image, vertical_flip = random_vertical_flip(image)

    # random distort image in HSV color space
    # NOTE: will cost more time for preprocess
    #       and slow down training speed
    #image = random_hsv_distort(image)

    # reshape boxes based on augment
    boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size, padding_shape=padding_size, offset=padding_offset, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)

    # random rotate image and boxes
    image, boxes = random_rotate(image, boxes)

    if len(boxes)>max_boxes:
        boxes = boxes[:max_boxes]

    # prepare image & box data
    image_data = np.array(image)
    image_data = normalize_image(image_data)
    box_data = np.zeros((max_boxes,5))
    if len(boxes)>0:
        box_data[:len(boxes)] = boxes

    return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                         anchors for a single ground truth

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    #Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    #and image relative coordinate.
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Sort anchors according to IoU score
        # to find out best assignment
        best_anchors = np.argsort(iou, axis=-1)[..., ::-1]

        if not multi_anchor_assign:
            best_anchors = best_anchors[..., 0]
            # keep index dim for the loop in following
            best_anchors = np.expand_dims(best_anchors, -1)

        for t, row in enumerate(best_anchors):
            for l in range(num_layers):
                for n in row:
                    # use different matching policy for single & multi anchor assign
                    if multi_anchor_assign:
                        matching_rule = (iou[t, n] > iou_thresh and n in anchor_mask[l])
                    else:
                        matching_rule = (n in anchor_mask[l])

                    if matching_rule:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

    return y_true


class Yolo3DataGenerator(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, shuffle=True, **kwargs):
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.enhance_augment = enhance_augment
        self.multi_anchor_assign = multi_anchor_assign
        self.indexes = np.arange(len(self.annotation_lines))
        self.shuffle = shuffle
        # prepare multiscale config
        # TODO: error happens when using Sequence data generator with
        #       multiscale input shape, disable multiscale first
        if rescale_interval != -1:
            raise ValueError("tf.keras.Sequence generator doesn't support multiscale input, pls remove related config")
        #self.rescale_interval = rescale_interval
        self.rescale_interval = -1

        self.rescale_step = 0
        self.input_shape_list = get_multiscale_list()

    def __len__(self):
        # get iteration loops on each epoch
        return max(1, math.ceil(len(self.annotation_lines) / float(self.batch_size)))

    def __getitem__(self, index):
        # generate annotation indexes for every batch
        batch_indexs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        # fetch annotation lines based on index
        batch_annotation_lines = [self.annotation_lines[i] for i in batch_indexs]

        if self.rescale_interval > 0:
            # Do multi-scale training on different input shape
            self.rescale_step = (self.rescale_step + 1) % self.rescale_interval
            if self.rescale_step == 0:
                self.input_shape = self.input_shape_list[random.randint(0, len(self.input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(self.batch_size):
            image, box = get_ground_truth_data(batch_annotation_lines[b], self.input_shape, augment=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        if self.enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes, self.multi_anchor_assign)

        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_end(self):
        # shuffle annotation data on epoch end
        if self.shuffle == True:
            np.random.shuffle(self.annotation_lines)



def yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # prepare multiscale config
    rescale_step = 0
    input_shape_list = get_multiscale_list()
    while True:
        if rescale_interval > 0:
            # Do multi-scale training on different input shape
            rescale_step = (rescale_step + 1) % rescale_interval
            if rescale_step == 0:
                input_shape = input_shape_list[random.randint(0, len(input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment=True)
            # print('input_shape', input_shape)
            # print('annotation_lines[i]', annotation_lines[i])
            # print('image', image.shape)
            # print('box', box)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        if enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, multi_anchor_assign)

        # print('batch_size', batch_size)
        # print('image_data', len(image_data), image_data[0].shape)
        # print('y_true', len(y_true), y_true[2].shape)

        yield [image_data, *y_true], np.zeros(batch_size)

def yolo3_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign)

