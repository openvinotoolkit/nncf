"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
from PIL import Image

import tensorflow as tf
from beta.examples.tensorflow.common.object_detection.utils import box_utils
from beta.examples.tensorflow.common.object_detection.utils.yolo_v4_utils import normalize_image, letterbox_resize, random_resize_crop_pad, reshape_boxes, random_hsv_distort, random_horizontal_flip, random_vertical_flip, random_grayscale, random_brightness, random_chroma, random_contrast, random_sharpness, random_blur, random_rotate, random_mosaic_augment, random_cutmix_augment, rand # random_motion_blur
from beta.examples.tensorflow.common.object_detection.utils import dataloader_utils
from beta.examples.tensorflow.common.object_detection.utils import input_utils


class YOLOv4Preprocessor:
    """Parser to parse an image and its annotations into a dictionary of tensors."""
    def __init__(self, config, is_train):
        """Initializes parameters for parsing annotations in the dataset.
        """

        self._max_num_instances = config.preprocessing.get('max_num_instances', 100)
        self._is_training = is_train
        self._global_batch_size = config.batch_size
        self._num_preprocess_workers = config.get('workers', tf.data.experimental.AUTOTUNE)

        self._parse_fn = self._parse_train_data
        # self._parse_fn = self._parse_train_data_tf
        self._parse_fn2 = self._parse_train_data2

        self.input_shape = config.input_shape
        self.enhance_mosaic_augment = config.preprocessing.enhance_mosaic_augment
        self.enhance_cutmix_augment = config.preprocessing.enhance_cutmix_augment
        self.anchors = config.anchors
        self.num_classes = config.model_params.num_classes
        self.multi_anchor_assign = config.preprocessing.multi_anchor_assign

        # TF preprocessing
        self._output_size = config.input_info.sample_size[1:3]
        self._aug_rand_hflip = config.preprocessing.get('aug_rand_hflip', False)
        self._aug_scale_min = config.preprocessing.get('aug_scale_min', 1.0)
        self._aug_scale_max = config.preprocessing.get('aug_scale_max', 1.0)
    def create_preprocess_input_fn(self):
        """Parses data to an image and associated training labels.
        """
        return self._tfds_decoder, self._pipeline_fn


    # https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
    def _get_ground_truth_data(self, image, boxes, input_shape, filename, max_boxes=100):
        '''random preprocessing for real-time data augmentation'''
        # line = annotation_line.split()
        # image = Image.open(line[0]) # PIL Image object containing image data
        image_size = image.size
        model_input_size = tuple(reversed(input_shape))
        # boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # random resize image and crop|padding to target size
        image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)

        # random horizontal flip image
        image, horizontal_flip = random_horizontal_flip(image)

        # # random vertical flip image
        # image, vertical_flip = random_vertical_flip(image)

        # reshape boxes based on augment
        boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size,
                              padding_shape=padding_size, offset=padding_offset,
                              horizontal_flip=horizontal_flip) # , vertical_flip=vertical_flip

        # # random rotate image and boxes
        # image, boxes = random_rotate(image, boxes)

        # boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size,
        #                       padding_shape=padding_size, offset=padding_offset)

        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]

        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            box_data[:len(boxes)] = boxes


        # # the most computationally expensive ops
        image_data = np.array(image).astype(np.float32)
        # image_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
        # image = input_utils.random_adjust_brightness(image_tensor)
        # image = input_utils.random_adjust_contrast(image)
        # image = input_utils.random_adjust_hue(image)
        # image = input_utils.random_adjust_saturation(image)
        # image_data = tf.math.divide(image, 255.0)

        # # random adjust brightness
        # image = random_brightness(image)
        # # random adjust color level
        # image = random_chroma(image)
        # # random adjust contrast
        # image = random_contrast(image)
        # # random adjust sharpness
        # image = random_sharpness(image)
        # # random convert image to grayscale
        # image = random_grayscale(image)
        # # prepare image data
        # image_data = np.array(image).astype(np.float32)
        # image_data = normalize_image(image_data)

        return image_data, box_data
    def _preprocess(self, image, filename, groundtruth_classes, groundtruth_boxes, input_shape):

        image_np = image.numpy()
        image_pil = Image.fromarray(image_np)

        image_shape = tf.shape(input=image)[0:2]
        denormalized_boxes = box_utils.denormalize_boxes(groundtruth_boxes, image_shape)

        boxes = []
        for denormalized_box, category_id in zip(denormalized_boxes.numpy(), groundtruth_classes.numpy()):
            x_min = int(denormalized_box[1])
            y_min = int(denormalized_box[0])
            x_max = int(denormalized_box[3])
            y_max = int(denormalized_box[2])
            boxes.append([x_min, y_min, x_max, y_max, int(category_id)])
        boxes = np.array(boxes)

        input_shape = input_shape.numpy()
        image, box = self._get_ground_truth_data(image_pil, boxes, input_shape, filename)

        return image, box
    def _parse_train_data(self, data):
        """Parses data for training"""
        image = data['image']
        filename = data['source_filename']
        groundtruth_classes = data['groundtruth_classes']
        groundtruth_boxes = data['groundtruth_boxes']

        image, box = tf.py_function(self._preprocess, [image, filename, groundtruth_classes, groundtruth_boxes, self.input_shape], [tf.float32, tf.float32])
        image.set_shape([None, None, 3])
        box.set_shape([None, 5])

        image = input_utils.random_adjust_brightness(image)
        image = input_utils.random_adjust_contrast(image)
        image = input_utils.random_adjust_hue(image)
        image = input_utils.random_adjust_saturation(image)
        image = tf.math.divide(image, 255.0)

        # # Random input
        # image = tf.random.uniform([608, 608, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)
        # box = tf.zeros([20, 5], dtype=tf.float64)

        out = {}
        out['image'] = image
        out['box'] = box
        out['filename'] = filename
        out['source_id'] = data['source_id']

        return out
    def _parse_train_data_tf(self, data):
        """Parses data for training"""
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        filename = data['source_filename']

        # Gets original image and its size.
        image = data['image']
        image_shape = tf.shape(input=image)[0:2]

        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(boxes, image_shape)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=self._output_size,
            aug_scale_min=self._aug_scale_min,
            aug_scale_max=self._aug_scale_max)
        image_height, image_width, _ = image.get_shape().as_list()

        # Resizes and crops boxes.
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)

        # Flips image randomly during training.
        image, boxes = input_utils.random_horizontal_flip(image, boxes) # pylint: disable=W0632
        image, boxes = input_utils.random_vertical_flip(image, boxes)
        # image, boxes = input_utils.random_rotation90(image, boxes)

        # Filters out ground truth boxes that are all zeros.
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices, axis=None)
        classes = tf.gather(classes, indices, axis=None)

        # Rearrange boxes and add classes to boxes
        # convert to x_min, y_min, x_max, y_max
        boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
        classes = tf.expand_dims(tf.cast(classes, tf.float32), 1)
        boxes_classes = tf.concat([boxes, classes], axis=1)

        boxes_classes = input_utils.pad_to_fixed_size(boxes_classes, self._max_num_instances, -1)

        image = input_utils.random_adjust_brightness(image)
        image = input_utils.random_adjust_contrast(image)
        image = input_utils.random_adjust_hue(image)
        image = input_utils.random_adjust_saturation(image)

        # Normalizes images
        image = tf.math.divide(image, 255.0)

        out = {}
        out['image'] = image
        out['box'] = boxes_classes
        out['filename'] = filename
        out['source_id'] = data['source_id']

        return out


    def _preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
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
            iou = intersect_area / (box_area + anchor_area - intersect_area + 0.0000001)

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
    def _preprocess2(self, image_data, box_data, filename):

        image_data = image_data.numpy()
        box_data = box_data.numpy()

        # if self.enhance_augment == 'mosaic':
        #     # add random mosaic augment on batch ground truth data
        #     image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)
        #
        #     random_val = rand()
        #     if random_val < 0.2:
        #         image_data, box_data = random_mosaic_augment(image_data, box_data, prob=1.0)
        #     elif 0.2 < random_val < 0.3:
        #         image_data, box_data = random_cutmix_augment(image_data, box_data, prob=1.0)

        if self.enhance_mosaic_augment and not self.enhance_cutmix_augment:
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)
        elif self.enhance_cutmix_augment and not self.enhance_mosaic_augment:
            image_data, box_data = random_cutmix_augment(image_data, box_data, prob=0.2)
        elif self.enhance_mosaic_augment and self.enhance_cutmix_augment:
            random_val = rand()
            if random_val < 0.2:
                image_data, box_data = random_mosaic_augment(image_data, box_data, prob=1.0)
            elif 0.2 < random_val < 0.3:
                image_data, box_data = random_cutmix_augment(image_data, box_data, prob=1.0)

        anchors = np.array(self.anchors).astype(float).reshape(-1, 2)
        y_true1, y_true2, y_true3 = self._preprocess_true_boxes(box_data, self.input_shape, anchors, self.num_classes, self.multi_anchor_assign)

        # image_data = tf.convert_to_tensor(image_data, dtype=tf.float64)
        # y_true1 = tf.convert_to_tensor(y_true1, dtype=tf.float32)
        # y_true2 = tf.convert_to_tensor(y_true2, dtype=tf.float32)
        # y_true3 = tf.convert_to_tensor(y_true3, dtype=tf.float32)

        return image_data, y_true1, y_true2, y_true3
    def _preprocess2_tf(self, image_data, true_boxes, iou_thresh=0.2):
        anchors = tf.constant(self.anchors, dtype=tf.float32)
        anchors = tf.reshape(anchors, [-1, 2])
        num_layers = len(anchors)//3 # default setting
        anchor_mask = tf.constant([[6,7,8], [3,4,5], [0,1,2]], dtype=tf.int32)

        #Transform box info to (x_center, y_center, box_width, box_height, cls_id)
        #and image relative coordinate.
        true_boxes = tf.cast(true_boxes, dtype=tf.float32)
        input_shape = tf.constant(self.input_shape, dtype=tf.float32)
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        boxes01 = boxes_xy / input_shape[::-1]
        boxes23 = boxes_wh / input_shape[::-1]
        true_boxes = tf.concat([boxes01, boxes23, tf.expand_dims(true_boxes[:, :, -1], 2)], axis=2)

        batch_size = true_boxes.shape[0]
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        grid_shapes = tf.stack(grid_shapes, axis=0)
        y_true = [tf.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes),
            dtype=tf.float32) for l in range(num_layers)]

        # list_of_indices = [[], [], []]
        # list_of_values = [[], [], []]

        # import copy
        # def zeros_list(shape):
        #     if len(shape) == 1:
        #         return [[0] for i in range(shape[0])]
        #     items = shape[0]
        #     newshape = shape[1:]
        #     sublist = zeros_list(newshape)
        #     return [copy.deepcopy(sublist) for i in range(items)]
        #
        # print('creating zeros_list')
        # # y_true = [zeros_list([batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes]) for l in range(num_layers)]
        # y_true = [zeros_list([batch_size, 19, 19, 3, 85]), zeros_list([batch_size, 38, 38, 3, 85]), zeros_list([batch_size, 76, 76, 3, 85])]
        # print('completed zeros_list')

        # Expand dim to apply broadcasting.
        anchors = tf.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(batch_size):
            # Discard zero rows.
            wh = tf.boolean_mask(boxes_wh[b], valid_mask[b])

            if len(wh) == 0:
                continue

            # Expand dim to apply broadcasting.
            wh = tf.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = tf.math.maximum(box_mins, anchor_mins)
            intersect_maxes = tf.math.minimum(box_maxes, anchor_maxes)
            intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area + 0.0000001)

            # Sort anchors according to IoU score
            # to find out best assignment
            best_anchors = tf.argsort(iou, axis=-1)[..., ::-1]

            if not self.multi_anchor_assign:
                best_anchors = best_anchors[..., 0]
                # keep index dim for the loop in following
                best_anchors = tf.expand_dims(best_anchors, -1)

            # for t, row in enumerate(best_anchors):
            for t in range(len(best_anchors)):
                row = best_anchors[t]
                for l in range(num_layers):
                    list_of_indices = []
                    list_of_values = []
                    for n in row:
                        # use different matching policy for single & multi anchor assign
                        if self.multi_anchor_assign:
                            iou_check = tf.cond(
                                pred=tf.equal(iou[t, n] > iou_thresh, tf.constant(True)),
                                true_fn=lambda: True,
                                false_fn=lambda: False
                            )
                            # (iou_check and n in anchor_mask[l])
                            n_in_anchor_mask = tf.reduce_any(tf.equal(anchor_mask[l], n))
                            matching_rule = tf.math.logical_and(iou_check, n_in_anchor_mask)
                        else:
                            matching_rule = (n in anchor_mask[l])

                        if matching_rule:
                            i = tf.cast(tf.math.floor(true_boxes[b, t, 0] * grid_shapes[l][1]), dtype=tf.int32)
                            j = tf.cast(tf.math.floor(true_boxes[b, t, 1] * grid_shapes[l][0]), dtype=tf.int32)
                            k = tf.squeeze(tf.where(tf.equal(anchor_mask[l], n)))
                            k = tf.cast(k, dtype=tf.int32)
                            c = tf.cast(true_boxes[b, t, 4], dtype=tf.int32)

                            for g in range(4):
                                list_of_indices.append([b, j, i, k, g])
                                list_of_values.append(true_boxes[b, t, g])

                            list_of_indices.append([b, j, i, k, 4])
                            list_of_values.append(tf.cast(1, tf.float32))

                            list_of_indices.append([b, j, i, k, 5+c])
                            list_of_values.append(tf.cast(1, tf.float32))

                    if len(list_of_indices) > 0:
                        # print('list_of_values', list_of_values)
                        delta = tf.SparseTensor(list_of_indices, list_of_values, y_true[l].shape)
                        delta = tf.sparse.reorder(delta)
                        y_true[l] = y_true[l] + tf.sparse.to_dense(delta)


            # for t, row in enumerate(best_anchors):
            #     for l in range(num_layers):
            #         # list_of_indices = []
            #         # list_of_values = []
            #         for n in row:
            #             # use different matching policy for single & multi anchor assign
            #             if self.multi_anchor_assign:
            #                 iou_check = tf.cond(
            #                     pred=tf.equal(iou[t, n] > iou_thresh, tf.constant(True)),
            #                     true_fn=lambda: True,
            #                     false_fn=lambda: False
            #                 )
            #                 matching_rule = (iou_check and n in anchor_mask[l])
            #             else:
            #                 matching_rule = (n in anchor_mask[l])
            #
            #             if matching_rule:
            #                 i = tf.cast(tf.math.floor(true_boxes[b, t, 0] * grid_shapes[l][1]), dtype=tf.int32)
            #                 j = tf.cast(tf.math.floor(true_boxes[b, t, 1] * grid_shapes[l][0]), dtype=tf.int32)
            #                 k = tf.squeeze(tf.where(tf.equal(anchor_mask[l], n)))
            #                 k = tf.cast(k, dtype=tf.int32)
            #                 c = tf.cast(true_boxes[b, t, 4], dtype=tf.int32)
            #
            #                 if [b, j, i, k, 0] not in list_of_indices[l]:
            #
            #                     for g in range(4):
            #                         list_of_indices[l].append([b, j, i, k, g])
            #                         list_of_values[l].append(true_boxes[b, t, g])
            #
            #                     list_of_indices[l].append([b, j, i, k, 4])
            #                     list_of_values[l].append(tf.cast(1, tf.float32))
            #
            #                     list_of_indices[l].append([b, j, i, k, 5 + c])
            #                     list_of_values[l].append(tf.cast(1, tf.float32))
            #
            # for l in range(num_layers):
            #     if len(list_of_indices[l]) > 0:
            #         # print('list_of_indices[l]', [np.array(item) for item in  list_of_indices[l]])
            #         # print('\nlist_of_values[l]', [np.array(item) for item in list_of_values[l]])
            #         delta = tf.SparseTensor(list_of_indices[l], list_of_values[l], y_true[l].shape)
            #         delta = tf.sparse.reorder(delta)
            #         y_true[l] = y_true[l] + tf.sparse.to_dense(delta)
        return y_true
    def _preprocess2_test(self, image_data, box_data):
        # image_data = image_data *1
        # box_data = box_data * 2
        image_data = image_data.numpy()
        box_data = box_data.numpy()

        anchors = np.array(self.anchors).astype(float).reshape(-1, 2)
        y_true1, y_true2, y_true3 = self._preprocess_true_boxes(box_data, self.input_shape, anchors, self.num_classes, self.multi_anchor_assign)

        return image_data, y_true1, y_true2, y_true3
    def _parse_train_data2(self, data):

        image_data = data['image']
        box_data = data['box']
        filename = data['filename']
        im_shape = image_data.shape

        image_data, out0, out1, out2 = tf.py_function(self._preprocess2, [image_data, box_data, filename], [tf.float32, tf.float32, tf.float32, tf.float32]) # , tf.float32
        image_data.set_shape(im_shape)
        out0.set_shape([im_shape[0], 19, 19, 3, 85])
        out1.set_shape([im_shape[0], 38, 38, 3, 85])
        out2.set_shape([im_shape[0], 76, 76, 3, 85])

        # # Test of tf.py_function: tensor -> numpy -> tensor
        # # box_data_shape = box_data.shape
        # image_data, out0, out1, out2 = tf.py_function(self._preprocess2_test, [image_data, box_data], [tf.float32, tf.float32, tf.float32, tf.float32])
        # image_data.set_shape(im_shape)
        # out0.set_shape([im_shape[0], 19, 19, 3, 85])
        # out1.set_shape([im_shape[0], 38, 38, 3, 85])
        # out2.set_shape([im_shape[0], 76, 76, 3, 85])

        # # Random input
        # out0 = tf.zeros([im_shape[0], 19, 19, 3, 85], dtype=tf.float32)
        # out1 = tf.zeros([im_shape[0], 38, 38, 3, 85], dtype=tf.float32)
        # out2 = tf.zeros([im_shape[0], 76, 76, 3, 85], dtype=tf.float32)

        # out = {}
        # out['image_input'] = image_data
        # out['y_true_0'] = out0
        # out['y_true_1'] = out1
        # out['y_true_2'] = out2
        # out['filename'] = filename
        # return out, tf.zeros(64, dtype=tf.dtypes.float32)

        labels = {
            'y_true_0': out0,
            'y_true_1': out1,
            'y_true_2': out2,
        }
        return image_data, labels








    def _tfds_decoder(self, features_dict):

        def _decode_image(features):
            image = tf.image.decode_jpeg(features['image'], channels=3, dct_method='INTEGER_ACCURATE')
            image.set_shape([None, None, 3])
            return image

        def _convert_labels_to_91_classes(features):
            # 0..79 --> 0..90
            match = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                                 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                                 80, 81, 82, 84, 85, 86, 87, 88, 89, 90], dtype=tf.int64)

            labels = features['objects']['label']
            labels = tf.gather(match, labels, axis=None)
            return labels

        image = _decode_image(features_dict)
        if self._is_training:
            labels = features_dict['objects']['label']
        else:
            labels = _convert_labels_to_91_classes(features_dict)

        decoded_tensors = {
            'image': image,
            'source_filename': features_dict['image/filename'],
            'source_id': tf.cast(features_dict['image/id'], tf.int32),
            'groundtruth_classes': labels,
            'groundtruth_is_crowd': features_dict['objects']['is_crowd'],
            'groundtruth_area': features_dict['objects']['area'],
            'groundtruth_boxes': features_dict['objects']['bbox'],
        }

        return decoded_tensors
    def _pipeline_fn(self, dataset, decoder_fn):

        if self._is_training:
            preprocess_input_fn = self._parse_fn
            preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
            dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers) # self._num_preprocess_workers , deterministic=False
            dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

            # part of preprocessing which requires batches
            preprocess_input_fn2 = self._parse_fn2
            preprocess_pipeline2 = lambda record: preprocess_input_fn2(record)
            dataset = dataset.map(preprocess_pipeline2, num_parallel_calls=self._num_preprocess_workers) # self._num_preprocess_workers

            # # test
            # # remove preprocessing completly
            # preprocess_input_fn = self._parse_train_data_dummy
            # preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
            # dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers) # self._num_preprocess_workers
            # dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

        else:
            preprocess_input_fn = self._parse_predict_data
            preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
            dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers) # self._num_preprocess_workers
            dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

        return dataset










    def get_image_info(self, image):

        desired_size = tf.convert_to_tensor(self.input_shape, dtype=tf.float32)

        image_size = tf.cast(tf.shape(input=image)[0:2], tf.float32)
        scaled_size = desired_size

        scale = tf.minimum(scaled_size[0] / image_size[0],
                          scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        offset = tf.zeros((2,), tf.int32)

        image_info = tf.stack([
            image_size,
            tf.cast(desired_size, tf.float32), image_scale,
            tf.cast(offset, tf.float32)
        ])
        return image_info
    def _preprocess_predict_image(self, image):
        image = image.numpy()
        model_image_size = self.input_shape
        image_pil = Image.fromarray(image)
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        resized_image = letterbox_resize(image_pil, tuple(reversed(model_image_size)))
        image_data = np.asarray(resized_image).astype('float32')
        image_data = normalize_image(image_data)
        image_data = tf.convert_to_tensor(image_data, dtype=tf.float32)
        return image_data
    def _parse_predict_data(self, data):
        """Parses data for prediction"""
        image_data = data['image']
        image_shape = tf.shape(input=image_data)[0:2]

        # filename = data['source_filename']
        # groundtruth_classes = data['groundtruth_classes']
        # groundtruth_boxes = data['groundtruth_boxes']

        # needed only for eval
        image_info = self.get_image_info(image_data)

        # image preprocessing
        # print('image_data', image_data.shape, type(image_data))
        image_data = tf.py_function(self._preprocess_predict_image, [image_data], Tout=tf.float32)
        # print('image_data', type(image_data))
        image_data.set_shape([None, None, 3])

        labels = {
            'image_info': image_info,
            # 'source_id': data['source_id']
        }

        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape)
        groundtruths = {
            'source_id': data['source_id'],
            'num_detections': tf.squeeze(tf.shape(data['groundtruth_classes'])),
            'boxes': boxes,
            'classes': data['groundtruth_classes'],
            'areas': data['groundtruth_area'],
            'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
        }
        groundtruths['source_id'] = dataloader_utils.process_source_id(groundtruths['source_id'])
        groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
        labels.update(groundtruths)

        return image_data, labels





    def _parse_train_data_dummy(self, data):
        """Parses data for training"""

        image = tf.random.uniform([608, 608, 3], minval=0, maxval=1, dtype=tf.dtypes.float32)

        # out0 = tf.random.uniform([19, 19, 3, 85], minval=0, maxval=1, dtype=tf.dtypes.float32)
        # out1 = tf.random.uniform([38, 38, 3, 85], minval=0, maxval=1, dtype=tf.dtypes.float32)
        # out2 = tf.random.uniform([76, 76, 3, 85], minval=0, maxval=1, dtype=tf.dtypes.float32)

        out0 = tf.zeros([19, 19, 3, 85], dtype=tf.float32)
        out1 = tf.zeros([38, 38, 3, 85], dtype=tf.float32)
        out2 = tf.zeros([76, 76, 3, 85], dtype=tf.float32)

        labels = {
            'y_true_0': out0,
            'y_true_1': out1,
            'y_true_2': out2,
        }

        return image, labels









def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)
    is_train = True
    preprocessor = YOLOv4Preprocessor(config, is_train)
    preprocessor.enhance_cutmix_augment = False
    preprocessor.enhance_mosaic_augment = False

    image_data = tf.random.uniform([9, 608, 608, 3], minval=0, maxval=1, dtype=tf.float32)
    box_data = tf.random.uniform([9, 100, 5], minval=0, maxval=80, dtype=tf.float32)
    filename = "111"

    print('Old preprocessing')
    im_shape = image_data.shape
    image_data, out0, out1, out2 = tf.py_function(preprocessor._preprocess2, [image_data, box_data, filename],
                                                  [tf.float64, tf.float32, tf.float32, tf.float32])
    image_data.set_shape(im_shape)
    out0.set_shape([im_shape[0], 19, 19, 3, 85])
    out1.set_shape([im_shape[0], 38, 38, 3, 85])
    out2.set_shape([im_shape[0], 76, 76, 3, 85])

    print('New preprocessing')
    # out0_tf, out1_tf, out2_tf = preprocessor._preprocess2_tf(image_data, box_data)
    preprocess_with_graph = tf.function(preprocessor._preprocess2_tf)
    out0_tf, out1_tf, out2_tf = preprocess_with_graph(image_data, box_data)

    assert tf.math.reduce_all(out0.numpy() == out0_tf.numpy())


if __name__ == "__main__":
    print('main')
    from beta.examples.tensorflow.object_detection.main import get_argument_parser, get_config_from_argv
    import sys
    main(sys.argv[1:])