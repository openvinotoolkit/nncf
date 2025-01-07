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

import numpy as np
import tensorflow as tf
from PIL import Image

from examples.tensorflow.common.object_detection.utils import box_utils
from examples.tensorflow.common.object_detection.utils import dataloader_utils
from examples.tensorflow.common.object_detection.utils import input_utils
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import letterbox_resize
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import normalize_image
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import random_horizontal_flip
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import random_mosaic_augment
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import random_resize_crop_pad
from examples.tensorflow.common.object_detection.utils.yolo_v4_utils import reshape_boxes


class YOLOv4Preprocessor:
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, config, is_train):
        """
        Initializes parameters for parsing annotations in the dataset.
        """
        self._max_num_instances = config.preprocessing.get("max_num_instances", 100)
        self._is_training = is_train
        self._global_batch_size = config.batch_size
        self._num_preprocess_workers = config.get("workers", tf.data.experimental.AUTOTUNE)

        self._parse_fn = self._parse_train_data
        self._parse_fn2 = self._parse_train_data2

        self._input_shape = config.input_shape
        self._enhance_mosaic_augment = config.preprocessing.enhance_mosaic_augment
        self._anchors = config.anchors
        self._num_classes = config.model_params.num_classes
        self._multi_anchor_assign = config.preprocessing.multi_anchor_assign

    def create_preprocess_input_fn(self):
        """Parses data to an image and associated training labels."""
        return self._tfds_decoder, self._pipeline_fn

    def _get_ground_truth_data(self, image, boxes, input_shape, max_boxes=100):
        """Random preprocessing for real-time data augmentation"""
        image_size = image.size
        model_input_size = tuple(reversed(input_shape))

        image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)
        image, horizontal_flip = random_horizontal_flip(image)
        image_data = np.array(image).astype(np.float32)

        # Reshape boxes based on augment
        boxes = reshape_boxes(
            boxes,
            src_shape=image_size,
            target_shape=model_input_size,
            padding_shape=padding_size,
            offset=padding_offset,
            horizontal_flip=horizontal_flip,
        )
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            box_data[: len(boxes)] = boxes

        return image_data, box_data

    def _preprocess(self, image, groundtruth_classes, groundtruth_boxes, input_shape):
        image_np = image.numpy()
        image_pil = Image.fromarray(image_np)

        image_shape = image.shape[0:2]
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
        image, box = self._get_ground_truth_data(image_pil, boxes, input_shape)

        return image, box

    def _parse_train_data(self, data):
        """Parses data for training"""
        image = data["image"]
        groundtruth_classes = data["groundtruth_classes"]
        groundtruth_boxes = data["groundtruth_boxes"]

        image, box = tf.py_function(
            self._preprocess,
            [image, groundtruth_classes, groundtruth_boxes, self._input_shape],
            [tf.float32, tf.float32],
        )
        image.set_shape([None, None, 3])
        box.set_shape([None, 5])

        image = input_utils.random_adjust_brightness(image)
        image = input_utils.random_adjust_contrast(image)
        image = input_utils.random_adjust_hue(image)
        image = input_utils.random_adjust_saturation(image)
        image = tf.math.divide(image, 255.0)

        out = {}
        out["image"] = image
        out["box"] = box
        out["source_id"] = data["source_id"]

        return out

    def _preprocess_true_boxes(
        self, true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2
    ):
        """
        Preprocess true boxes to training input format

        :param true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        :param input_shape: array-like, hw, multiples of 32
        :param anchors: array, shape=(N, 2), wh
        :param num_classes: integer
        :param multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                             anchors for a single ground truth
        :return y_true: list of array, shape like yolo_outputs, xywh are reletive value
        """
        assert (true_boxes[..., 4] < num_classes).all(), "class id must be less than num_classes"
        num_layers = len(anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

        # Transform box info to (x_center, y_center, box_width, box_height, cls_id)
        # and image relative coordinate.
        true_boxes = np.array(true_boxes, dtype="float32")
        input_shape = np.array(input_shape, dtype="int32")
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        batch_size = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[layer_idx] for layer_idx in range(num_layers)]
        y_true = [
            np.zeros(
                (
                    batch_size,
                    grid_shapes[layer_idx][0],
                    grid_shapes[layer_idx][1],
                    len(anchor_mask[layer_idx]),
                    5 + num_classes,
                ),
                dtype="float32",
            )
            for layer_idx in range(num_layers)
        ]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.0
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(batch_size):
            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0:
                continue

            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.0
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
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
                for layer_idx in range(num_layers):
                    for n in row:
                        # use different matching policy for single & multi anchor assign
                        if multi_anchor_assign:
                            matching_rule = iou[t, n] > iou_thresh and n in anchor_mask[layer_idx]
                        else:
                            matching_rule = n in anchor_mask[layer_idx]

                        if matching_rule:
                            i = np.floor(true_boxes[b, t, 0] * grid_shapes[layer_idx][1]).astype("int32")
                            j = np.floor(true_boxes[b, t, 1] * grid_shapes[layer_idx][0]).astype("int32")
                            k = anchor_mask[layer_idx].index(n)
                            c = true_boxes[b, t, 4].astype("int32")
                            y_true[layer_idx][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                            y_true[layer_idx][b, j, i, k, 4] = 1
                            y_true[layer_idx][b, j, i, k, 5 + c] = 1
        return y_true

    def _preprocess2(self, image_data, box_data):
        image_data = image_data.numpy()
        box_data = box_data.numpy()

        if self._enhance_mosaic_augment:
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        anchors = np.array(self._anchors).astype(float).reshape(-1, 2)
        y_true1, y_true2, y_true3 = self._preprocess_true_boxes(
            box_data, self._input_shape, anchors, self._num_classes, self._multi_anchor_assign
        )
        return image_data, y_true1, y_true2, y_true3

    def _parse_train_data2(self, data):
        image_data = data["image"]
        box_data = data["box"]
        im_shape = image_data.shape

        image_data, out0, out1, out2 = tf.py_function(
            self._preprocess2, [image_data, box_data], [tf.float32, tf.float32, tf.float32, tf.float32]
        )
        image_data.set_shape(im_shape)
        out0.set_shape([im_shape[0], 19, 19, 3, 85])
        out1.set_shape([im_shape[0], 38, 38, 3, 85])
        out2.set_shape([im_shape[0], 76, 76, 3, 85])

        labels = {
            "y_true_0": out0,
            "y_true_1": out1,
            "y_true_2": out2,
        }

        return image_data, labels

    def _get_image_info(self, image):
        desired_size = tf.convert_to_tensor(self._input_shape, dtype=tf.float32)
        image_size = tf.cast(tf.shape(input=image)[0:2], tf.float32)
        scaled_size = desired_size

        scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        offset = tf.zeros((2,), tf.int32)

        image_info = tf.stack([image_size, tf.cast(desired_size, tf.float32), image_scale, tf.cast(offset, tf.float32)])
        return image_info

    def _preprocess_predict_image(self, image):
        image = image.numpy()
        model_image_size = self._input_shape
        image_pil = Image.fromarray(image)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        resized_image = letterbox_resize(image_pil, tuple(reversed(model_image_size)))
        image_data = np.asarray(resized_image).astype("float32")
        image_data = normalize_image(image_data)
        image_data = tf.convert_to_tensor(image_data, dtype=tf.float32)
        return image_data

    def _parse_predict_data(self, data):
        """Parses data for prediction"""
        image_data = data["image"]
        image_shape = tf.shape(input=image_data)[0:2]

        # needed only for eval
        image_info = self._get_image_info(image_data)

        # image preprocessing
        image_data = tf.py_function(self._preprocess_predict_image, [image_data], Tout=tf.float32)
        image_data.set_shape([None, None, 3])

        labels = {
            "image_info": image_info,
        }

        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(data["groundtruth_boxes"], image_shape)
        groundtruths = {
            "source_id": data["source_id"],
            "num_detections": tf.squeeze(tf.shape(data["groundtruth_classes"])),
            "boxes": boxes,
            "classes": data["groundtruth_classes"],
            "areas": data["groundtruth_area"],
            "is_crowds": tf.cast(data["groundtruth_is_crowd"], tf.int32),
        }
        groundtruths["source_id"] = dataloader_utils.process_source_id(groundtruths["source_id"])
        groundtruths = dataloader_utils.pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
        labels.update(groundtruths)

        return image_data, labels

    def _tfds_decoder(self, features_dict):
        def _decode_image(features):
            image = tf.image.decode_jpeg(features["image"], channels=3, dct_method="INTEGER_ACCURATE")
            image.set_shape([None, None, 3])
            return image

        def _convert_labels_to_91_classes(features):
            # 0..79 --> 0..90
            match = tf.constant(
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
                ],
                dtype=tf.int64,
            )  # fmt: skip

            labels = features["objects"]["label"]
            labels = tf.gather(match, labels, axis=None)
            return labels

        image = _decode_image(features_dict)
        if self._is_training:
            labels = features_dict["objects"]["label"]
        else:
            labels = _convert_labels_to_91_classes(features_dict)

        decoded_tensors = {
            "image": image,
            "source_id": tf.cast(features_dict["image/id"], tf.int32),
            "groundtruth_classes": labels,
            "groundtruth_is_crowd": features_dict["objects"]["is_crowd"],
            "groundtruth_area": features_dict["objects"]["area"],
            "groundtruth_boxes": features_dict["objects"]["bbox"],
        }
        return decoded_tensors

    def _pipeline_fn(self, dataset, decoder_fn):
        if self._is_training:
            preprocess_input_fn = self._parse_fn
            preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
            dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers)
            dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

            # part of preprocessing which requires batches
            preprocess_input_fn2 = self._parse_fn2
            dataset = dataset.map(preprocess_input_fn2, num_parallel_calls=self._num_preprocess_workers)
        else:
            preprocess_input_fn = self._parse_predict_data
            preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
            dataset = dataset.map(preprocess_pipeline, num_parallel_calls=self._num_preprocess_workers)
            dataset = dataset.batch(self._global_batch_size, drop_remainder=True)
        return dataset
