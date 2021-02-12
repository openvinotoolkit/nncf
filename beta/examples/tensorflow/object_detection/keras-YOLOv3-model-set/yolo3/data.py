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
    image = Image.open(line[0]) # PIL Image object containing image data
    # print('annotation_line', annotation_line)
    # print('init pixel 100, 100, 0', np.array(image).shape, np.array(image)[100,100,:])
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

    # print('\n ORIGIN after random_resize_crop_pad\n', np.array(image)[100:102, 100:102, :])
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


def yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # prepare multiscale config
    # rescale_step = 0
    # input_shape_list = get_multiscale_list()
    while True:
        # if rescale_interval > 0:
        #     # Do multi-scale training on different input shape
        #     rescale_step = (rescale_step + 1) % rescale_interval
        #     if rescale_step == 0:
        #         input_shape = input_shape_list[random.randint(0, len(input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(batch_size):
            # if i==0:
            #     np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment=True)
            # print('input_shape', input_shape)
            # print('annotation_lines[i]', annotation_lines[i])
            # print('image:', image.shape, image.dtype) # image: (608, 608, 3) float64
            # print('box:', box.shape, box.dtype) # box: (100, 5) float64
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # print('box_data[0] after batching', box_data[0])
        if enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, multi_anchor_assign)

        # print('batch_size', batch_size)
        # print('image_data', len(image_data), image_data[0].shape, image_data[0].dtype) # 64 (608, 608, 3) float64
        # print('y_true', len(y_true), y_true[0].shape, y_true[0].dtype)
        # print('y_true', len(y_true), y_true[1].shape, y_true[1].dtype)
        # print('y_true', len(y_true), y_true[2].shape, y_true[2].dtype) # 3 (64, 76, 76, 3, 85) float32

        yield [image_data, *y_true], np.zeros(batch_size)

def yolo3_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return yolo3_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign)









def get_dataset_builders(config, num_devices):
    train_builder = COCODatasetBuilder(config=config,
                                       is_train=True,
                                       num_devices=num_devices)

    val_builder = COCODatasetBuilder(config=config,
                                     is_train=False,
                                     num_devices=num_devices)

    return train_builder, val_builder




from abc import ABC, abstractmethod

import tensorflow_datasets as tfds

from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.utils import set_hard_limit_num_open_files


class BaseDatasetBuilder(ABC):
    """Abstract dataset loader and input processing."""
    def __init__(self, config, is_train, num_devices):
        self._config = config

        self._is_train = is_train
        self._num_devices = num_devices
        self._global_batch_size = config['global_batch_size'] # config.batch_size

        # Dataset params
        self._dataset_dir = config['dataset_dir'] # config.dataset_dir
        self._dataset_name = config.get('dataset', None)
        self._dataset_type = config.get('dataset_type', 'tfds')
        self._as_supervised = False

        # Dataset loader
        self._dataset_loader = None

        # TFDS params
        self._skip_decoding = False

        # Dict with TFRecordDatasets
        self._tfrecord_datasets = {}

        self._split = 'train' if self._is_train else 'validation'

    @property
    def is_train(self):
        """Returns a `bool` flag which specifies whether it is a training or evaluation dataset."""
        return self._is_train

    @property
    def batch_size(self):
        """Returns per replica batch size."""
        return self._global_batch_size // self._num_devices

    @property
    def global_batch_size(self):
        """Returns global batch size."""
        return self.batch_size * self._num_devices

    @property
    def steps_per_epoch(self):
        """Returns steps per epoch"""
        return self.num_examples // self.global_batch_size

    @property
    @abstractmethod
    def num_examples(self):
        """Returns number of examples in the current dataset."""

    @property
    @abstractmethod
    def num_classes(self):
        """Returns number of classes in the current dataset."""

    @abstractmethod
    def _pipeline(self, dataset):
        """The pipeline which decodes and preprocesses the input data for model."""

    def build(self):
        dataset_builders = {
            'tfds': self._load_tfds,
            'tfrecords': self._load_tfrecords,
        }

        builder = dataset_builders.get(self._dataset_type, None)
        if builder is None:
            raise ValueError('Unknown dataset type {}'.format(self._dataset_type))

        dataset = builder()
        dataset = self._pipeline(dataset)

        return dataset

    def _load_tfds(self):
        logger.info('Using TFDS to load {} data.'.format(self._split))

        set_hard_limit_num_open_files()

        self._dataset_loader = tfds.builder(self._dataset_name,
                                            data_dir=self._dataset_dir)

        self._dataset_loader.download_and_prepare()

        decoders = {'image': tfds.decode.SkipDecoding()} \
            if self._skip_decoding else None

        read_config = tfds.ReadConfig(
            interleave_cycle_length=64,
            interleave_block_length=1)

        dataset = self._dataset_loader.as_dataset(
            split=self._split,
            as_supervised=self._as_supervised,
            shuffle_files=True,
            decoders=decoders,
            read_config=read_config)

        return dataset

    def _load_tfrecords(self):
        logger.info('Using TFRecords to load {} data.'.format(self._split))

        dataset_key = self._dataset_name.replace('/', '')
        if dataset_key in self._tfrecord_datasets:
            self._dataset_loader = self._tfrecord_datasets[dataset_key](
                config=self._config, is_train=self._is_train
            )
        else:
            raise ValueError('Unknown dataset name: {}'.format(self._dataset_name))

        dataset = self._dataset_loader.as_dataset()

        return dataset








from functools import partial
import tensorflow as tf

# from beta.examples.tensorflow.common.dataset_builder import BaseDatasetBuilder
from beta.examples.tensorflow.common.object_detection.datasets import tfrecords as records_dataset
# from beta.examples.tensorflow.common.object_detection.datasets.preprocessing_selector import get_preprocess_input_fn


class COCODatasetBuilder(BaseDatasetBuilder):
    """COCO2017 dataset loader and input processing."""
    def __init__(self, config, is_train, num_devices):
        super().__init__(config, is_train, num_devices)

        # Pipeline params
        self._shuffle_buffer_size = 1000
        self._num_preprocess_workers = config.get('workers', tf.data.experimental.AUTOTUNE)
        self._cache = False
        self._include_mask = config.get('include_mask', False)

        # TFDS params
        self._skip_decoding = True

        self._tfrecord_datasets = records_dataset.__dict__


    @property
    def num_examples(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.splits[self._split].num_examples
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_examples
        return None

    @property
    def num_classes(self):
        if self._dataset_type == 'tfds':
            return self._dataset_loader.info.features['objects']['label'].num_classes
        if self._dataset_type == 'tfrecords':
            return self._dataset_loader.num_classes
        return None


    def _tfds_decoder(self, features_dict):

        def _decode_image(features):
            # image = tf.io.decode_image(features['image'], channels=3)
            image = tf.image.decode_jpeg(features['image'], channels=3, dct_method='INTEGER_ACCURATE')
            image.set_shape([None, None, 3])
            return image

        # TODO: convert classes here
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
        # labels = _convert_labels_to_91_classes(features_dict)

        decoded_tensors = {
            'image': image,
            'source_filename': features_dict['image/filename'],
            'source_id': tf.cast(features_dict['image/id'], tf.int32), # Really needed? not sure..
            'groundtruth_classes': features_dict['objects']['label'],
            'groundtruth_boxes': features_dict['objects']['bbox'],
        }

        return decoded_tensors



    def _pipeline(self, dataset):
        if self._cache:
            dataset = dataset.cache()

        if self._is_train:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(self._shuffle_buffer_size)

        if self._dataset_type == 'tfrecords':
            decoder_fn = partial(self._dataset_loader.decoder, include_mask=self._include_mask)
        else:
            decoder_fn = self._tfds_decoder

        preprocess_input_fn = get_preprocess_input_fn(self._config, self._is_train)
        preprocess_pipeline = lambda record: preprocess_input_fn(decoder_fn(record))
        # preprocess_pipeline = lambda record: decoder_fn(record)
        dataset = dataset.map(preprocess_pipeline, num_parallel_calls=1) # self._num_preprocess_workers
        dataset = dataset.batch(self._global_batch_size, drop_remainder=True)

        # preprocessing which requires batches
        preprocess_input_fn2 = YOLOv4Preprocessor(self._config, self._is_train).create_preprocess_input_fn2()
        preprocess_pipeline2 = lambda record: preprocess_input_fn2(record)
        dataset = dataset.map(preprocess_pipeline2, num_parallel_calls=1) # self._num_preprocess_workers

        # dataset = dataset.flat_map(lambda x: tf.data.Dataset().from_tensor_slices(x))

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset



from beta.examples.tensorflow.object_detection.preprocessing.retinanet_preprocessing import RetinaNetPreprocessor
from beta.examples.tensorflow.segmentation.preprocessing.maskrcnn_preprocessing import MaskRCNNPreprocessor

def get_preprocess_input_fn(config, is_train):
    model_name = config['model'] # config.model
    if model_name == 'RetinaNet':
        preprocess_input_fn = RetinaNetPreprocessor(config, is_train).create_preprocess_input_fn()
    elif model_name == 'MaskRCNN':
        preprocess_input_fn = MaskRCNNPreprocessor(config, is_train).create_preprocess_input_fn()
    elif model_name == 'YOLOv4':
        preprocess_input_fn = YOLOv4Preprocessor(config, is_train).create_preprocess_input_fn()
    else:
        raise ValueError('Unknown model name {}'.format(model_name))

    return preprocess_input_fn



from beta.examples.tensorflow.common.object_detection.utils import box_utils

class YOLOv4Preprocessor:
    """Parser to parse an image and its annotations into a dictionary of tensors."""
    def __init__(self, config, is_train):
        """Initializes parameters for parsing annotations in the dataset.
        """

        self._is_training = is_train

        # Data is parsed depending on the `is_training` flag
        if self._is_training:
            self._parse_fn = self._parse_train_data
            self._annotation_file = 'train2017.txt'
        else:
            self._parse_fn = self._parse_train_data # self._parse_predict_data
            self._annotation_file = 'val2017.txt'

        self._parse_fn2 = self._parse_train_data2

        self._annotation_lines = self._get_annotation_lines()
        self.input_shape = config['input_shape']
        self.enhance_augment = config['enhance_augment']
        self.anchors = self.get_anchors(config['anchors_path'])
        self.num_classes = config['num_classes']
        self.multi_anchor_assign = config['multi_anchor_assign']
        self.batch_size = config['global_batch_size']

    def _get_annotation_lines(self):
        with open(self._annotation_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        lines_dict = {}
        for line in lines:
            line_data = line.split()
            if self._is_training:
                _, img_name = line_data[0].split('train2017/')
            else:
                _, img_name = line_data[0].split('val2017/')
            # lines_dict['image_name.jpg'] = boxes
            lines_dict[img_name] = np.array([np.array(list(map(int, box.split(',')))) for box in line_data[1:]])

        # # KeyError: '000000015830.jpg'
        # print('len dict', len(lines_dict))
        # for i, key in enumerate(lines_dict):
        #     print(key, lines_dict[key])
        #     if i == 10:
        #         break

        return lines_dict

    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def create_preprocess_input_fn(self):
        """Parses data to an image and associated training labels.
        """
        return self._parse_fn

    def create_preprocess_input_fn2(self):
        """Parses data to an image and associated training labels.
        """
        return self._parse_fn2

    def _parse_train_data(self, data):
        """Parses data for training and evaluation."""
        image = data['image'] # <dtype: 'uint8'> <class 'tensorflow.python.framework.ops.Tensor'> (None, None, 3)
        filename = data['source_filename']
        # print(data['source_id'])
        groundtruth_classes = data['groundtruth_classes']
        groundtruth_boxes = data['groundtruth_boxes']

        def preprocess(image, filename, groundtruth_classes, groundtruth_boxes, input_shape):

            def get_ground_truth_data(image, boxes, input_shape, filename, max_boxes=100):
                '''random preprocessing for real-time data augmentation'''
                # line = annotation_line.split()
                # image = Image.open(line[0]) # PIL Image object containing image data
                image_size = image.size
                model_input_size = tuple(reversed(input_shape))
                # boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

                # random resize image and crop|padding to target size
                image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)

                # print('\nafter random_resize_crop_pad\n', filename, np.array(image)[100:102,100:102,:])

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
                # image = random_blur(image)

                # random do motion blur to image
                # image = random_motion_blur(image, prob=0.2)

                # random vertical flip image
                image, vertical_flip = random_vertical_flip(image)

                # random distort image in HSV color space
                # NOTE: will cost more time for preprocess
                #       and slow down training speed
                # image = random_hsv_distort(image)

                # reshape boxes based on augment
                boxes = reshape_boxes(boxes, src_shape=image_size, target_shape=model_input_size,
                                      padding_shape=padding_size, offset=padding_offset,
                                      horizontal_flip=horizontal_flip,
                                      vertical_flip=vertical_flip)

                # random rotate image and boxes
                image, boxes = random_rotate(image, boxes)

                if len(boxes) > max_boxes:
                    boxes = boxes[:max_boxes]

                # prepare image & box data
                image_data = np.array(image)
                image_data = normalize_image(image_data)
                box_data = np.zeros((max_boxes, 5))
                if len(boxes) > 0:
                    box_data[:len(boxes)] = boxes

                return image_data, box_data

            image_shape = tf.shape(input=image)[0:2]

            image_np = image.numpy()
            # print('From inside TFDS:')
            # print('\nfilename', filename)
            # print('image init np\n', filename, image_np[100, 100, :])
            image_pil = Image.fromarray(image_np)
            # filename = filename.numpy().decode("utf-8")
            # print('\nfilename', filename)
            # boxes = self._annotation_lines[filename]
            # print('boxes', boxes)
            # print('groundtruth_classes', groundtruth_classes.numpy())
            # print('groundtruth_boxes', groundtruth_boxes.numpy())
            # print('image_np shape', image_np.shape)
            # filename 000000008010.jpg
            # boxes[[451 253 585 293 37]
            # [115 234 272 269 37]
            # [191 212 243 312 0]
            # [497 211 545 337 0]]
            # groundtruth_classes [37 37 0 0]
            # groundtruth_boxes [[0.5729345  0.7058125  0.66336346 0.9161875]
            # [0.52835214  0.18032813  0.60884875  0.42585936]
            # [0.48013544 0.2993906  0.7059368  0.38214064]
            # [0.47713318 0.77720314  0.7617833 0.85235935]]

            denormalized_boxes = box_utils.denormalize_boxes(groundtruth_boxes, image_shape)

            boxes = []
            for denormalized_box, category_id in zip(denormalized_boxes.numpy(), groundtruth_classes.numpy()):
                x_min = int(denormalized_box[1])
                y_min = int(denormalized_box[0])
                x_max = int(denormalized_box[3])
                y_max = int(denormalized_box[2])
                boxes.append([x_min, y_min, x_max, y_max, int(category_id)])
            boxes = np.array(boxes)
            # print('boxes_final', boxes)

            input_shape = input_shape.numpy()

            # print('data type in', type(image_pil))
            # print('\nimage init np(pil)\n', filename, np.array(image_pil)[100:102,100:102,:])

            image, box = get_ground_truth_data(image_pil, boxes, input_shape, filename)

            # print('\nimage preprocessed first stage\n', filename, image[100:102,100:102,:])

            image = tf.convert_to_tensor(image, dtype=tf.float64)
            box = tf.convert_to_tensor(box, dtype=tf.float64)

            return image, box

        image, box = tf.py_function(preprocess, [image, filename, groundtruth_classes, groundtruth_boxes, self.input_shape], [tf.float64, tf.float64])
        image.set_shape([None, None, 3])
        box.set_shape([None, 5])

        # # print('PIPELINE:')
        # # print('data[image]', image.dtype, type(image), image.shape)
        # def image_enhance(image, filename):
        #     # print('begine image_enhance', type(image))
        #     # from tensor eager to numpy and to PIL
        #     print('filename', filename)
        #     image = Image.fromarray(image.numpy())
        #     # print('from image_enhance', type(image))
        #
        #     # python processing
        #     input_shape = (608, 608)
        #     model_input_size = tuple(reversed(input_shape))
        #     image, padding_size, padding_offset = random_resize_crop_pad(image, target_size=model_input_size)
        #
        #     # convert back to numpy and to tensor eager
        #     image = tf.keras.preprocessing.image.img_to_array(image)
        #     image = normalize_image(image)
        #     image = tf.convert_to_tensor(image, dtype=tf.uint8)
        #
        #     # print('return image type', type(image), image.dtype)
        #
        #     return image
        #
        # image = tf.py_function(image_enhance, [image, filename], tf.uint8)
        # image.set_shape([None, None, 3])
        # # print('data[image]', image.dtype, type(image), image.shape)
        #
        # labels = [tf.ones([19, 19, 3, 85], tf.float32), tf.ones([38, 38, 3, 85], tf.float32), tf.ones([76, 76, 3, 85], tf.float32)]

        out = {}
        out['image'] = image
        out['box'] = box
        out['filename'] = filename

        return out




    def _parse_train_data2(self, data):

        image_data = data['image']
        box_data = data['box']
        filename = data['filename']

        im_shape = image_data.shape
        image_data, out0, out1, out2 = tf.py_function(self.preprocess2, [image_data, box_data, filename], [tf.float64, tf.float32, tf.float32, tf.float32]) # , tf.float32
        image_data.set_shape(im_shape)
        out0.set_shape([im_shape[0], 19, 19, 3, 85])
        out1.set_shape([im_shape[0], 38, 38, 3, 85])
        out2.set_shape([im_shape[0], 76, 76, 3, 85])

        # out = {}
        # out['image_input'] = image_data
        # out['y_true_0'] = out1
        # out['y_true_1'] = out2
        # out['y_true_2'] = out3
        # out['filename'] = filename
        #
        # return out, tf.zeros(64, dtype=tf.dtypes.float32)

        # Packs labels for model_fn outputs.
        labels = {
            'y_true_0': out0,
            'y_true_1': out1,
            'y_true_2': out2
        }

        return image_data, labels


    def preprocess2(self, image_data, box_data, filename):
        image_data = image_data.numpy()
        box_data = box_data.numpy()

        # print('box_data[0] after batching', box_data[0])

        if self.enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)


        # print('\nimage preprocessed\n', filename[0], image_data[0])

        y_true1, y_true2, y_true3 = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes, self.multi_anchor_assign)

        image_data = tf.convert_to_tensor(image_data, dtype=tf.float64)
        y_true1 = tf.convert_to_tensor(y_true1, dtype=tf.float32)
        y_true2 = tf.convert_to_tensor(y_true2, dtype=tf.float32)
        y_true3 = tf.convert_to_tensor(y_true3, dtype=tf.float32)

        # zeros = np.zeros(self.batch_size)
        # zeros = tf.convert_to_tensor(zeros, dtype=tf.float32)

        return image_data, y_true1, y_true2, y_true3



    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
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
