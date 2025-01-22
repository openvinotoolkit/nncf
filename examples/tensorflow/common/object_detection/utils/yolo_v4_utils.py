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

import random

import numpy as np
from PIL import Image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def letterbox_resize(image, target_size):
    """
    Resize image with unchanged aspect ratio using padding

    :param image: origin image to be resize
            PIL Image object containing image data
    :param target_size: target image size,
            tuple of format (width, height).
    :param return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value
    :return new_image: resized PIL Image object.
    :return padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
    :return offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w / src_w, target_h / src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w) // 2
    dy = (target_h - padding_h) // 2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.Resampling.BICUBIC)
    new_image = Image.new("RGB", target_size, (128, 128, 128))
    new_image.paste(image, offset)

    return new_image


def random_resize_crop_pad(image, target_size, aspect_ratio_jitter=0.1, scale_jitter=0.7):
    """
    Randomly resize image and crop|padding to target size. It can
    be used for data augment in training data preprocess

    :param image: origin image to be resize
            PIL Image object containing image data
    :param target_size: target image size,
            tuple of format (width, height).
    :param aspect_ratio_jitter: jitter range for random aspect ratio,
            scalar to control the aspect ratio of random resized image.
    :param scale_jitter: jitter range for random resize scale,
            scalar to control the resize scale of random resized image.
    :return new_image: target sized PIL Image object.
    :return padding_size: random generated padding image size.
            will be used to reshape the ground truth bounding box
    :return padding_offset: random generated offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    target_w, target_h = target_size

    # generate random aspect ratio & scale for resize
    rand_aspect_ratio = (target_w / target_h * rand(1 - aspect_ratio_jitter, 1 + aspect_ratio_jitter)) / (
        rand(1 - aspect_ratio_jitter, 1 + aspect_ratio_jitter)
    )
    rand_scale = rand(scale_jitter, 1 / scale_jitter)

    # calculate random padding size and resize
    if rand_aspect_ratio < 1:
        padding_h = int(rand_scale * target_h)
        padding_w = int(padding_h * rand_aspect_ratio)
    else:
        padding_w = int(rand_scale * target_w)
        padding_h = int(padding_w / rand_aspect_ratio)
    padding_size = (padding_w, padding_h)
    image = image.resize(padding_size, Image.Resampling.BICUBIC)

    # get random offset in padding image
    dx = int(rand(0, target_w - padding_w))
    dy = int(rand(0, target_h - padding_h))
    padding_offset = (dx, dy)

    # create target image
    new_image = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    new_image.paste(image, padding_offset)

    return new_image, padding_size, padding_offset


def reshape_boxes(boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False, vertical_flip=False):
    """
    Reshape bounding boxes from src_shape image to target_shape image,
    usually for training data preprocess

    :param boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).
    :param src_shape: origin image shape,
            tuple of format (width, height).
    :param target_shape: target image shape,
            tuple of format (width, height).
    :param padding_shape: padding image shape,
            tuple of format (width, height).
    :param offset: top-left offset when padding target image.
            tuple of format (dx, dy).
    :param horizontal_flip: whether to do horizontal flip.
            boolean flag.
    :param vertical_flip: whether to do vertical flip.
            boolean flag.
    :return boxes: reshaped bounding box numpy array
    """
    if len(boxes) > 0:
        src_w, src_h = src_shape
        target_w, target_h = target_shape
        padding_w, padding_h = padding_shape
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * padding_w / src_w + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * padding_h / src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0, 2]] = target_w - boxes[:, [2, 0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1, 3]] = target_h - boxes[:, [3, 1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]  # discard invalid box

    return boxes


def random_horizontal_flip(image, prob=0.5):
    """
    Random horizontal flip for image

    :param image: origin image for horizontal flip
            PIL Image object containing image data
    :param prob: probability for random flip,
            scalar to control the flip probability.
    :return image: adjusted PIL Image object.
    :return flip: boolean flag for horizontal flip action
    """
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return image, flip


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def merge_mosaic_bboxes(bboxes, crop_x, crop_y, image_size):
    # adjust & merge mosaic samples bboxes as following area order:
    # -----------
    # |     |   |
    # |  0  | 3 |
    # |     |   |
    # -----------
    # |  1  | 2 |
    # -----------
    assert bboxes.shape[0] == 4, "mosaic sample number should be 4"
    max_boxes = bboxes.shape[1]
    height, width = image_size
    merge_bbox = []
    for i in range(bboxes.shape[0]):
        for box in bboxes[i]:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

            if i == 0:  # bboxes[0] is for top-left area
                if y_min > crop_y or x_min > crop_x:
                    continue
                if y_min < crop_y < y_max:
                    y_max = crop_y
                if x_min < crop_x < x_max:
                    x_max = crop_x

            if i == 1:  # bboxes[1] is for bottom-left area
                if y_max < crop_y or x_min > crop_x:
                    continue
                if y_min < crop_y < y_max:
                    y_min = crop_y
                if x_min < crop_x < x_max:
                    x_max = crop_x

            if i == 2:  # bboxes[2] is for bottom-right area
                if y_max < crop_y or x_max < crop_x:
                    continue
                if y_min < crop_y < y_max:
                    y_min = crop_y
                if x_min < crop_x < x_max:
                    x_min = crop_x

            if i == 3:  # bboxes[3] is for top-right area
                if y_min > crop_y or x_max < crop_x:
                    continue
                if y_min < crop_y < y_max:
                    y_max = crop_y
                if x_min < crop_x < x_max:
                    x_min = crop_x

            if abs(x_max - x_min) < max(10, width * 0.01) or abs(y_max - y_min) < max(10, height * 0.01):
                # if the adjusted bbox is too small, bypass it
                continue

            merge_bbox.append([x_min, y_min, x_max, y_max, box[4]])

    if len(merge_bbox) > max_boxes:
        merge_bbox = merge_bbox[:max_boxes]

    box_data = np.zeros((max_boxes, 5))
    if len(merge_bbox) > 0:
        box_data[: len(merge_bbox)] = merge_bbox
    return box_data


def random_mosaic_augment(image_data, boxes_data, prob=0.1):
    """
    Random add mosaic augment on batch images and boxes, from YOLOv4
    reference:
        https://github.com/klauspa/Yolov4-tensorflow/blob/master/data.py
        https://github.com/clovaai/CutMix-PyTorch
        https://github.com/AlexeyAB/darknet

    :param image_data: origin images for mosaic augment
            numpy array for normalized batch image data
    :param boxes_data: origin bboxes for mosaic augment
            numpy array for batch bboxes
    :param prob: probability for augment ,
            scalar to control the augment probability.
    :return image_data: augmented batch image data.
    :return boxes_data: augmented batch bboxes data.
    """
    do_augment = rand() < prob

    if do_augment:
        batch_size = len(image_data)
        assert batch_size >= 4, "mosaic augment need batch size >= 4"

        def get_mosaic_samples():
            # random select 4 images from batch as mosaic samples
            random_index = random.sample(list(range(batch_size)), 4)

            random_images = []
            random_bboxes = []
            for idx in random_index:
                random_images.append(image_data[idx])
                random_bboxes.append(boxes_data[idx])
            return random_images, np.array(random_bboxes)

        min_offset = 0.2
        new_images = []
        new_boxes = []
        height, width = image_data[0].shape[:2]
        # each batch has batch_size images, so we also need to
        # generate batch_size mosaic images
        for _ in range(batch_size):
            images, bboxes = get_mosaic_samples()

            # crop_x = np.random.randint(int(width*min_offset), int(width*(1 - min_offset)))
            # crop_y = np.random.randint(int(height*min_offset), int(height*(1 - min_offset)))
            crop_x = int(random.uniform(int(width * min_offset), int(width * (1 - min_offset))))  # nosec
            crop_y = int(random.uniform(int(height * min_offset), int(height * (1 - min_offset))))  # nosec

            merged_boxes = merge_mosaic_bboxes(bboxes, crop_x, crop_y, image_size=(height, width))
            # no valid bboxes, drop this loop
            # if merged_boxes is None:
            # i = i - 1
            # continue

            # crop out selected area as following mosaic sample images order:
            # -----------
            # |     |   |
            # |  0  | 3 |
            # |     |   |
            # -----------
            # |  1  | 2 |
            # -----------
            area_0 = images[0][:crop_y, :crop_x, :]
            area_1 = images[1][crop_y:, :crop_x, :]
            area_2 = images[2][crop_y:, crop_x:, :]
            area_3 = images[3][:crop_y, crop_x:, :]

            # merge selected area to new image
            area_left = np.concatenate([area_0, area_1], axis=0)
            area_right = np.concatenate([area_3, area_2], axis=0)
            merged_image = np.concatenate([area_left, area_right], axis=1)

            new_images.append(merged_image)
            new_boxes.append(merged_boxes)

        new_images = np.stack(new_images)
        new_boxes = np.array(new_boxes)
        image_data = new_images
        boxes_data = new_boxes

    return image_data, boxes_data


def normalize_image(image):
    """
    Normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    :param image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0
    :return image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image / 255.0
    return image
