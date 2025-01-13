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

import types
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from numpy import random
from PIL.Image import Image


def intersect(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        augmentations.Compose([
            transforms.CenterCrop(10),
            transforms.ToTensor(),
            ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Image, target: Dict) -> Tuple[np.ndarray, List[Dict]]:
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Lambda:
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img: np.ndarray, target: List[Dict]):
        return self.lambd(img, target)


class ConvertFromInts:
    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        return image.astype(np.float32), target


class Normalize:
    def __init__(self, mean: np.float32, std: np.float32, normalize_coef: np.float32):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.normalize_coef = normalize_coef

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        image = image.astype(np.float32)
        image /= self.normalize_coef
        image -= self.mean
        image /= self.std
        return image.astype(np.float32), target


class ToAbsoluteCoords:
    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        boxes = np.asarray([x["bbox"] for x in target])
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        for i, x in enumerate(target):
            x["bbox"] = boxes[i]
        return image, target


class ToPercentCoords:
    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        boxes = np.asarray([x["bbox"] for x in target])
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        for i, x in enumerate(target):
            x["bbox"] = boxes[i]
        return image, target


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        image = cv2.resize(image, (self.size, self.size))
        return image, target


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, target


class RandomHue:
    def __init__(self, delta=18.0):
        assert 0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            idx = int(random.randint(len(self.perms)))
            swap = self.perms[idx]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target


class ConvertColor:
    def __init__(self, current="BGR", transform="HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, target


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, target


class ToCV2Image:
    def __call__(self, tensor, target: List[Dict]):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), target


class ToTensor:
    def __call__(self, cvimage, target: List[Dict]):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), target


class RandomSampleCrop:
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        height, width, _ = image.shape
        while True:
            boxes = np.asarray([x["bbox"] for x in target])
            labels = np.asarray([x["label_idx"] for x in target])
            # randomly choose a mode
            r_ind = int(random.randint(len(self.sample_options)))
            mode = self.sample_options[r_ind]
            if mode is None:
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1] : rect[3], rect[0] : rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                target = target[: len(current_boxes)]
                for i, x in enumerate(target):
                    x["bbox"] = current_boxes[i]
                    x["label_idx"] = current_labels[i]
                return current_image, target


class Expand:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        if random.randint(2):
            return image, target
        boxes = np.asarray([x["bbox"] for x in target])
        height, width, _ = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        new_width = int(width * ratio)
        new_height = int(height * ratio)
        left = int(left)
        top = int(top)
        expand_image_cv = cv2.copyMakeBorder(
            image,
            top=top,
            bottom=new_height - (top + height),
            left=left,
            right=new_width - (left + width),
            borderType=cv2.BORDER_CONSTANT,
            value=self.mean,
        )
        image = expand_image_cv

        if boxes is not None:
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
        for i, x in enumerate(target):
            x["bbox"] = boxes[i]

        return image, target


class RandomMirror:
    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        boxes = np.asarray([x["bbox"] for x in target])

        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        for i, x in enumerate(target):
            x["bbox"] = boxes[i]

        return image, target


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image: np.ndarray):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform="HSV"),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="BGR"),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image: np.ndarray, target: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        im = image.copy()
        im, target = self.rand_brightness(im, target)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, target = distort(im, target)
        return self.rand_light_noise(im, target)


class SSDAugmentation:
    def __init__(self, size: np.int32, mean: np.float32, std: np.float32, normalize_coef: np.float32):
        self.size = size
        self.mean = mean
        self.std = std
        self.normalize_coef = normalize_coef
        self.augment = Compose(
            [
                ToAbsoluteCoords(),
                Expand(self.mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                ConvertFromInts(),
                PhotometricDistort(),
                Normalize(self.mean, self.std, self.normalize_coef),
            ]
        )

    def __call__(self, img: Image, target: List[Dict]) -> Tuple[torch.Tensor, np.ndarray]:
        res = self.augment(np.array(img)[:, :, ::-1], target)
        boxes = np.asarray([x["bbox"] for x in res[1]])
        labels = np.asarray([x["label_idx"] for x in res[1]])
        return torch.from_numpy(res[0][:, :, (2, 1, 0)]).permute(2, 0, 1), np.hstack(
            (boxes, np.expand_dims(labels, axis=1))
        )
