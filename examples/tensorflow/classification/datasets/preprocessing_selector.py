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

import nncf
from examples.tensorflow.classification.datasets.preprocessing.cifar import cifar10_preprocess_image
from examples.tensorflow.classification.datasets.preprocessing.cifar import cifar100_preprocess_image
from examples.tensorflow.classification.datasets.preprocessing.imagenet import imagenet_1000_to_1001_classes
from examples.tensorflow.classification.datasets.preprocessing.imagenet import imagenet_preprocess_image
from examples.tensorflow.classification.datasets.preprocessing.imagenet import imagenet_slim_preprocess_image

PREPROCESSING_FN_MAP = {
    "imagenet2012": imagenet_preprocess_image,
    "cifar10": cifar10_preprocess_image,
    "cifar100": cifar100_preprocess_image,
    "imagenet2012_slim": imagenet_slim_preprocess_image,
}

LABEL_PREPROCESSING_FN_MAP = {"imagenet2012": imagenet_1000_to_1001_classes}


def get_preprocessing(dataset_name, model_name, preset=None):
    if not preset:
        preset = dataset_name
    if preset not in PREPROCESSING_FN_MAP:
        raise nncf.ValidationError(
            "Preprocessing for dataset {} and model {} was not recognized".format(dataset_name, model_name)
        )

    ext_kwargs = {}
    if preset == "imagenet2012":
        ext_kwargs = {"model_name": model_name}

    def preprocessing_fn(image, image_size, is_training, dtype, **kwargs):
        kwargs.update(ext_kwargs)
        return PREPROCESSING_FN_MAP[preset](
            image=image, image_size=image_size, is_training=is_training, dtype=dtype, **kwargs
        )

    return preprocessing_fn


def get_label_preprocessing_fn(dataset_name, num_classes, loader_num_classes):
    if num_classes == loader_num_classes or dataset_name not in LABEL_PREPROCESSING_FN_MAP:
        return lambda label: label

    def preprocessing_fn(label):
        kwargs = {}
        class_diff = num_classes - loader_num_classes
        return LABEL_PREPROCESSING_FN_MAP[dataset_name](label=label, class_diff=class_diff, **kwargs)

    return preprocessing_fn
