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
import urllib.parse
from functools import partial
from os import path as osp
from typing import Dict, Optional

import requests
import torch
import torchvision.models

import examples.torch.common.models as custom_models
from examples.torch.classification.models.mobilenet_v2_32x32 import MobileNetV2For32x32
from examples.torch.common import restricted_pickle_module
from examples.torch.common.example_logger import logger
from nncf.definitions import CACHE_MODELS_PATH
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.utils import safe_thread_call


def load_model(
    model, pretrained=True, num_classes=1000, model_params=None, weights_path: str = None
) -> torch.nn.Module:
    """

    ** WARNING: This is implemented using torch.load functionality,
    which itself uses Python's pickling facilities that may be used to perform
    arbitrary code execution during unpickling. Only load the data you trust.

    """
    logger.info("Loading model: {}".format(model))
    if model_params is None:
        model_params = {}
    if model in torchvision.models.__dict__:
        load_model_fn = partial(
            torchvision.models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    elif model in custom_models.__dict__:
        load_model_fn = partial(
            custom_models.__dict__[model], num_classes=num_classes, pretrained=pretrained, **model_params
        )
    elif model == "mobilenet_v2_32x32":
        load_model_fn = partial(MobileNetV2For32x32, num_classes=100)
    else:
        raise Exception("Undefined model name")
    loaded_model = safe_thread_call(load_model_fn)
    if not pretrained and weights_path is not None:
        # Check if provided path is a url and download the checkpoint if yes
        if is_url(weights_path):
            weights_path = download_checkpoint(weights_path)
        sd = torch.load(weights_path, map_location="cpu", pickle_module=restricted_pickle_module)
        if MODEL_STATE_ATTR in sd:
            sd = sd[MODEL_STATE_ATTR]
        load_state(loaded_model, sd, is_resume=False)
    return loaded_model


MODEL_STATE_ATTR = "state_dict"
COMPRESSION_STATE_ATTR = "compression_state"


def load_resuming_checkpoint(resuming_checkpoint_path: str):
    if osp.isfile(resuming_checkpoint_path):
        logger.info("=> loading checkpoint '{}'".format(resuming_checkpoint_path))
        checkpoint = torch.load(resuming_checkpoint_path, map_location="cpu", pickle_module=restricted_pickle_module)
        return checkpoint
    raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint_path))


def extract_model_and_compression_states(resuming_checkpoint: Optional[Dict] = None):
    if resuming_checkpoint is None:
        return None, None
    compression_state = resuming_checkpoint.get(COMPRESSION_STATE_ATTR)
    model_state_dict = resuming_checkpoint.get(MODEL_STATE_ATTR)
    return model_state_dict, compression_state


def is_url(uri):
    """
    Checks if given URI is a URL
    :param uri: URI to check
    :return: True if URI is a URL, and False otherwise
    """
    try:
        parsed_url = urllib.parse.urlparse(uri)
        return parsed_url.scheme and parsed_url.netloc
    except:  # noqa: E722
        return False


def download_checkpoint(url):
    """
    Downloads a checkpoint by URL and returns the path where it was downloaded
    :param url: URL to download a checkpoint from
    :return: path where the checkpoint was downloaded
    """
    if not CACHE_MODELS_PATH.exists():
        CACHE_MODELS_PATH.mkdir(parents=True)
    download_path = CACHE_MODELS_PATH / url.split("/")[-1]
    if not download_path.exists():
        print("Downloading checkpoint ...")
        checkpoint = requests.get(url)
        with open(download_path, "wb") as f:
            f.write(checkpoint.content)
    return str(download_path)
