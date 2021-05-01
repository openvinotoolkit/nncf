"""
 Copyright (c) 2019 Intel Corporation
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
from os import path as osp

import torch
import torchvision.models
from functools import partial

import examples.common.models as custom_models
from examples.common.example_logger import logger
import examples.common.restricted_pickle_module as restricted_pickle_module
from nncf import load_state
from nncf.utils import safe_thread_call
from tests.test_models.mobilenet_v2_32x32 import MobileNetV2For32x32


def load_model(model, pretrained=True, num_classes=1000, model_params=None,
               weights_path: str = None) -> torch.nn.Module:
    """

       ** WARNING: This is implemented using torch.load functionality,
       which itself uses Python's pickling facilities that may be used to perform
       arbitrary code execution during unpickling. Only load the data you trust.

    """
    logger.info("Loading model: {}".format(model))
    if model_params is None:
        model_params = {}
    if model in torchvision.models.__dict__:
        load_model_fn = partial(torchvision.models.__dict__[model], num_classes=num_classes, pretrained=pretrained,
                                **model_params)
    elif model in custom_models.__dict__:
        load_model_fn = partial(custom_models.__dict__[model], num_classes=num_classes, pretrained=pretrained,
                                **model_params)
    elif model == "mobilenet_v2_32x32":
        load_model_fn = partial(MobileNetV2For32x32, num_classes=100)
    else:
        raise Exception("Undefined model name")
    loaded_model = safe_thread_call(load_model_fn)
    if not pretrained and weights_path is not None:
        sd = torch.load(weights_path, map_location='cpu', pickle_module=restricted_pickle_module)
        load_state(loaded_model, sd, is_resume=False)
    return loaded_model


def load_resuming_model_state_dict_and_checkpoint_from_path(resuming_checkpoint_path):
    logger.info('Resuming from checkpoint {}...'.format(resuming_checkpoint_path))
    resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu',
                                     pickle_module=restricted_pickle_module)
    # use checkpoint itself in case only the state dict was saved,
    # i.e. the checkpoint was created with `torch.save(module.state_dict())`
    resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
    return resuming_model_state_dict, resuming_checkpoint


def load_resuming_checkpoint(resuming_checkpoint_path: str):
    if osp.isfile(resuming_checkpoint_path):
        logger.info("=> loading checkpoint '{}'".format(resuming_checkpoint_path))
        checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu',
                                pickle_module=restricted_pickle_module)
        return checkpoint
    raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint_path))
