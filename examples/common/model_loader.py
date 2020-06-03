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
import torch
import torchvision.models
from functools import partial

import examples.common.models as custom_models
from examples.common.example_logger import logger
from nncf import load_state
from nncf.utils import safe_thread_call


def load_model(model, pretrained=True, num_classes=1000, model_params=None,
               weights_path: str = None) -> torch.nn.Module:
    logger.info("Loading model: {}".format(model))
    if model_params is None:
        model_params = {}
    if model in torchvision.models.__dict__:
        load_model_fn = partial(torchvision.models.__dict__[model], num_classes=num_classes, pretrained=pretrained,
                                **model_params)
    elif model in custom_models.__dict__:
        load_model_fn = partial(custom_models.__dict__[model], num_classes=num_classes, pretrained=pretrained,
                                **model_params)
    else:
        raise Exception("Undefined model name")
    loaded_model = safe_thread_call(load_model_fn)
    if not pretrained and weights_path is not None:
        sd = torch.load(weights_path, map_location='cpu')
        load_state(loaded_model, sd, is_resume=False)
    return loaded_model
