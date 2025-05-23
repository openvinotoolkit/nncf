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

import torch
from torch import nn

from examples.torch.common import restricted_pickle_module
from examples.torch.common.example_logger import logger
from examples.torch.object_detection.layers.modules.ssd_head import MultiOutputSequential
from examples.torch.object_detection.layers.modules.ssd_head import SSDDetectionOutput
from nncf.torch.checkpoint_loading import load_state


def conv_bn(inp, oup, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def mobilenet(start_input_channels=3):
    model = MultiOutputSequential(
        [11, 13],
        [
            conv_bn(start_input_channels, 32, 3, 2, 1),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        ],
    )
    return model


def extra_layers(start_input_channels):
    return MultiOutputSequential(
        [1, 3, 5, 7],
        [
            conv_bn(start_input_channels, 256, 1, 1, 0),
            conv_bn(256, 512, 3, 2, 1),
            conv_bn(512, 128, 1, 1, 0),
            conv_bn(128, 256, 3, 2, 1),
            conv_bn(256, 128, 1, 1, 0),
            conv_bn(128, 256, 3, 2, 1),
            conv_bn(256, 64, 1, 1, 0),
            conv_bn(64, 128, 3, 2, 1),
        ],
    )


class MobileNetSSD(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.basenet = mobilenet()
        self.extras = extra_layers(1024)

        NUM_INPUT_FEATURES = [512, 1024, 512, 256, 256, 128]
        self.detection_head = SSDDetectionOutput(NUM_INPUT_FEATURES, num_classes, cfg)

    def forward(self, x):
        img_tensor = x[0].clone().unsqueeze(0)

        sources, x = self.basenet(x)
        extra_sources, x = self.extras(x)

        return self.detection_head(sources + extra_sources, img_tensor)


def build_ssd_mobilenet(cfg, size, num_classes, config):
    if size != 300:
        msg = "Only Mobilenet-SSD with input size 300 is supported"
        raise ValueError(msg)
    mobilenet_ssd = MobileNetSSD(num_classes, cfg)

    if config.basenet and (config.resuming_checkpoint_path is None) and (config.weights is None):
        logger.debug("Loading base network...")
        #
        # ** WARNING: torch.load functionality uses Python's pickling facilities that
        # may be used to perform arbitrary code execution during unpickling. Only load the data you
        # trust.
        #
        basenet_weights = torch.load(config.basenet, pickle_module=restricted_pickle_module, weights_only=False)[
            "state_dict"
        ]
        new_weights = {}
        for wn, wv in basenet_weights.items():
            wn = wn.replace("model.", "")
            new_weights[wn] = wv

        load_state(mobilenet_ssd.basenet, new_weights, is_resume=False)
    return mobilenet_ssd
