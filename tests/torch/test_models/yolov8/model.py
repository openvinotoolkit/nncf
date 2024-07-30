# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Source: ultralytics/ultralytics/nn/tasks.py
Commit: 673e76b86282859ead5517bd04dee896a647db93
"""

import contextlib
import math

import torch
import torch.nn as nn

from tests.torch.test_models.yolov8.block import C1
from tests.torch.test_models.yolov8.block import C2
from tests.torch.test_models.yolov8.block import C3
from tests.torch.test_models.yolov8.block import C3TR
from tests.torch.test_models.yolov8.block import ELAN1
from tests.torch.test_models.yolov8.block import PSA
from tests.torch.test_models.yolov8.block import SPP
from tests.torch.test_models.yolov8.block import SPPELAN
from tests.torch.test_models.yolov8.block import SPPF
from tests.torch.test_models.yolov8.block import AConv
from tests.torch.test_models.yolov8.block import ADown
from tests.torch.test_models.yolov8.block import Bottleneck
from tests.torch.test_models.yolov8.block import BottleneckCSP
from tests.torch.test_models.yolov8.block import C2f
from tests.torch.test_models.yolov8.block import C2fAttn
from tests.torch.test_models.yolov8.block import C3Ghost
from tests.torch.test_models.yolov8.block import C3x
from tests.torch.test_models.yolov8.block import CBFuse
from tests.torch.test_models.yolov8.block import CBLinear
from tests.torch.test_models.yolov8.block import GhostBottleneck
from tests.torch.test_models.yolov8.block import GhostConv
from tests.torch.test_models.yolov8.block import HGBlock
from tests.torch.test_models.yolov8.block import HGStem
from tests.torch.test_models.yolov8.block import ImagePoolingAttn
from tests.torch.test_models.yolov8.block import RepC3
from tests.torch.test_models.yolov8.block import RepNCSPELAN4
from tests.torch.test_models.yolov8.block import ResNetLayer
from tests.torch.test_models.yolov8.block import SCDown
from tests.torch.test_models.yolov8.conv import Concat
from tests.torch.test_models.yolov8.conv import Conv
from tests.torch.test_models.yolov8.conv import ConvTranspose
from tests.torch.test_models.yolov8.conv import DWConv
from tests.torch.test_models.yolov8.conv import DWConvTranspose2d
from tests.torch.test_models.yolov8.conv import Focus
from tests.torch.test_models.yolov8.head import OBB
from tests.torch.test_models.yolov8.head import Classify
from tests.torch.test_models.yolov8.head import Detect
from tests.torch.test_models.yolov8.head import Pose
from tests.torch.test_models.yolov8.head import Segment
from tests.torch.test_models.yolov8.head import WorldDetect
from tests.torch.test_models.yolov8.head import v10Detect


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            print(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            print(f"activation: {act}")  # print

    if verbose:
        print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            # C2fCIB,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            # if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fCIB}:
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3}:
                args.insert(2, n)  # number of repeats
                n = 1
        # elif m is AIFI:
        #    args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        # elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
        #    args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


YOLOV8_CONFIG = {
    "nc": 80,
    "scales": {
        "n": [0.33, 0.25, 1024],
        "s": [0.33, 0.5, 1024],
        "m": [0.67, 0.75, 768],
        "l": [1.0, 1.0, 512],
        "x": [1.0, 1.25, 512],
    },
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C2f", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C2f", [512]],
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 3, "C2f", [256]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],
        [-1, 3, "C2f", [512]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 9], 1, "Concat", [1]],
        [-1, 3, "C2f", [1024]],
        [[15, 18, 21], 1, "Detect", ["nc"]],
    ],
    "scale": "n",
    "yaml_file": "yolov8n.yaml",
    "ch": 3,
}


class YoloV8Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.save = parse_model(YOLOV8_CONFIG, YOLOV8_CONFIG["ch"], verbose=False)

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x
