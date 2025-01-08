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
# Efficient Net implementation from: https://github.com/lukemelas/EfficientNet-PyTorch

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

from nncf.torch import register_module

wrapper = register_module()
wrapper(Conv2dStaticSamePadding)


def efficient_net(pretrained=True, num_classes=1000, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained(num_classes=num_classes, **kwargs)
    return EfficientNet.from_name(num_classes=num_classes, **kwargs)
