# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class OVConvLayoutElem(Enum):
    """
    Layout elements descriptor for convolutional and linear openvino layers:
        C_IN: Input channels dimension.
        C_OUT: Output channels dimension.
        SPATIAL: Spatial dimension.
        GROUPS: Groups dimention.
    """

    C_IN = "channels_in"
    C_OUT = "channels_out"
    SPATIAL = "spatial"
    GROUPS = "groups"
