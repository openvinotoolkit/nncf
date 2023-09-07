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
from typing import List, Tuple, Type, Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.common.tensor_impl_np import NPNNCFTensor
from nncf.common.tensor_impl_np import NPNNCFTensorBackend


class ONNXNNCFTensor(NPNNCFTensor):
    pass


class ONNXNNCFTensorBackend(NPNNCFTensorBackend):
    pass