"""
 Copyright (c) 2021 Intel Corporation
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

from typing import List

from nncf.common.graph.layer_attributes import BaseLayerAttributes


class TFNodeAttributes(BaseLayerAttributes):
    def __init__(self, data_format: str):
        self._data_format = data_format

    def get_data_format(self) -> str:
        return self._data_format


class TFWeightedNodeAttributes(TFNodeAttributes):
    def __init__(self, data_format: str, weight_shape: List[int]):
        super().__init__(data_format)
        self._weight_shape = weight_shape

    def get_weight_shape(self) -> List[int]:
        return self._weight_shape
