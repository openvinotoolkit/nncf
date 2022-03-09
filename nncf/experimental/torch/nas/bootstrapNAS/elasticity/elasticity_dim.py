"""
 Copyright (c) 2022 Intel Corporation
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
from enum import Enum


class ElasticityDim(Enum):
    """
    Defines elasticity dimension or type of elasticity applied to the model
    """
    KERNEL = 'kernel'
    WIDTH = 'width'
    DEPTH = 'depth'

    @classmethod
    def from_str(cls, dim: str) -> 'ElasticityDim':
        if dim == ElasticityDim.KERNEL.value:
            return ElasticityDim.KERNEL
        if dim == ElasticityDim.WIDTH.value:
            return ElasticityDim.WIDTH
        if dim == ElasticityDim.DEPTH.value:
            return ElasticityDim.DEPTH
        raise RuntimeError(f"Unknown elasticity dimension: {dim}."
                           f"List of supported: {[e.value for e in ElasticityDim]}")
