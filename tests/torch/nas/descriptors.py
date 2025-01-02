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

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import create_elasticity_builder_from_config
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.model_creation import create_nncf_network
from tests.torch.helpers import get_empty_config
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import ThreeConvModel


class ElasticityDesc:
    def __init__(
        self,
        elasticity_dim: ElasticityDim,
        model_cls: Optional[Callable] = None,
        ref_state: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        params: Dict[str, Any] = None,
        input_size: Optional[List[int]] = None,
        ref_search_space: Optional[Any] = None,
        ref_output_fn: Optional[Callable] = None,
    ):
        self.elasticity_dim = elasticity_dim
        self.model_cls = model_cls
        self.ref_state = ref_state
        self.name = name
        self.params = {} if params is None else params
        self.input_size = input_size
        self.ref_search_space = ref_search_space
        self.ref_output_fn = ref_output_fn

    def __str__(self):
        if self.name:
            return self.name
        result = self.elasticity_dim.value
        if hasattr(self.model_cls, "__name__"):
            result += "_" + self.model_cls.__name__
        return result

    def build_handler(self):
        model = self.model_cls()
        move_model_to_cuda_if_available(model)
        input_size = self.input_size
        if not input_size:
            input_size = model.INPUT_SIZE
        config = get_empty_config(input_sample_sizes=input_size)
        nncf_network = create_nncf_network(model, config)
        builder = self.create_builder()
        handler = builder.build(nncf_network)
        return handler, builder

    def create_builder(self) -> SingleElasticityBuilder:
        return create_elasticity_builder_from_config(self.params, self.elasticity_dim)

    def create_builder_with_config(self, config: Dict[str, Any]) -> SingleElasticityBuilder:
        return create_elasticity_builder_from_config(config, self.elasticity_dim)


class WidthElasticityDesc:
    def __init__(self, desc: ElasticityDesc, width_num_params_indicator: Optional[int] = -1):
        self._elasticity_desc = desc
        self._width_num_params_indicator = width_num_params_indicator

    @property
    def ref_search_space(self):
        return self._elasticity_desc.ref_search_space

    def build_handler(self):
        handler, builder = self._elasticity_desc.build_handler()
        handler.width_num_params_indicator = self._width_num_params_indicator
        return handler, builder

    def __str__(self):
        return str(self._elasticity_desc) + "_wi:" + str(self._width_num_params_indicator)


class ModelStats(NamedTuple):
    flops: int
    num_weights: int

    def __eq__(self, other: Tuple[int, int]):
        return other[0] == self.flops and other[1] == self.num_weights


class RefModelStats(NamedTuple):
    supernet: ModelStats
    kernel_stage: ModelStats
    width_stage: ModelStats
    depth_stage: ModelStats


class MultiElasticityTestDesc(NamedTuple):
    model_creator: Any
    ref_model_stats: RefModelStats = None
    blocks_to_skip: List[List[str]] = None
    input_sizes: List[int] = [1, 3, 32, 32]
    algo_params: Dict = {}
    name: str = None

    def __str__(self):
        if hasattr(self.model_creator, "__name__"):
            name = self.model_creator.__name__
        elif self.name is not None:
            name = self.name
        else:
            name = "NOT_DEFINED"
        return name


THREE_CONV_TEST_DESC = MultiElasticityTestDesc(
    model_creator=ThreeConvModel,
    ref_model_stats=RefModelStats(
        supernet=ModelStats(17400, 87),
        kernel_stage=ModelStats(7800, 39),
        depth_stage=ModelStats(6000, 30),
        width_stage=ModelStats(2000, 10),
    ),
    blocks_to_skip=[["ThreeConvModel/NNCFConv2d[conv1]/conv2d_0", "ThreeConvModel/NNCFConv2d[conv_to_skip]/conv2d_0"]],
    algo_params={"width": {"min_width": 1, "width_step": 1}},
    input_sizes=ThreeConvModel.INPUT_SIZE,
)
