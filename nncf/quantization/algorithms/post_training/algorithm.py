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

import itertools
from typing import Callable, List, Optional, TypeVar

from nncf import Dataset
from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.post_training.pipeline import create_ptq_pipeline
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")
TPass = Callable[[TModel], TModel]


class PostTrainingQuantization(Algorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) ChannelAlignment
    2) MinMaxQuantization
    3) FastBiasCorrection or BiasCorrection
    """

    def __init__(
        self,
        mode: Optional[QuantizationMode] = None,
        preset: Optional[QuantizationPreset] = None,
        target_device: TargetDevice = TargetDevice.ANY,
        subset_size: int = 300,
        fast_bias_correction: bool = True,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
    ):
        """
        :param mode: Special quantization mode that specify different ways of the optimization.
        :param preset: A preset controls the quantization mode (symmetric and asymmetric).
            It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
            Default value is None. In this case, `mixed` preset is used for `transformer`
            model type otherwise `performace`.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device.
        :param subset_size: Size of a subset to calculate activations
            statistics used for quantization.
        :param fast_bias_correction: Setting this option to `False` enables a different
            bias correction method which is more accurate, in general, and takes
            more time but requires less memory.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param advanced_parameters: Advanced quantization parameters for
            fine-tuning the quantization algorithm
        """
        self._pipeline = create_ptq_pipeline(
            mode=mode,
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            fast_bias_correction=fast_bias_correction,
            model_type=model_type,
            ignored_scope=ignored_scope,
            advanced_parameters=advanced_parameters,
        )

    @property
    def available_backends(self) -> List[BackendType]:
        backends = set(BackendType)
        for algorithm in itertools.chain.from_iterable(self._pipeline.pipeline_steps):
            backends = backends.intersection(algorithm.available_backends)
        return list(backends)

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        return self._pipeline.get_statistic_points_for_step(0, model, graph)

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        if dataset is None and len(self._pipeline.pipeline_steps) > 1:
            raise ValueError(
                "A dataset is required for the post-training quantization "
                "algorithm to collect statistics for intermediate models."
            )

        step_index_to_statistics = None
        if statistic_points:
            step_index_to_statistics = {0: statistic_points}

        return self._pipeline.run_from_step(model, dataset, graph, 0, step_index_to_statistics)
