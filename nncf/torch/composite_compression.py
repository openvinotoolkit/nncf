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
from typing import TypeVar

import torch.nn

import nncf
from nncf import NNCFConfig
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.common.composite_compression import CompositeCompressionLoss
from nncf.config.extractors import extract_algorithm_names
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork

TModel = TypeVar("TModel")


class PTCompositeCompressionLoss(CompositeCompressionLoss, PTCompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self) -> torch.nn.ModuleList:
        return self._child_losses


class PTCompositeCompressionAlgorithmBuilder(CompositeCompressionAlgorithmBuilder, PTCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        algo_names = extract_algorithm_names(config)
        if len(algo_names) < 2:
            raise nncf.ValidationError(
                "Composite algorithm builder must be supplied with a config with more than one "
                "compression algo specified!"
            )
        for algo_name in algo_names:
            algo_builder = PT_COMPRESSION_ALGORITHMS.get(algo_name)
            self._child_builders.append(algo_builder(config, should_init=should_init))

    def __bool__(self):
        return bool(self.child_builders)

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        transformer = PTModelTransformer(model)
        layout = self.get_transformation_layout(model)
        transformed_model = transformer.transform(layout)

        self.initialize(transformed_model)

        return transformed_model

    def _build_controller(self, model: TModel) -> PTCompressionAlgorithmController:
        """
        Simple implementation of building controller without setting builder state and loading controller's one.
        Builds `PTCompositeCompressionAlgorithmController` to handle the additional
        modules, parameters, and hooks inserted into the model to enable
        algorithm-specific compression.

        :param model: The model with additional modifications necessary to enable
         algorithm-specific compression during fine-tuning.
        :return: The instance of the `PTCompositeCompressionAlgorithmController`.
        """
        if len(self._child_builders) == 1:
            return self._child_builders[0].build_controller(model)
        composite_ctrl = PTCompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model: TModel) -> PTTransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `PTTransformationLayout` class containing
            a list of algorithm-specific modifications.
        """
        transformations = PTTransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations

    def initialize(self, model: TModel) -> None:
        for builder in self.child_builders:
            if builder.should_init:
                builder.initialize(model)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        pass  # Higher-level get_transformation_layout is overridden, no need to define this


class PTCompositeCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, PTCompressionAlgorithmController
):
    def __init__(self, target_model: TModel):
        super().__init__(target_model)
        self._loss = PTCompositeCompressionLoss()

    def distributed(self):
        for ctrl in self.child_ctrls:
            ctrl.distributed()

    def prepare_for_export(self):
        for child_ctrl in self.child_ctrls:
            child_ctrl.prepare_for_export()

    @property
    def compression_rate(self) -> float:
        sum_compression_rate = 0
        not_none_compression_rate_cnt = 0
        for child_ctrl in self.child_ctrls:
            compression_rate = child_ctrl.compression_rate
            if compression_rate is not None:
                sum_compression_rate += sum_compression_rate
                not_none_compression_rate_cnt += 1
        return sum_compression_rate / max(not_none_compression_rate_cnt, 1)
