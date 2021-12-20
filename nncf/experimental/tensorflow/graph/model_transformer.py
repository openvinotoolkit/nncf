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

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.experimental.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.experimental.tensorflow.patch_tf import TFPatcher
from nncf.experimental.tensorflow.patch_tf import Hook


class TFModelTransformer(ModelTransformer):
    """
    Applies transformations to the NNCF network.
    """

    def __init__(self, model: NNCFNetwork):
        """
        Initializes the model transformer.

        :param model: NNCF network.
        """
        super().__init__(model)

    def transform(self, transformation_layout: TFTransformationLayout) -> NNCFNetwork:
        """
        Applies transformations to the model.

        :param transformation_layout: An instance of `TransformationLayout` that
            includes a list of transformations to be applied to the NNCF network.
        :return: The transformed NNCF network.
        """

        model_hooks = getattr(self._model, '_hooks')
        for command in transformation_layout.transformations:
            if command.type == TransformationType.INSERT:
                hook = Hook(command.insertion_objects, command.target_point)
                model_hooks.append(hook)
                TFPatcher.add_hook(hook)
            elif command.type == TransformationType.REMOVE:
                # TODO(andrey-churkin): Add support
                pass
            else:
                raise ValueError(f'Transformation type {command.type} does not support.')

        with self._model.distribute_strategy.scope():
            for op in self._model.nncf_operations:
                op.build(self._model)

        return self._model
