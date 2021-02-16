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

from typing import Callable, Optional

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.graph.transformations.layout import TransformationLayout
from beta.nncf.tensorflow.graph.transformations.commands import TFMultipleInsertionCommands
from beta.nncf.tensorflow.graph.transformations.commands import TFLayer

GRAPH_NODE_TYPES = [
    TargetType.LAYER,
    TargetType.LAYER_WEIGHT_OPERATION
]


class TFTransformationLayout(TransformationLayout):
    def register(self, transformation: TransformationCommand) -> None:
        if transformation.type == TransformationType.REMOVE:
            self._transformations.append(transformation)
        elif transformation.type == TransformationType.INSERT:
            self._register_insertion_transformation(transformation)
        elif transformation.type == TransformationType.MULTI_INSERT:
            self._register_multiple_insertion_transformation(transformation)

    def _register_insertion_transformation(self, transformation: TransformationCommand) -> None:
        idx = self._find_transformation(
            transformation,
            lambda t0, t1: t0.check_command_compatibility(t1)
        )
        if idx is not None:
            self.transformations[idx] = self.transformations[idx] + transformation
            return

        idx = self._find_transformation(
            transformation,
            lambda t0, t1: t0.type == TransformationType.MULTI_INSERT and \
                           t0.check_insertion_command(t1)
        )
        if idx is not None:
            self.transformations[idx].add_insertion_command(transformation)
            return

        idx = self._find_transformation(
            transformation,
            lambda t0, t1: t0.type == TransformationType.INSERT and \
                           check_target_point(t0.target_point, t1.target_point)
        )
        if idx is not None:
            self.transformations.append(
                TFMultipleInsertionCommands(
                    target_point=TFLayer(transformation.target_point.layer_name),
                    check_target_point_fn=check_target_point,
                    commands=[self.transformations.pop(idx), transformation]
                ))
            return
        self.transformations.append(transformation)

    def _register_multiple_insertion_transformation(self, transformation: TransformationCommand) -> None:
        idx = self._find_transformation(
            transformation,
            lambda t0, t1: t0.check_command_compatibility(t1)
        )
        if idx is not None:
            self.transformations[idx] = self.transformations[idx] + transformation
            return

        merged_transformations = []
        for t in self.transformations:
            if transformation.check_insertion_command(t):
                transformation.add_insertion_command(t)
                merged_transformations.append(t)
        for t in merged_transformations:
            self.transformations.remove(t)
        self.transformations.append(transformation)

    def _find_transformation(self, transformation: TransformationCommand, condition: Callable) ->Optional[int]:
        for idx, t in enumerate(self.transformations):
            if condition(t, transformation):
                return idx
        return None


def check_target_point(tp0: TargetPoint, tp1: TargetPoint) -> bool:
    return tp0.type in GRAPH_NODE_TYPES and \
           tp1.type in GRAPH_NODE_TYPES and \
           tp0.layer_name == tp1.layer_name
