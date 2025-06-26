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

from typing import Callable, Optional

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.tensorflow.graph.transformations.commands import TFLayer
from nncf.tensorflow.graph.transformations.commands import TFLayerPoint
from nncf.tensorflow.graph.transformations.commands import TFMultipleInsertionCommands

GRAPH_NODE_TYPES = [TargetType.LAYER, TargetType.OPERATION_WITH_WEIGHTS]


class TFTransformationLayout(TransformationLayout):
    def register(self, transformation: TransformationCommand) -> None:
        if transformation.type == TransformationType.REMOVE:
            self._transformations.append(transformation)
        elif transformation.type == TransformationType.INSERT:
            self._register_insertion_transformation(transformation)
        elif transformation.type == TransformationType.MULTI_INSERT:
            self._register_multiple_insertion_transformation(transformation)

    def _register_insertion_transformation(self, transformation: TransformationCommand) -> None:
        start_idx = self._find_transformation(
            lambda t: t.type == TransformationType.REMOVE
            and is_object_removed(t.target_point, transformation.target_point),
            reverse=True,
        )
        start_idx = 0 if start_idx is None else start_idx + 1

        idx = self._find_transformation(lambda t: t.check_command_compatibility(transformation), start_idx=start_idx)
        if idx is not None:
            self.transformations[idx] = self.transformations[idx] + transformation
            return

        idx = self._find_transformation(
            lambda t: t.type == TransformationType.MULTI_INSERT and t.check_insertion_command(transformation),
            start_idx=start_idx,
        )
        if idx is not None:
            self.transformations[idx].add_insertion_command(transformation)
            return

        idx = self._find_transformation(
            lambda t: t.type == TransformationType.INSERT
            and check_target_points(t.target_point, transformation.target_point),
            start_idx=start_idx,
        )
        if idx is not None:
            self.transformations[idx] = TFMultipleInsertionCommands(
                target_point=TFLayer(transformation.target_point.layer_name),
                check_target_points_fn=check_target_points,
                commands=[self.transformations[idx], transformation],
            )
            return

        self.transformations.append(transformation)

    def _register_multiple_insertion_transformation(self, transformation: TransformationCommand) -> None:
        start_idx = self._find_transformation(
            lambda t: t.type == TransformationType.REMOVE
            and is_object_removed(t.target_point, transformation.target_point),
            reverse=True,
        )
        start_idx = 0 if start_idx is None else start_idx + 1

        idx = self._find_transformation(lambda t: t.check_command_compatibility(transformation), start_idx=start_idx)
        if idx is not None:
            self.transformations[idx] = self.transformations[idx] + transformation
            return

        merged_transformations = []
        for t in self.transformations[start_idx:]:
            if transformation.check_insertion_command(t):
                transformation.add_insertion_command(t)
                merged_transformations.append(t)
        for t in merged_transformations:
            self.transformations.remove(t)
        self.transformations.append(transformation)

    def _find_transformation(self, condition: Callable, start_idx: int = 0, reverse: bool = False) -> Optional[int]:
        transformations_iterator = (
            reversed(list(enumerate(self.transformations[start_idx:])))
            if reverse
            else enumerate(self.transformations[start_idx:])
        )
        for idx, t in transformations_iterator:
            if condition(t):
                return idx
        return None


def check_target_points(tp0: TargetPoint, tp1: TargetPoint) -> bool:
    return (
        isinstance(tp0, TFLayerPoint)
        and isinstance(tp1, TFLayerPoint)
        and tp0.type in GRAPH_NODE_TYPES
        and tp1.type in GRAPH_NODE_TYPES
        and tp0.layer_name == tp1.layer_name
    )


def is_object_removed(removed_target: TargetPoint, command_target: TargetPoint) -> bool:
    layer_removed = (
        removed_target.type == TargetType.LAYER
        and command_target.type in GRAPH_NODE_TYPES
        and removed_target.layer_name == command_target.layer_name
    )

    operation_removed = (
        removed_target.type == TargetType.OPERATION_WITH_WEIGHTS
        and removed_target.type == command_target.type
        and removed_target.layer_name == command_target.layer_name
        and removed_target.weights_attr_name == command_target.weights_attr_name
    )

    return layer_removed or operation_removed
