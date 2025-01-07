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

from typing import List

from nncf.common.graph.transformations.commands import TransformationCommand


class TransformationLayout:
    """
    Represents a list of transformation commands that has been prepared to be
    correctly applied to a model. In practice, stacking compression algorithms or
    a specific model transformation algorithm imposes some restrictions on the
    order in which transformation commands are applied. `TransformationLayout`
    addresses these issues.
    """

    def __init__(self) -> None:
        """
        Initialize Transformation Layout.
        """
        self._transformations: List[TransformationCommand] = []

    @property
    def transformations(self) -> List[TransformationCommand]:
        return self._transformations

    def register(self, transformation: TransformationCommand) -> None:
        """
        Registers the transformation command in the transformation layout.

        :param transformation: The transformation command to be registered in
            the transformation layout.
        """
        self.transformations.append(transformation)

    def update(self, other: "TransformationLayout") -> None:
        """
        D.update(other), updates D from other.

        :param other: Another transformation layout.
        """
        for transformation in other.transformations:
            self.register(transformation)
