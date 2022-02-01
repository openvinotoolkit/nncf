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

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.graph.transformations.layout import TransformationLayout


class TFTransformationLayoutV2(TransformationLayout):
    def register(self, transformation: TransformationCommand) -> None:
        """
        Registers the transformation command in the transformation layout.

        The `TFTransformationLayoutV2` is a simplified version of the
        `TransformationLayout` class where some redundant functionality
        was removed.

        :param transformation: The transformation command to be registered in
            the transformation layout.
        """
        if transformation.type == TransformationType.REMOVE:
            # TODO(andrey-churkin): Add support.
            pass
        elif transformation.type == TransformationType.INSERT:
            self._register_insertion_transformation(transformation)
        else:
            raise ValueError(f'Unknown type of transformation command: {transformation.type}')

    def _register_insertion_transformation(self, transformation: TransformationCommand) -> None:
        idx = None
        for curr_idx, t in enumerate(self.transformations):
            if t.check_command_compatibility(transformation):
                assert idx is None
                idx = curr_idx

        if idx is not None:
            self.transformations[idx] = self.transformations[idx].union(transformation)
        else:
            self.transformations.append(transformation)
