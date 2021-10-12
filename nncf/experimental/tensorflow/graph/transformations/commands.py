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

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType


class TFTargetPoint(TargetPoint):
    """
    Describes where the compression operation should be placed.
    """

    def __init__(self,
                 op_name: str,
                 op_type_name: str,
                 port_id: int,
                 target_type: TargetType):
        """
        Initializes target point for TensorFlow backend.

        :param op_name: Name of a node in the `FuncGraph`.
        :param op_type_name: Type of operation.
        :param port_id: Port id.
        :param target_type: Type of the target point.
        """
        super().__init__(target_type)
        self.op_name = op_name
        self.op_type_name = op_type_name
        self.port_id = port_id
