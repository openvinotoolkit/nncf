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

from typing import List
import onnx

def find_ignored_scopes(disallowed_op_types: List[str], model: onnx.ModelProto) -> List[str]:
    disallowed_op_types = set(disallowed_op_types)
    return [node.name for node in model.graph.node if node.op_type in disallowed_op_types]
