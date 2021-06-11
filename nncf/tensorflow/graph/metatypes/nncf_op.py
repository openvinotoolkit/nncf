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

from typing import List

from nncf.common.graph import NNCFGraphNodeType
from nncf.common.graph.operator_metatypes import OperatorMetatype


class InputNoopMetatype(OperatorMetatype):
    name = "input_noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [NNCFGraphNodeType.INPUT_NODE]


class OutputNoopMetatype(OperatorMetatype):
    name = "output_noop"

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return [NNCFGraphNodeType.OUTPUT_NODE]
