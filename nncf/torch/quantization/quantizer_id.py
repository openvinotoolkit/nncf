"""
 Copyright (c) 2020 Intel Corporation
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
from nncf.common.graph.graph import NNCFNodeName


class QuantizerId:
    """
    Unique identifier of a quantizer. It's used to store and search all quantizers in a single
    structure. Also it provides the scope, where the quantizer was inserted.
    """

    def get_base(self):
        raise NotImplementedError

    def get_suffix(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return str(self.get_base()) + self.get_suffix()

    def __hash__(self):
        return hash((self.get_base(), self.get_suffix()))

    def __eq__(self, other: 'QuantizerId'):
        return (self.get_base() == other.get_base()) and (self.get_suffix() == other.get_suffix())


class WeightQuantizerId(QuantizerId):
    """ Unique identifier of a quantizer for weights."""

    def __init__(self, target_node_name: NNCFNodeName):
        self.target_node_name = target_node_name

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return '|WEIGHT'


class NonWeightQuantizerId(QuantizerId):
    """
    Unique identifier of a quantizer, which corresponds to non-weight operations, such as
    ordinary activation, function and input
    """

    def __init__(self, target_node_name: NNCFNodeName,
                 input_port_id=None):
        self.target_node_name = target_node_name
        self.input_port_id = input_port_id

    def get_base(self) -> str:
        return self.target_node_name

    def get_suffix(self) -> str:
        return '|OUTPUT' if self.input_port_id is None else '|INPUT{}'.format(self.input_port_id)
