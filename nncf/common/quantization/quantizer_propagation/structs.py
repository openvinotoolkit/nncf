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

from enum import Enum
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import UnifiedScaleType


class QuantizationTrait(Enum):
    """
    General, hardware-agnostic specifications for the relation of operators to quantization.
    Hardware-specific quantization configuration is handled elsewhere.
    """

    NON_QUANTIZABLE = -1
    QUANTIZATION_AGNOSTIC = 0
    INPUTS_QUANTIZABLE = 1

    # For embeddings, the inputs are integer and are used to index into the weight of the layer,
    # therefore if the weight is already quantized there may be no need to quantize the outgoing tensor.
    # TODO: unify scales for such operations if they become linked through a downstream op (such as
    # two embeddings added to each other)
    OUTPUT_QUANTIZATION_AS_WEIGHTS = 2

    # A concat node should ultimately either have all of its inputs quantized with the same configuration
    # or none at all. This cannot be determined ahead of time, hence the special trait.
    CONCAT = 100


class PropagatingQuantizer:
    """
    Used in conjunction with QuantizerPropagationStateGraph to keep track of
    the allowed quantization configs corresponding to the model operation node
    whose inputs it quantizes, and also of the nodes/edges in the model control
    graph that this quantizer affects. It should be moved against the data flow of
    the model, tracking the affected nodes and edges of
    QuantizerPropagationStateGraph. No actual quantization modules are used here,
    only the associated configs (such as bitwidths, modes, signed/unsigned
    attributes etc.)
    """

    def __init__(self, id_: int, quant_configs: List[QuantizerConfig], init_location_node_key: str,
                 unified_scale_type: Optional[UnifiedScaleType] = None):
        """
        :param id_: The unique identifier of the new propagating quantizer.
        :param quant_configs: The quantizer configurations that this quantizer currently allows.
        :param init_location_node_key: The node key in QuantizerPropagationStateGraph that this
          quantizer is being inserted into.
        :param unified_scale_type: The type of unified scales for this quantizer - if unspecified,
          this quantizer won't require unified scales.
        """
        self.potential_quant_configs = quant_configs  # type: List[QuantizerConfig]
        self.affected_edges = set()
        self.affected_ip_nodes = set()  # type: Set[str]
        self.propagation_path = []  # type: PropagationPath
        self.current_location_node_key = init_location_node_key
        self.last_accepting_location_node_key = None
        self.id = id_
        self.unified_scale_type = unified_scale_type
        self.affected_operator_nodes = set()
        self.quantized_input_sink_operator_nodes = set()
        self.downstream_propagating_quantizers = set()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class QuantizerPropagationStateGraphNodeType(Enum):
    PRE_HOOK = 0
    POST_HOOK = 1
    OPERATOR = 2
    AUXILIARY_BARRIER = 3


class SharedAffectedOpsPropagatingQuantizerGroup:
    """ Combines propagating quantizers that share affected operations """
    def __init__(self, affecting_prop_quants: Set[PropagatingQuantizer], affected_op_node_keys: Set[str]):
        self.affecting_prop_quants = affecting_prop_quants  # type: Set[PropagatingQuantizer]
        self.affected_op_node_keys = affected_op_node_keys  # type: Set[str]

    def update(self, other: 'SharedAffectedOpsPropagatingQuantizerGroup'):
        self.affected_op_node_keys.update(other.affected_op_node_keys)
        self.affecting_prop_quants.update(other.affecting_prop_quants)


PropagationPath = List[Tuple[str, str]]
