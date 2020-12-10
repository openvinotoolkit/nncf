from collections import namedtuple, Counter

import torch
from copy import deepcopy
from enum import Enum
from typing import Dict, List

from nncf.nncf_network import InsertionInfo, InsertionPoint, InsertionType
from nncf.quantization.layers import QuantizerConfig, BaseQuantizer


class QuantizerSetupType(Enum):
    PATTERN_BASED = "pattern_based"
    PROPAGATION_BASED = "propagation_based"

    @staticmethod
    def from_str(quantizer_setup_type: str) -> 'QuantizerSetupType':
        if quantizer_setup_type == QuantizerSetupType.PATTERN_BASED.value:
            return QuantizerSetupType.PATTERN_BASED
        if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED.value:
            return QuantizerSetupType.PROPAGATION_BASED
        raise RuntimeError("Unknown quantizer setup type. Please select 'pattern_based' or 'propagation_based'.")


class QuantizationConstraints:
    REF_QCONF_OBJ = QuantizerConfig()

    def __init__(self, **kwargs):
        """Use attribute names of QuantizerConfig as arguments
        to set up constraints.
        E.g. QuantizationConstraint(bits=8, per_channel=True) will set up
        a constraint that corresponds to all 8-bit per-channel quantizers, either
        symmetric or asymmetric, either signed or unsigned."""

        for attr_name in kwargs:
            if not hasattr(QuantizationConstraints.REF_QCONF_OBJ, attr_name):
                raise RuntimeError("Invalid constraint - QuantizerConfig has no attribute '{}'".format(attr_name))
        self.qconf_attr_vs_constraint_dict = kwargs

    def apply_constraints_to(self, qconfig: QuantizerConfig) -> QuantizerConfig:
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if constraint is not None:
                setattr(qconfig, attr_name, constraint)
        return qconfig

    def is_config_compatible(self, qconfig: QuantizerConfig) -> bool:
        is_compatible = True
        for attr_name, constraint in self.qconf_attr_vs_constraint_dict.items():
            if attr_name == 'logarithm_scale':
                continue  # Scale storage type is internal and should not affect HW config matching
            if constraint is not None:
                qconf_attr_value = getattr(qconfig, attr_name)
                if qconf_attr_value != constraint:
                    is_compatible = False
        return is_compatible

    def get_updated_constraints(self, overriding_constraints: 'QuantizationConstraints') -> 'QuantizationConstraints':
        new_dict = deepcopy(self.qconf_attr_vs_constraint_dict)
        new_dict.update(overriding_constraints.qconf_attr_vs_constraint_dict)
        return QuantizationConstraints(**new_dict)

    @classmethod
    def from_config_dict(cls, config_dict: Dict):
        return cls(bits=config_dict.get("bits"),
                   mode=config_dict.get("mode"),
                   per_channel=config_dict.get("per_channel"),
                   signedness_to_force=config_dict.get("signed"),
                   logarithm_scale=config_dict.get("logarithm_scale"))

    def constrain_qconfig_list(self, quantizer_config_list: List[QuantizerConfig]) -> List[QuantizerConfig]:
        assert quantizer_config_list is not None

        constrained_quantizer_config_list = list(filter(
            self.is_config_compatible,
            quantizer_config_list
        ))

        # TODO: Make the logic more flexible when the flag "warning as error" is implemented.
        # It means that the qconfig from overrides must be selected as final config
        # even if it is not valid in hw-config.
        if not constrained_quantizer_config_list:
            raise RuntimeError()

        return constrained_quantizer_config_list


class QuantizerGroup(Enum):
    ACTIVATIONS = "activations"
    WEIGHTS = "weights"


QuantizableModule = namedtuple('QuantizableModule', 'module module_scope qconfig_list')


class NonWeightQuantizerInfo:
    def __init__(self, quantizer_module_ref: BaseQuantizer,
                 affected_insertions: List[InsertionInfo]):
        self.quantizer_module_ref = quantizer_module_ref
        self.affected_insertions = affected_insertions


class WeightQuantizerInfo:
    def __init__(self,
                 quantizer_module_ref: BaseQuantizer,
                 quantized_module: torch.nn.Module):
        self.quantizer_module_ref = quantizer_module_ref
        self.quantized_module = quantized_module


class QuantizersBetweenQuantizableLayers:
    """ Contains locations of quantizers between inputs quantizable layers: input agnostic operation execution context
    for activations and scope - for quantized modules """

    def __init__(self):
        self.activation_quantizer_insertion_points = set()  # type: Set[InsertionPoint]
        self.quantized_module_scopes = set()  # type: Set['Scope']

    def add_activation_quantizer_insertion_point(self, ip: InsertionPoint):
        self.activation_quantizer_insertion_points.add(ip)

    def add_quantized_module_scope(self, scope: 'Scope'):
        self.quantized_module_scopes.add(scope)

    def __bool__(self) -> bool:
        return bool(self.activation_quantizer_insertion_points) and bool(self.quantized_module_scopes)

    def update(self, other: 'QuantizersBetweenQuantizableLayers'):
        self.activation_quantizer_insertion_points.update(other.activation_quantizer_insertion_points)
        self.quantized_module_scopes.update(other.quantized_module_scopes)


QuantizationPointId = int


class QuantizationPointBase:
    def __init__(self, insertion_point: InsertionPoint):
        self.insertion_point = insertion_point
    def is_activation_quantization_point(self) -> bool:
        return self.insertion_point.insertion_type == InsertionType.OPERATOR_PRE_HOOK or \
               self.insertion_point.insertion_type == InsertionType.OPERATOR_POST_HOOK

    def is_weight_quantization_point(self) -> bool:
        return self.insertion_point.insertion_type == InsertionType.NNCF_MODULE_PRE_OP

    def assign_input_shape(self, input_shape):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class SingleConfigQuantizationPoint(QuantizationPointBase):
    def __init__(self, insertion_point: InsertionPoint, qconfig: QuantizerConfig):
        super().__init__(insertion_point)
        self.qconfig = deepcopy(qconfig)

    def assign_input_shape(self, input_shape):
        self.qconfig.input_shape = input_shape

    def __str__(self):
        return str(self.insertion_point) + ' ' + str(self.qconfig)

class MultiConfigQuantizationPoint(QuantizationPointBase):
    def __init__(self, insertion_point: InsertionPoint, possible_qconfigs: List[QuantizerConfig]):
        super().__init__(insertion_point)
        self.possible_qconfigs = deepcopy(possible_qconfigs)

    def select_qconfig(self, qconfig: QuantizerConfig) -> SingleConfigQuantizationPoint:
        if qconfig not in self.possible_qconfigs:
            raise ValueError("Invalid selection for a quantizer config!")
        return SingleConfigQuantizationPoint(self.insertion_point, qconfig)

    def assign_input_shape(self, input_shape):
        for qconfig in self.possible_qconfigs:
            qconfig.input_shape = input_shape

    def __str__(self):
        return str(self.insertion_point) + ' ' + ';'.join([str(qc) for qc in self.possible_qconfigs])

class QuantizerSetupBase:
    def __init__(self):
        self.quantization_points = {}  # type: Dict[QuantizationPointId, QuantizationPointBase]
        self.unified_scale_groups = []  # type: List[Set[QuantizationPointId]]
        self.shared_input_operation_set_groups = []  # type: List[Set[QuantizationPointId]]

    def add_independent_quantization_point(self, qp: QuantizationPointBase):
        if self.quantization_points.keys():
            new_id = max(self.quantization_points.keys()) + 1
        else:
            new_id = 0
        self.quantization_points[new_id] = qp

    def add_unified_scale_group(self, qp_group: List[QuantizationPointBase]):
        new_start_id = max(self.quantization_points.keys()) + 1
        new_points_dict = {new_start_id + i: qp for i, qp in enumerate(qp_group)}
        self.quantization_points.update(new_points_dict)
        self.unified_scale_groups.append(set(new_points_dict.keys()))

    def __discard_independent(self, id_: QuantizationPointId):
        if id_ in self.quantization_points:
            self.quantization_points.pop(id_)
        for unified_scale_group in self.unified_scale_groups:
            unified_scale_group.discard(id_)

    def discard(self, id_: QuantizationPointId, keep_shared_input_qps: bool = False):
        if id_ in self.quantization_points:
            self.__discard_independent(id_)

            # If a quantizer from the shared operation set group is removed, then the
            # entire group has to be removed from the setup, otherwise an operation would have a mix between
            # quantized and unquantized inputs.
            indices_to_delete = []
            for idx, shared_input_operation_set_group in enumerate(self.shared_input_operation_set_groups):
                if id_ in shared_input_operation_set_group:
                    shared_input_operation_set_group.discard(id_)
                    indices_to_delete.append(idx)

            if not keep_shared_input_qps:
                for idx in sorted(indices_to_delete, reverse=True):
                    for additional_id in self.shared_input_operation_set_groups[idx]:
                        self.__discard_independent(additional_id)
                    del self.shared_input_operation_set_groups[idx]

    def mark_activation_quantizer_configs_with_input_shapes(self, original_nncf_graph: 'NNCFGraph'):
        for qp in self.quantization_points.values():
            insertion_point = qp.insertion_point
            if qp.is_activation_quantization_point():
                ia_op_exec_context = qp.insertion_point.ia_op_exec_context
                # TODO: use insertion_info.shape_to_operate_on?
                if insertion_point.input_port_id is not None:
                    quantizer_input_shape = original_nncf_graph.get_input_shapes_for_ia_op_exec_context(
                        ia_op_exec_context)[insertion_point.input_port_id]
                else:
                    # Tailored for post-hook quantization and first output quantization only
                    quantizer_input_shape = original_nncf_graph.get_output_shapes_for_ia_op_exec_context(
                        ia_op_exec_context)[0]

                qp.assign_input_shape(quantizer_input_shape)


class SingleConfigQuantizerSetup(QuantizerSetupBase):
    def __init__(self):
        super().__init__()
        self.quantization_points = {}  # type: Dict[QuantizationPointId, SingleConfigQuantizationPoint]


class MultiConfigQuantizerSetup(QuantizerSetupBase):
    def __init__(self):
        super().__init__()
        self.quantization_points = {}  # type: Dict[QuantizationPointId, MultiConfigQuantizationPoint]

    def select_qconfigs(self, qp_id_vs_selected_qconfig_dict: Dict[QuantizationPointId, QuantizerConfig]) -> \
            SingleConfigQuantizerSetup:
        retval = SingleConfigQuantizerSetup()
        retval.unified_scale_groups = deepcopy(self.unified_scale_groups)
        retval.shared_input_operation_set_groups = deepcopy(self.shared_input_operation_set_groups)
        if Counter(qp_id_vs_selected_qconfig_dict.keys()) != Counter(self.quantization_points.keys()):
            raise ValueError("The set of quantization points for a selection is inconsistent with quantization"
                             "points in the quantizer setup!")
        for qp_id in self.quantization_points:
            retval.quantization_points[qp_id] = self.quantization_points[qp_id].select_qconfig(
                qp_id_vs_selected_qconfig_dict[qp_id]
            )
        return retval

    def select_first_qconfig_for_each_point(self) -> SingleConfigQuantizerSetup:
        qp_id_vs_qconfig_dict = {}  # type: Dict[QuantizationPointId, QuantizerConfig]
        for qp_id, qp in self.quantization_points.items():
            qp_id_vs_qconfig_dict[qp_id] = qp.possible_qconfigs[0]
        return self.select_qconfigs(qp_id_vs_qconfig_dict)
