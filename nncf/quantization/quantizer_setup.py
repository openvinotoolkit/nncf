from collections import Counter
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.quantization.layers import QuantizerConfig
from nncf.tensor_statistics.collectors import ReductionShape
from nncf.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.tensor_statistics.statistics import TensorStatistic
from nncf.utils import get_scale_shape

QuantizationPointId = int


class QuantizationPointBase:
    def __init__(self, insertion_point: PTTargetPoint,
                 scopes_of_directly_quantized_operators: List['Scope']):
        self.insertion_point = insertion_point
        self.scopes_of_directly_quantized_operators = scopes_of_directly_quantized_operators

    def is_activation_quantization_point(self) -> bool:
        return self.insertion_point.target_type == TargetType.OPERATOR_PRE_HOOK or \
               self.insertion_point.target_type == TargetType.OPERATOR_POST_HOOK

    def is_weight_quantization_point(self) -> bool:
        return self.insertion_point.target_type == TargetType.OPERATION_WITH_WEIGHTS

    def get_all_scale_shapes(self, input_shape: Tuple[int]) -> List[Tuple[int]]:
        raise NotImplementedError

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SingleConfigQuantizationPoint(QuantizationPointBase):
    def __init__(self, insertion_point: PTTargetPoint, qconfig: QuantizerConfig,
                 scopes_of_directly_quantized_operators: List['Scope']):
        super().__init__(insertion_point, scopes_of_directly_quantized_operators)
        self.qconfig = deepcopy(qconfig)

    def __str__(self):
        return str(self.insertion_point) + ' ' + str(self.qconfig)

    def get_all_scale_shapes(self, input_shape: Tuple[int]) -> List[Tuple[int]]:
        return [tuple(get_scale_shape(
            input_shape,
            is_weights=self.is_weight_quantization_point(), per_channel=self.qconfig.per_channel))]


class MultiConfigQuantizationPoint(QuantizationPointBase):
    def __init__(self, insertion_point: PTTargetPoint, possible_qconfigs: List[QuantizerConfig],
                 scopes_of_directly_quantized_operators: List['Scope']):
        super().__init__(insertion_point, scopes_of_directly_quantized_operators)
        self.possible_qconfigs = possible_qconfigs

    @property
    def possible_qconfigs(self):
        return deepcopy(self._possible_qconfigs)

    @possible_qconfigs.setter
    def possible_qconfigs(self, qconfigs: List[QuantizerConfig]):
        self._possible_qconfigs = deepcopy(qconfigs)

    def select_qconfig(self, qconfig: QuantizerConfig) -> SingleConfigQuantizationPoint:
        if qconfig not in self.possible_qconfigs:
            raise ValueError("Invalid selection for a quantizer config!")
        return SingleConfigQuantizationPoint(self.insertion_point, qconfig, self.scopes_of_directly_quantized_operators)

    def __str__(self):
        return str(self.insertion_point) + ' ' + ';'.join([str(qc) for qc in self.possible_qconfigs])

    def get_all_scale_shapes(self, input_shape: Tuple[int]) -> List[Tuple[int]]:
        scale_shapes_across_configs = set()  # type: Set[Tuple[int]]
        for qc in self.possible_qconfigs:
            scale_shapes_across_configs.add(tuple(get_scale_shape(
                list(input_shape),
                is_weights=self.is_weight_quantization_point(), per_channel=qc.per_channel)))
        return list(scale_shapes_across_configs)


class QuantizerSetupBase:
    def __init__(self):
        self.quantization_points = {}  # type: Dict[QuantizationPointId, QuantizationPointBase]
        self.unified_scale_groups = {}  # type: Dict[int, Set[QuantizationPointId]]
        self.shared_input_operation_set_groups = {}  # type: Dict[int, Set[QuantizationPointId]]
        self._next_unified_scale_gid = 0
        self._next_shared_inputs_gid = 0

    def add_independent_quantization_point(self, qp: QuantizationPointBase):
        if self.quantization_points.keys():
            new_id = max(self.quantization_points.keys()) + 1
        else:
            new_id = 0
        self.quantization_points[new_id] = qp

    def register_unified_scale_group(self, qp_group: List[QuantizationPointId]) -> int:
        for qp_id in qp_group:
            gid = self.get_unified_scale_group_id(qp_id) is not None
            if gid:
                raise RuntimeError("QP id {} is already in unified scale group {}".format(qp_id, gid))
        gid = self._next_unified_scale_gid
        self.unified_scale_groups[self._next_unified_scale_gid] = set(qp_group)
        self._next_unified_scale_gid += 1
        return gid

    def register_shared_inputs_group(self, qp_group: List[QuantizationPointId]) -> int:
        for qp_id in qp_group:
            gid = self.get_shared_inputs_group_id(qp_id) is not None
            if gid:
                raise RuntimeError("QP id {} is already in shared input group {}".format(qp_id, gid))
        gid = self._next_shared_inputs_gid
        self.shared_input_operation_set_groups[self._next_shared_inputs_gid] = set(qp_group)
        self._next_shared_inputs_gid += 1
        return gid

    def __discard_independent(self, id_: QuantizationPointId):
        if id_ in self.quantization_points:
            self.quantization_points.pop(id_)
        for unified_scale_group in self.unified_scale_groups.values():
            unified_scale_group.discard(id_)

    def discard(self, id_: QuantizationPointId, keep_shared_input_qps: bool = False):
        if id_ in self.quantization_points:
            self.__discard_independent(id_)

            # If a quantizer from the shared operation set group is removed, then the
            # entire group has to be removed from the setup, otherwise an operation would have a mix between
            # quantized and unquantized inputs.
            indices_to_delete = []
            for gid, shared_input_operation_set_group in self.shared_input_operation_set_groups.items():
                if id_ in shared_input_operation_set_group:
                    shared_input_operation_set_group.discard(id_)
                    indices_to_delete.append(gid)

            if not keep_shared_input_qps:
                for idx in sorted(indices_to_delete, reverse=True):
                    for additional_id in self.shared_input_operation_set_groups[idx]:
                        self.__discard_independent(additional_id)
                    del self.shared_input_operation_set_groups[idx]

    def get_unified_scale_group_id(self,
                                   qp_id: QuantizationPointId) -> Optional[int]:
        for gid, unified_scale_group in self.unified_scale_groups.items():
            if qp_id in unified_scale_group:
                return gid
        return None

    def get_shared_inputs_group_id(self,
                                   qp_id: QuantizationPointId) -> Optional[int]:
        for gid, shared_inputs_group in self.shared_input_operation_set_groups.items():
            if qp_id in shared_inputs_group:
                return gid
        return None

    def register_existing_qp_id_in_unified_scale_group(self, qp_id: QuantizationPointId, unified_scale_gid: int):
        gid = self.get_unified_scale_group_id(qp_id)
        if gid is not None:
            raise RuntimeError("QP id {} is already in unified scale group {}".format(qp_id, gid))
        self.unified_scale_groups[unified_scale_gid].add(qp_id)

    def register_existing_qp_id_in_shared_input_group(self, qp_id: QuantizationPointId, shared_inputs_gid: int):
        gid = self.get_shared_inputs_group_id(qp_id)
        if gid is not None:
            raise RuntimeError("QP id {} is already in shared inputs group {}".format(qp_id, gid))
        self.shared_input_operation_set_groups[shared_inputs_gid].add(qp_id)

    def equivalent_to(self, other: 'QuantizerSetupBase') -> bool:
        this_qp_id_to_other_qp_id_dict = {}  # type: Dict[QuantizationPointId, QuantizationPointId]

        def _compare_qps(first: 'QuantizerSetupBase', second: 'QuantizerSetupBase') -> bool:
            for this_qp_id, this_qp in first.quantization_points.items():
                matches = []  # type: List[QuantizationPointId]
                for other_qp_id, other_qp in second.quantization_points.items():
                    if this_qp == other_qp:
                        matches.append(other_qp_id)
                if len(matches) == 0:
                    return False
                assert len(matches) == 1  # separate quantization points should not compare equal to each other
                this_qp_id_to_other_qp_id_dict[this_qp_id] = matches[0]
            return True

        def _compare_shared_input_groups(first: 'QuantizerSetupBase', second: 'QuantizerSetupBase') -> bool:
            for this_same_input_group_set in first.shared_input_operation_set_groups.values():
                translated_id_set = set(this_qp_id_to_other_qp_id_dict[this_qp_id]
                                        for this_qp_id in this_same_input_group_set)
                matches = []

                for other_shared_inputs_group in second.shared_input_operation_set_groups.values():
                    if translated_id_set == other_shared_inputs_group:
                        matches.append(other_shared_inputs_group)
                if not matches:
                    return False
                assert len(matches) == 1  # shared inputs group entries should be present in only one group
            return True

        def _compare_unified_scale_groups(first: 'QuantizerSetupBase', second: 'QuantizerSetupBase') -> bool:
            for this_unified_scales_group in first.unified_scale_groups.values():
                translated_id_set = set(this_qp_id_to_other_qp_id_dict[this_qp_id]
                                        for this_qp_id in this_unified_scales_group)
                matches = []
                for other_unified_scales_group in second.unified_scale_groups.values():
                    if translated_id_set == other_unified_scales_group:
                        matches.append(other_unified_scales_group)
                if not matches:
                    return False
                assert len(matches) == 1  # unified scale group entries should be present in only one group
            return True

        return _compare_qps(self, other) and _compare_qps(other, self) and \
               _compare_shared_input_groups(self, other) and _compare_shared_input_groups(self, other) and \
               _compare_unified_scale_groups(self, other) and _compare_unified_scale_groups(self, other)


class SingleConfigQuantizerSetup(QuantizerSetupBase):
    def __init__(self):
        super().__init__()
        self.quantization_points = {}  # type: Dict[QuantizationPointId, SingleConfigQuantizationPoint]

    def get_minmax_values(self,
                          tensor_statistics: Dict[PTTargetPoint, Dict[ReductionShape, TensorStatistic]],
                          target_model: NNCFNetwork) -> \
            Dict[QuantizationPointId, MinMaxTensorStatistic]:
        retval = {}
        for qp_id, qp in self.quantization_points.items():
            ip = qp.insertion_point
            if ip not in tensor_statistics:
                nncf_logger.debug("IP {} not found in tensor statistics".format(ip))
                retval[qp_id] = None
            else:
                if qp.is_weight_quantization_point():
                    module = target_model.get_module_by_scope(qp.insertion_point.module_scope)
                    input_shape = module.weight.shape
                else:
                    input_shape = target_model.get_input_shape_for_insertion_point(qp.insertion_point)
                scale_shape = tuple(get_scale_shape(input_shape,
                                                    qp.is_weight_quantization_point(),
                                                    qp.qconfig.per_channel))
                if scale_shape not in tensor_statistics[ip]:
                    nncf_logger.debug("Did not collect tensor statistics at {} for shape {}".format(ip, scale_shape))
                    retval[qp_id] = None
                minmax_stat = MinMaxTensorStatistic.from_stat(tensor_statistics[ip][scale_shape])
                retval[qp_id] = minmax_stat
        return retval


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
