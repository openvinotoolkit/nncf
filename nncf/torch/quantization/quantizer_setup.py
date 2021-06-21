from collections import Counter
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.quantization.layers import QuantizerConfig
from nncf.torch.quantization.structs import UnifiedScaleType
from nncf.torch.tensor_statistics.collectors import ReductionShape
from nncf.torch.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import TensorStatistic
from nncf.torch.utils import get_scale_shape

QuantizationPointId = int


class QuantizationPointBase:
    def __init__(self, insertion_point: PTTargetPoint,
                 directly_quantized_operator_node_names: List[NNCFNodeName]):
        self.insertion_point = insertion_point
        self.directly_quantized_operator_node_names = directly_quantized_operator_node_names

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
                 directly_quantized_operator_node_names: List[NNCFNodeName]):
        super().__init__(insertion_point, directly_quantized_operator_node_names)
        self.qconfig = deepcopy(qconfig)

    def __str__(self):
        return str(self.insertion_point) + ' ' + str(self.qconfig)

    def get_all_scale_shapes(self, input_shape: Tuple[int]) -> List[Tuple[int]]:
        return [tuple(get_scale_shape(
            input_shape,
            is_weights=self.is_weight_quantization_point(), per_channel=self.qconfig.per_channel))]


class MultiConfigQuantizationPoint(QuantizationPointBase):
    def __init__(self, insertion_point: PTTargetPoint, possible_qconfigs: List[QuantizerConfig],
                 directly_quantized_operator_node_names: List[NNCFNodeName]):
        super().__init__(insertion_point, directly_quantized_operator_node_names)
        self.possible_qconfigs = possible_qconfigs

    @property
    def possible_qconfigs(self):
        return deepcopy(self._possible_qconfigs)

    @possible_qconfigs.setter
    def possible_qconfigs(self, qconfigs: List[QuantizerConfig]):
        self._possible_qconfigs = deepcopy(qconfigs)

    def select_qconfig(self, qconfig: QuantizerConfig) -> SingleConfigQuantizationPoint:
        if qconfig not in self.possible_qconfigs:
            # Allow selecting an "unsigned" or "signed" version if "any-signed" version is present
            qconfig_any = deepcopy(qconfig)
            qconfig_any.signedness_to_force = None
            if qconfig_any not in self.possible_qconfigs:
                raise ValueError("Invalid selection for a quantizer config - "
                                 "tried to select {} among [{}]".format(qconfig,
                                                                        ",".join(
                                                                            [str(q) for q in self.possible_qconfigs])))
            qconfig = qconfig_any
        return SingleConfigQuantizationPoint(self.insertion_point, qconfig, self.directly_quantized_operator_node_names)

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

    def remove_unified_scale_from_point(self, qp_id: QuantizationPointId):
        gid = self.get_unified_scale_group_id(qp_id)
        if gid is None:
            nncf_logger.debug("Attempted to remove QP id {} from associated unified scale group, but the QP"
                              "is not in any unified scale group - ignoring.".format(qp_id))
            return
        self.unified_scale_groups[gid].discard(qp_id)
        if not self.unified_scale_groups[gid]:
            nncf_logger.debug("Removed last entry from a unified scale group {} - removing group itself".format(gid))
            self.unified_scale_groups.pop(gid)

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
                          target_model_graph: NNCFGraph) -> \
            Dict[QuantizationPointId, MinMaxTensorStatistic]:
        retval = {}
        for qp_id, qp in self.quantization_points.items():
            ip = qp.insertion_point
            if ip not in tensor_statistics:
                nncf_logger.debug("IP {} not found in tensor statistics".format(ip))
                retval[qp_id] = None
            else:
                target_node = target_model_graph.get_node_by_name(qp.insertion_point.target_node_name)
                if qp.is_weight_quantization_point():
                    input_shape = target_node.layer_attributes.get_weight_shape()
                else:
                    input_shape = target_model_graph.get_input_shape_for_insertion_point(qp.insertion_point)
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
        self._unified_scale_qpid_vs_type = {}  # type: Dict[QuantizationPointId, UnifiedScaleType]

    def register_unified_scale_group_with_types(self, qp_group: List[QuantizationPointId],
                                                us_types: List[UnifiedScaleType]) -> int:
        assert len(qp_group) == len(us_types)
        gid = super().register_unified_scale_group(qp_group)
        for qp_id, us_type in zip(qp_group, us_types):
            self._unified_scale_qpid_vs_type[qp_id] = us_type
        return gid

    def select_qconfigs(self, qp_id_vs_selected_qconfig_dict: Dict[QuantizationPointId, QuantizerConfig],
                        strict: bool =True) -> \
            SingleConfigQuantizerSetup:
        retval = SingleConfigQuantizerSetup()
        retval.unified_scale_groups = deepcopy(self.unified_scale_groups)
        retval.shared_input_operation_set_groups = deepcopy(self.shared_input_operation_set_groups)

        if Counter(qp_id_vs_selected_qconfig_dict.keys()) != Counter(self.quantization_points.keys()):
            raise ValueError("The set of quantization points for a selection is inconsistent with quantization"
                             "points in the quantizer setup!")
        for qp_id in self.quantization_points:
            if strict:
                retval.quantization_points[qp_id] = self.quantization_points[qp_id].select_qconfig(
                    qp_id_vs_selected_qconfig_dict[qp_id]
                )
            else:
                multi_qp = self.quantization_points[qp_id]
                qconfig = qp_id_vs_selected_qconfig_dict[qp_id]
                retval.quantization_points[qp_id] = SingleConfigQuantizationPoint(
                    multi_qp.insertion_point, qconfig,
                    multi_qp.directly_quantized_operator_node_names)

        # Segregate the unified scale groups into sub-groups based on what exact config was chosen.
        for us_group in self.unified_scale_groups.values():
            per_channel_qids = set()
            per_tensor_qids = set()
            for us_qid in us_group:
                final_qconfig = retval.quantization_points[us_qid].qconfig
                if final_qconfig.per_channel:
                    per_channel_qids.add(us_qid)
                else:
                    per_tensor_qids.add(us_qid)

            if per_tensor_qids:
                for qid in per_tensor_qids:
                    retval.remove_unified_scale_from_point(qid)

                retval.register_unified_scale_group(list(per_tensor_qids))

            for per_channel_qid in per_channel_qids:
                us_type = self._unified_scale_qpid_vs_type[per_channel_qid]
                if us_type is UnifiedScaleType.UNIFY_ONLY_PER_TENSOR:
                    nncf_logger.debug("Per-channel quantizer config selected in a MultiConfigQuantizerSetup for a "
                                      "unified scale point that only supports per-tensor scale unification, disabling "
                                      "unified scales for this point.")
                retval.remove_unified_scale_from_point(per_channel_qid)

        return retval

    def select_first_qconfig_for_each_point(self) -> SingleConfigQuantizerSetup:
        qp_id_vs_qconfig_dict = {}  # type: Dict[QuantizationPointId, QuantizerConfig]
        for qp_id, qp in self.quantization_points.items():
            qp_id_vs_qconfig_dict[qp_id] = qp.possible_qconfigs[0]
        return self.select_qconfigs(qp_id_vs_qconfig_dict)

    @classmethod
    def from_single_config_setup(cls, single_conf_setup: SingleConfigQuantizerSetup) -> 'MultiConfigQuantizerSetup':
        retval = cls()
        for qp_id, qp in single_conf_setup.quantization_points.items():
            multi_pt = MultiConfigQuantizationPoint(
                insertion_point=qp.insertion_point,
                possible_qconfigs=[deepcopy(qp.qconfig)],
                directly_quantized_operator_node_names=qp.directly_quantized_operator_node_names)
            retval.quantization_points[qp_id] = multi_pt
        for qp_set in single_conf_setup.unified_scale_groups.values():
            qp_list = list(qp_set)
            qp_types = [UnifiedScaleType.UNIFY_ALWAYS for _ in qp_list]
            retval.register_unified_scale_group_with_types(qp_list,
                                                           qp_types)
        for qp_set in single_conf_setup.shared_input_operation_set_groups.values():
            qp_list = list(qp_set)
            retval.register_shared_inputs_group(qp_list)
        return retval
