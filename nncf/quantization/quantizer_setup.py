from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Dict

from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import InsertionPoint, InsertionType
from nncf.quantization.layers import QuantizerConfig
from nncf.tensor_statistics.collectors import ReductionShape
from nncf.tensor_statistics.statistics import TensorStatistic, MinMaxTensorStatistic
from nncf.utils import get_scale_shape

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

    def get_all_scale_shapes(self) -> List[Tuple[int]]:
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

    def get_all_scale_shapes(self) -> List[Tuple[int]]:
        return [tuple(get_scale_shape(
            self.qconfig.input_shape,
            is_weights=self.is_weight_quantization_point(), per_channel=self.qconfig.per_channel))]


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

    def get_all_scale_shapes(self) -> List[Tuple[int]]:
        scale_shapes_across_configs = set()  # type: Set[Tuple[int]]
        for qc in self.possible_qconfigs:
            scale_shapes_across_configs.add(tuple(get_scale_shape(
                qc.input_shape,
                is_weights=self.is_weight_quantization_point(), per_channel=qc.per_channel)))
        return list(scale_shapes_across_configs)


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

    def get_minmax_values(self, tensor_statistics: Dict[InsertionPoint, Dict[ReductionShape, TensorStatistic]]) -> \
            Dict[QuantizationPointId, MinMaxTensorStatistic]:
        retval = {}
        for qp_id, qp in self.quantization_points.items():
            ip = qp.insertion_point
            if ip not in tensor_statistics:
                nncf_logger.debug("IP {} not found in tensor statistics".format(ip))
                retval[qp_id] = None
            else:
                input_shape = self.quantization_points[qp_id].qconfig.input_shape
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
