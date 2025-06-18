# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from collections import Counter
from copy import deepcopy
from enum import Enum
from typing import Any, Optional

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import UnifiedScaleType
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.stateful_classes_registry import CommonStatefulClassesRegistry
from nncf.config.schemata.defaults import QUANTIZATION_NARROW_RANGE

QuantizationPointId = int

DEFAULT_QUANTIZER_CONFIG = QuantizerConfig(
    num_bits=8,
    mode=QuantizationMode.SYMMETRIC,
    signedness_to_force=None,
    per_channel=False,
    narrow_range=QUANTIZATION_NARROW_RANGE,
)


class QuantizationPointType(Enum):
    WEIGHT_QUANTIZATION = 0
    ACTIVATION_QUANTIZATION = 1


class QIPointStateNames:
    TARGET_NODE_NAME = "target_node_name"


class QuantizationInsertionPointBase(ABC):
    _state_names = QIPointStateNames

    def __init__(self, target_node_name: NNCFNodeName):
        self.target_node_name = target_node_name

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {self._state_names.TARGET_NODE_NAME: self.target_node_name}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "QuantizationInsertionPointBase":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


@CommonStatefulClassesRegistry.register()
class WeightQuantizationInsertionPoint(QuantizationInsertionPointBase):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, WeightQuantizationInsertionPoint) and self.target_node_name == other.target_node_name

    def __str__(self) -> str:
        return str(WeightQuantizerId(self.target_node_name))

    def __hash__(self) -> int:
        return hash(str(self))


class AQIPointStateNames:
    INPUT_PORT_ID = "input_port_id"
    TARGET_NODE_NAME = "target_node_name"


@CommonStatefulClassesRegistry.register()
class ActivationQuantizationInsertionPoint(QuantizationInsertionPointBase):
    _state_names = AQIPointStateNames  # type: ignore[assignment]

    def __init__(self, target_node_name: NNCFNodeName, input_port_id: Optional[int] = None):
        super().__init__(target_node_name)
        self.input_port_id = input_port_id

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ActivationQuantizationInsertionPoint)
            and self.target_node_name == other.target_node_name
            and self.input_port_id == other.input_port_id
        )

    def __str__(self) -> str:
        return str(NonWeightQuantizerId(self.target_node_name, self.input_port_id))

    def __hash__(self) -> int:
        return hash(str(self))

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            AQIPointStateNames.TARGET_NODE_NAME: self.target_node_name,
            AQIPointStateNames.INPUT_PORT_ID: self.input_port_id,
        }


class QuantizationPointBase:
    def __init__(
        self,
        quant_insertion_point: QuantizationInsertionPointBase,
        directly_quantized_operator_node_names: list[NNCFNodeName],
    ):
        self.insertion_point = quant_insertion_point
        self.directly_quantized_operator_node_names = directly_quantized_operator_node_names

    def is_activation_quantization_point(self) -> bool:
        return not self.is_weight_quantization_point()

    def is_weight_quantization_point(self) -> bool:
        return isinstance(self.insertion_point, WeightQuantizationInsertionPoint)

    def get_all_configs_list(self) -> list[QuantizerConfig]:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return bool(self.__dict__ == other.__dict__)


class SCQPointStateNames:
    QCONFIG = "qconfig"
    INSERTION_POINT = "qip"
    INSERTION_POINT_CLASS_NAME = "qip_class"
    NAMES_OF_QUANTIZED_OPS = "directly_quantized_operator_node_names"


class SingleConfigQuantizationPoint(QuantizationPointBase):
    _state_names = SCQPointStateNames

    def __init__(
        self,
        qip: QuantizationInsertionPointBase,
        qconfig: QuantizerConfig,
        directly_quantized_operator_node_names: list[NNCFNodeName],
    ):
        super().__init__(qip, directly_quantized_operator_node_names)
        self.qconfig = deepcopy(qconfig)

    def __str__(self) -> str:
        return str(self.insertion_point) + " " + str(self.qconfig)

    def get_all_configs_list(self) -> list[QuantizerConfig]:
        return [self.qconfig]

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.INSERTION_POINT: self.insertion_point.get_state(),
            self._state_names.INSERTION_POINT_CLASS_NAME: self.insertion_point.__class__.__name__,
            self._state_names.QCONFIG: self.qconfig.get_state(),
            self._state_names.NAMES_OF_QUANTIZED_OPS: self.directly_quantized_operator_node_names,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SingleConfigQuantizationPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        insertion_point_cls_name = state[cls._state_names.INSERTION_POINT_CLASS_NAME]
        insertion_point_cls = CommonStatefulClassesRegistry.get_registered_class(insertion_point_cls_name)
        insertion_point = insertion_point_cls.from_state(state[cls._state_names.INSERTION_POINT])  # type: ignore
        kwargs = {
            cls._state_names.INSERTION_POINT: insertion_point,
            cls._state_names.QCONFIG: QuantizerConfig.from_state(state[cls._state_names.QCONFIG]),
            cls._state_names.NAMES_OF_QUANTIZED_OPS: state[cls._state_names.NAMES_OF_QUANTIZED_OPS],
        }
        return cls(**kwargs)


class MultiConfigQuantizationPoint(QuantizationPointBase):
    def __init__(
        self,
        qip: QuantizationInsertionPointBase,
        possible_qconfigs: list[QuantizerConfig],
        directly_quantized_operator_node_names: list[NNCFNodeName],
    ):
        super().__init__(qip, directly_quantized_operator_node_names)
        self.possible_qconfigs = possible_qconfigs

    @property
    def possible_qconfigs(self) -> list[QuantizerConfig]:
        return deepcopy(self._possible_qconfigs)

    @possible_qconfigs.setter
    def possible_qconfigs(self, qconfigs: list[QuantizerConfig]) -> None:
        self._possible_qconfigs = deepcopy(qconfigs)

    def select_qconfig(self, qconfig: QuantizerConfig) -> SingleConfigQuantizationPoint:
        if qconfig not in self.possible_qconfigs:
            # Allow selecting an "unsigned" or "signed" version if "any-signed" version is present
            qconfig_any = deepcopy(qconfig)
            qconfig_any.signedness_to_force = None
            if qconfig_any not in self.possible_qconfigs:
                msg = "Invalid selection for a quantizer config - tried to select {} among [{}]".format(
                    qconfig, ",".join([str(q) for q in self.possible_qconfigs])
                )
                raise ValueError(msg)
            qconfig = qconfig_any
        return SingleConfigQuantizationPoint(self.insertion_point, qconfig, self.directly_quantized_operator_node_names)

    def __str__(self) -> str:
        return str(self.insertion_point) + " " + ";".join([str(qc) for qc in self.possible_qconfigs])

    def get_all_configs_list(self) -> list[QuantizerConfig]:
        return self.possible_qconfigs


class QuantizerSetupBase:
    def __init__(self) -> None:
        self.quantization_points: dict[QuantizationPointId, QuantizationPointBase] = {}
        self.unified_scale_groups: dict[int, set[QuantizationPointId]] = {}
        self.shared_input_operation_set_groups: dict[int, set[QuantizationPointId]] = {}
        self._next_unified_scale_gid = 0
        self._next_shared_inputs_gid = 0

    def add_independent_quantization_point(self, qp: QuantizationPointBase) -> int:
        if self.quantization_points.keys():
            new_id = max(self.quantization_points.keys()) + 1
        else:
            new_id = 0
        self.quantization_points[new_id] = qp
        return new_id

    def register_unified_scale_group(self, qp_group: list[QuantizationPointId]) -> int:
        for qp_id in qp_group:
            usg_id = self.get_unified_scale_group_id(qp_id)
            if usg_id is not None:
                msg = f"QP id {qp_id} is already in unified scale group {usg_id}"
                raise nncf.InternalError(msg)
        gid = self._next_unified_scale_gid
        self.unified_scale_groups[self._next_unified_scale_gid] = set(qp_group)
        self._next_unified_scale_gid += 1
        return gid

    def register_shared_inputs_group(self, qp_group: list[QuantizationPointId]) -> int:
        for qp_id in qp_group:
            usg_id = self.get_shared_inputs_group_id(qp_id)
            if usg_id is not None:
                msg = f"QP id {qp_id} is already in unified scale group {usg_id}"
                raise nncf.InternalError(msg)
        gid = self._next_shared_inputs_gid
        self.shared_input_operation_set_groups[self._next_shared_inputs_gid] = set(qp_group)
        self._next_shared_inputs_gid += 1
        return gid

    def __discard_independent(self, id_: QuantizationPointId) -> None:
        if id_ in self.quantization_points:
            self.quantization_points.pop(id_)
        for unified_scale_group in self.unified_scale_groups.values():
            unified_scale_group.discard(id_)

    def discard(self, id_: QuantizationPointId, keep_shared_input_qps: bool = False) -> None:
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

    def get_unified_scale_group_id(self, qp_id: QuantizationPointId) -> Optional[int]:
        for gid, unified_scale_group in self.unified_scale_groups.items():
            if qp_id in unified_scale_group:
                return gid
        return None

    def get_shared_inputs_group_id(self, qp_id: QuantizationPointId) -> Optional[int]:
        for gid, shared_inputs_group in self.shared_input_operation_set_groups.items():
            if qp_id in shared_inputs_group:
                return gid
        return None

    def register_existing_qp_id_in_unified_scale_group(
        self, qp_id: QuantizationPointId, unified_scale_gid: int
    ) -> None:
        gid = self.get_unified_scale_group_id(qp_id)
        if gid is not None:
            msg = f"QP id {qp_id} is already in unified scale group {gid}"
            raise nncf.InternalError(msg)
        self.unified_scale_groups[unified_scale_gid].add(qp_id)

    def register_existing_qp_id_in_shared_input_group(self, qp_id: QuantizationPointId, shared_inputs_gid: int) -> None:
        gid = self.get_shared_inputs_group_id(qp_id)
        if gid is not None:
            msg = f"QP id {qp_id} is already in shared inputs group {gid}"
            raise nncf.InternalError(msg)
        self.shared_input_operation_set_groups[shared_inputs_gid].add(qp_id)

    def remove_unified_scale_from_point(self, qp_id: QuantizationPointId) -> None:
        gid = self.get_unified_scale_group_id(qp_id)
        if gid is None:
            nncf_logger.debug(
                f"Attempted to remove QP id {qp_id} from associated unified scale group, but the QP"
                f"is not in any unified scale group - ignoring."
            )
            return
        self.unified_scale_groups[gid].discard(qp_id)
        if not self.unified_scale_groups[gid]:
            nncf_logger.debug(f"Removed last entry from a unified scale group {gid} - removing group itself")
            self.unified_scale_groups.pop(gid)

    def equivalent_to(self, other: "QuantizerSetupBase") -> bool:
        this_qp_id_to_other_qp_id_dict: dict[QuantizationPointId, QuantizationPointId] = {}

        def _compare_qps(first: "QuantizerSetupBase", second: "QuantizerSetupBase") -> bool:
            for this_qp_id, this_qp in first.quantization_points.items():
                matches: list[QuantizationPointId] = []
                for other_qp_id, other_qp in second.quantization_points.items():
                    if this_qp == other_qp:
                        matches.append(other_qp_id)
                if len(matches) == 0:
                    return False
                assert len(matches) == 1  # separate quantization points should not compare equal to each other
                this_qp_id_to_other_qp_id_dict[this_qp_id] = matches[0]
            return True

        def _compare_shared_input_groups(first: "QuantizerSetupBase", second: "QuantizerSetupBase") -> bool:
            for this_same_input_group_set in first.shared_input_operation_set_groups.values():
                translated_id_set = set(
                    this_qp_id_to_other_qp_id_dict[this_qp_id] for this_qp_id in this_same_input_group_set
                )
                matches = []

                for other_shared_inputs_group in second.shared_input_operation_set_groups.values():
                    if translated_id_set == other_shared_inputs_group:
                        matches.append(other_shared_inputs_group)
                if not matches:
                    return False
                assert len(matches) == 1  # shared inputs group entries should be present in only one group
            return True

        def _compare_unified_scale_groups(first: "QuantizerSetupBase", second: "QuantizerSetupBase") -> bool:
            for this_unified_scales_group in first.unified_scale_groups.values():
                translated_id_set = set(
                    this_qp_id_to_other_qp_id_dict[this_qp_id] for this_qp_id in this_unified_scales_group
                )
                matches = []
                for other_unified_scales_group in second.unified_scale_groups.values():
                    if translated_id_set == other_unified_scales_group:
                        matches.append(other_unified_scales_group)
                if not matches:
                    return False
                assert len(matches) == 1  # unified scale group entries should be present in only one group
            return True

        return (
            _compare_qps(self, other)
            and _compare_qps(other, self)
            and _compare_shared_input_groups(self, other)
            and _compare_shared_input_groups(self, other)
            and _compare_unified_scale_groups(self, other)
            and _compare_unified_scale_groups(self, other)
        )


class SCQSetupStateNames:
    SHARED_INPUT_OPERATION_SET_GROUPS = "shared_input_operation_set_groups"
    UNIFIED_SCALE_GROUPS = "unified_scale_groups"
    QUANTIZATION_POINTS = "quantization_points"


class SingleConfigQuantizerSetup(QuantizerSetupBase):
    _state_names = SCQSetupStateNames

    def __init__(self) -> None:
        super().__init__()
        self.quantization_points: dict[QuantizationPointId, SingleConfigQuantizationPoint] = {}  # type: ignore

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """

        def set2list(pair: tuple[int, set[QuantizationPointId]]) -> tuple[int, list[QuantizationPointId]]:
            i, qp_id_set = pair
            return i, list(qp_id_set)

        quantization_points_state = {qp_id: qp.get_state() for qp_id, qp in self.quantization_points.items()}
        unified_scale_groups_state = dict(map(set2list, self.unified_scale_groups.items()))
        shared_input_operation_set_groups_state = dict(map(set2list, self.shared_input_operation_set_groups.items()))
        return {
            self._state_names.QUANTIZATION_POINTS: quantization_points_state,
            self._state_names.UNIFIED_SCALE_GROUPS: unified_scale_groups_state,
            self._state_names.SHARED_INPUT_OPERATION_SET_GROUPS: shared_input_operation_set_groups_state,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "SingleConfigQuantizerSetup":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        setup = SingleConfigQuantizerSetup()

        def decode_qp(pair: tuple[str, dict[str, Any]]) -> tuple[int, SingleConfigQuantizationPoint]:
            str_qp_id, qp_state = pair
            return int(str_qp_id), SingleConfigQuantizationPoint.from_state(qp_state)

        def list2set(pair: tuple[str, list[int]]) -> tuple[int, set[int]]:
            str_idx, qp_id_list = pair
            return int(str_idx), set(qp_id_list)

        setup.quantization_points = dict(map(decode_qp, state[cls._state_names.QUANTIZATION_POINTS].items()))
        setup.unified_scale_groups = dict(map(list2set, state[cls._state_names.UNIFIED_SCALE_GROUPS].items()))
        shared_input_operation_set_groups_state = state[cls._state_names.SHARED_INPUT_OPERATION_SET_GROUPS]
        setup.shared_input_operation_set_groups = dict(map(list2set, shared_input_operation_set_groups_state.items()))
        return setup


class MultiConfigQuantizerSetup(QuantizerSetupBase):
    def __init__(self) -> None:
        super().__init__()
        self.quantization_points: dict[QuantizationPointId, MultiConfigQuantizationPoint] = {}  # type: ignore
        self._unified_scale_qpid_vs_type: dict[QuantizationPointId, UnifiedScaleType] = {}

    def register_unified_scale_group_with_types(
        self, qp_group: list[QuantizationPointId], us_types: list[UnifiedScaleType]
    ) -> int:
        assert len(qp_group) == len(us_types)
        gid = super().register_unified_scale_group(qp_group)
        for qp_id, us_type in zip(qp_group, us_types):
            self._unified_scale_qpid_vs_type[qp_id] = us_type
        return gid

    def select_qconfigs(
        self, qp_id_vs_selected_qconfig_dict: dict[QuantizationPointId, QuantizerConfig], strict: bool = True
    ) -> SingleConfigQuantizerSetup:
        retval = SingleConfigQuantizerSetup()
        retval.unified_scale_groups = deepcopy(self.unified_scale_groups)
        retval.shared_input_operation_set_groups = deepcopy(self.shared_input_operation_set_groups)

        if Counter(qp_id_vs_selected_qconfig_dict.keys()) != Counter(self.quantization_points.keys()):
            msg = (
                "The set of quantization points for a selection is inconsistent with quantization"
                "points in the quantizer setup!"
            )
            raise ValueError(msg)
        for qp_id, qp in self.quantization_points.items():
            if strict:
                retval.quantization_points[qp_id] = qp.select_qconfig(qp_id_vs_selected_qconfig_dict[qp_id])
            else:
                multi_qp = qp
                qconfig = qp_id_vs_selected_qconfig_dict[qp_id]
                retval.quantization_points[qp_id] = SingleConfigQuantizationPoint(
                    multi_qp.insertion_point, qconfig, multi_qp.directly_quantized_operator_node_names
                )

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
                    nncf_logger.debug(
                        "Per-channel quantizer config selected in a MultiConfigQuantizerSetup for a "
                        "unified scale point that only supports per-tensor scale unification, disabling "
                        "unified scales for this point."
                    )
                retval.remove_unified_scale_from_point(per_channel_qid)

        return retval

    def select_first_qconfig_for_each_point(self) -> SingleConfigQuantizerSetup:
        qp_id_vs_qconfig_dict: dict[QuantizationPointId, QuantizerConfig] = {}
        for qp_id, qp in self.quantization_points.items():
            qp_id_vs_qconfig_dict[qp_id] = qp.possible_qconfigs[0]
        return self.select_qconfigs(qp_id_vs_qconfig_dict)

    @classmethod
    def from_single_config_setup(cls, single_conf_setup: SingleConfigQuantizerSetup) -> "MultiConfigQuantizerSetup":
        retval = cls()
        for qp_id, qp in single_conf_setup.quantization_points.items():
            multi_pt = MultiConfigQuantizationPoint(
                qip=qp.insertion_point,
                possible_qconfigs=[deepcopy(qp.qconfig)],
                directly_quantized_operator_node_names=qp.directly_quantized_operator_node_names,
            )
            retval.quantization_points[qp_id] = multi_pt
        for qp_set in single_conf_setup.unified_scale_groups.values():
            qp_list = list(qp_set)
            qp_types = [UnifiedScaleType.UNIFY_ALWAYS for _ in qp_list]
            retval.register_unified_scale_group_with_types(qp_list, qp_types)
        for qp_set in single_conf_setup.shared_input_operation_set_groups.values():
            qp_list = list(qp_set)
            retval.register_shared_inputs_group(qp_list)
        return retval
