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

from collections import OrderedDict
from enum import Enum
from typing import Type, List, Dict, Set, Optional

import addict as ad
import jstyleson as json
import warnings
import os

from nncf.config import product_dict
from nncf.definitions import NNCF_PACKAGE_ROOT_DIR, HW_CONFIG_RELATIVE_DIR
from nncf.dynamic_graph.operator_metatypes import OPERATOR_METATYPES
from nncf.hw_config_op_names import HWConfigOpName
from nncf.quantization.layers import QuantizerConfig, QuantizationMode, SymmetricQuantizer, AsymmetricQuantizer
from nncf.nncf_logger import logger as nncf_logger

class HWConfigType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    VPU = "vpu"
    DLA = "dla"

    @staticmethod
    def from_str(config_value: str) -> 'HWConfigType':
        if config_value == HWConfigType.CPU.value:
            return HWConfigType.CPU
        if config_value == HWConfigType.GPU.value:
            return HWConfigType.GPU
        if config_value == HWConfigType.VPU.value:
            return HWConfigType.VPU
        if config_value == HWConfigType.DLA.value:
            return HWConfigType.DLA
        raise RuntimeError("Unknown HW config type string")


def get_metatypes_by_hw_config_name(hw_config_name: HWConfigOpName) -> List['OperatorMetatype']:
    retval = []
    for op_meta in OPERATOR_METATYPES.registry_dict.values():  # type: OperatorMetatype
        if hw_config_name in op_meta.hw_config_names:
            retval.append(op_meta)
    return retval


class HWConfig(List):
    QUANTIZATION_ALGORITHM_NAME = "quantization"
    ATTRIBUTES_NAME = "attributes"
    SCALE_ATTRIBUTE_NAME = "scales"
    UNIFIED_TYPE_NAME = "unified"

    TYPE_TO_CONF_NAME_DICT = {
        HWConfigType.CPU: "cpu",
        HWConfigType.VPU: "vpu",
        HWConfigType.GPU: "gpu",
        HWConfigType.DLA: "dla"
    }

    def __init__(self):
        super().__init__()
        self.registered_algorithm_configs = {}
        self.target_device = None
        self.mergeable_operators_string = None

    @staticmethod
    def get_path_to_hw_config(hw_config_type: HWConfigType, hw_config_subtype: str = None):
        if hw_config_subtype is None:
            return '/'.join([NNCF_PACKAGE_ROOT_DIR, HW_CONFIG_RELATIVE_DIR,
                             HWConfig.TYPE_TO_CONF_NAME_DICT[hw_config_type]+".json"])
        return '/'.join([NNCF_PACKAGE_ROOT_DIR, HW_CONFIG_RELATIVE_DIR,
                         HWConfig.TYPE_TO_CONF_NAME_DICT[hw_config_type] + "_" + hw_config_subtype + ".json"])

    @classmethod
    def from_dict(cls, dct: dict):
        # pylint:disable=too-many-nested-blocks,too-many-branches
        hw_config = cls()
        hw_config.target_device = dct['target_device']

        for algorithm_name, algorithm_configs in dct.get('config', {}).items():
            hw_config.registered_algorithm_configs[algorithm_name] = {}
            for algo_config_alias, algo_config in algorithm_configs.items():
                for key, val in algo_config.items():
                    if not isinstance(val, list):
                        algo_config[key] = [val]

                hw_config.registered_algorithm_configs[algorithm_name][algo_config_alias] = list(
                    product_dict(algo_config))

        for op_dict in dct.get('operations', []):
            for algorithm_name in op_dict:
                if algorithm_name not in hw_config.registered_algorithm_configs:
                    continue
                tmp_config = {}
                for algo_and_op_specific_field_name, algorithm_configs in op_dict[algorithm_name].items():
                    if not isinstance(algorithm_configs, list):
                        algorithm_configs = [algorithm_configs]

                    tmp_config[algo_and_op_specific_field_name] = []
                    for algorithm_config in algorithm_configs:
                        if isinstance(algorithm_config, str):  # Alias was supplied
                            tmp_config[algo_and_op_specific_field_name].extend(
                                hw_config.registered_algorithm_configs[algorithm_name][algorithm_config])
                        else:
                            for key, val in algorithm_config.items():
                                if not isinstance(val, list):
                                    algorithm_config[key] = [val]

                            tmp_config[algo_and_op_specific_field_name].extend(list(product_dict(algorithm_config)))

                op_dict[algorithm_name] = tmp_config

            hw_config.append(ad.Dict(op_dict))

        return hw_config

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            json_config = json.load(f, object_pairs_hook=OrderedDict)
            hw_config = HWConfig.from_dict(json_config)
            mergeable_subgraph_path = os.path.splitext(path)[0] + "_subgraphs.txt"

            try:
                with open(mergeable_subgraph_path) as subgraph_file:
                    hw_config.mergeable_operators_string = " ".join(line.strip() for line in subgraph_file)
                    nncf_logger.info('Overriding default subgraph definitions since {} was found '\
                        .format(mergeable_subgraph_path))

            except IOError:
                nncf_logger.info('Using default subgraph definitions since {} not found '\
                    .format(mergeable_subgraph_path))

            return hw_config

    @staticmethod
    def get_quantization_mode_from_config_value(str_val: str):
        if str_val == "symmetric":
            return QuantizationMode.SYMMETRIC
        if str_val == "asymmetric":
            return QuantizationMode.ASYMMETRIC
        if str_val == "blockfp":
            return QuantizationMode.BLOCKFP
        raise RuntimeError("Invalid quantization type specified in HW config")

    @staticmethod
    def get_is_per_channel_from_config_value(str_val: str):
        if str_val == "perchannel":
            return True
        if str_val == "pertensor":
            return False
        raise RuntimeError("Invalid quantization granularity specified in HW config")

    @staticmethod
    def get_qconf_from_hw_config_subdict(quantization_subdict: Dict, for_weights=False):
        bits = quantization_subdict["bits"]
        mode = HWConfig.get_quantization_mode_from_config_value(quantization_subdict["mode"])
        if  mode == QuantizationMode.BLOCKFP:
            bits = quantization_subdict["bits"]
            exponent_bits = quantization_subdict["exponent_bits"]
            block_size = quantization_subdict["block_size"]
            folded = quantization_subdict.get("folded", False)

            return QuantizerConfig.create(mode=mode,
                                          exponent_bits=exponent_bits,
                                          bits=bits,
                                          block_size=block_size,
                                          folded=folded
                                         )


        is_per_channel = HWConfig.get_is_per_channel_from_config_value(quantization_subdict["granularity"])
        signedness_to_force = None
        if 'level_low' in quantization_subdict and 'level_high' in quantization_subdict:
            signedness_to_force = False
            if mode == QuantizationMode.SYMMETRIC:
                if quantization_subdict['level_low'] < 0 < quantization_subdict['level_high']:
                    signedness_to_force = True
                true_level_high, true_level_low, _ = \
                SymmetricQuantizer.calculate_level_ranges(bits, True, for_weights)
            else:
                signedness_to_force = True
                true_level_high, true_level_low, _ = AsymmetricQuantizer.calculate_level_ranges(bits)

            assert quantization_subdict['level_low'] == true_level_low, \
                    "Invalid value of quantizer parameter `level_low`.\
                        The parameter must be consistent with other parameters!"
            assert quantization_subdict['level_high'] == true_level_high, \
                    "Invalid value of quantizer parameter `level_high`.\
                        The parameter must be consistent with other parameters!"

        return QuantizerConfig.create(bits=bits,
                                      mode=mode,
                                      per_channel=is_per_channel,
                                      signedness_to_force=signedness_to_force,
                                      is_weights=for_weights)

    @staticmethod
    def is_qconf_list_corresponding_to_unspecified_op(qconf_list: Optional[List[QuantizerConfig]]):
        return qconf_list is None

    @staticmethod
    def is_wildcard_quantization(qconf_list: Optional[List[QuantizerConfig]]):
        # Corresponds to an op itself being specified in the HW config, but having no associated quantization
        # configs specified
        return qconf_list is not None and len(qconf_list) == 0

    def get_metatype_vs_quantizer_configs_map(self, for_weights=False) -> Dict[Type['OperatorMetatype'],
                                                                               Optional[List[QuantizerConfig]]]:
        # 'None' for ops unspecified in HW config, empty list for wildcard quantization ops
        retval = {k: None for k in OPERATOR_METATYPES.registry_dict.values()}
        config_key = "weights" if for_weights else "activations"
        for op_dict in self:
            hw_config_op_name = op_dict.type  # type: HWConfigOpName

            metatypes = get_metatypes_by_hw_config_name(hw_config_op_name)
            if not metatypes:
                warnings.warn("Operation name {} in HW config is not registered in NNCF under any supported operation "
                              "metatype - will be ignored".format(hw_config_op_name))

            if self.QUANTIZATION_ALGORITHM_NAME in op_dict:
                allowed_qconfs = op_dict[self.QUANTIZATION_ALGORITHM_NAME][config_key]
            else:
                allowed_qconfs = []

            qconf_list_with_possible_duplicates = []
            for hw_config_qconf_dict in allowed_qconfs:
                qconf_list_with_possible_duplicates.append(
                    self.get_qconf_from_hw_config_subdict(hw_config_qconf_dict, for_weights))

            qconf_list = list(OrderedDict.fromkeys(qconf_list_with_possible_duplicates))

            for meta in metatypes:
                retval[meta] = qconf_list

        return retval

    def get_operations_with_unified_scales(self) -> Set[Type['OperatorMetatype']]:
        retval = set()
        for op_dict in self:
            if self.ATTRIBUTES_NAME in op_dict:
                if self.SCALE_ATTRIBUTE_NAME in op_dict[self.ATTRIBUTES_NAME]:
                    if op_dict[self.ATTRIBUTES_NAME][self.SCALE_ATTRIBUTE_NAME] == self.UNIFIED_TYPE_NAME:
                        hw_config_op_name = op_dict.type  # type: HWConfigOpName
                        metatypes = get_metatypes_by_hw_config_name(hw_config_op_name)
                        if not metatypes:
                            warnings.warn(
                                "Operation name {} in HW config is not registered in NNCF under any supported "
                                "operation metatype - will be ignored".format(hw_config_op_name))
                        retval.update(metatypes)
        return retval
