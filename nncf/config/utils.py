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

from copy import deepcopy
from typing import Dict, List

from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs


def extract_compression_algorithm_configs(config: NNCFConfig) -> List[NNCFConfig]:
    """
    Extracts specific algorithm parameters for each compression algorithm from the
    common NNCFConfig.

    :param config: An instance of the NNCFConfig.
    :return: List of the NNCFConfigs, each of which contains specific algorithm
        parameters related only to this algorithm
    """
    compression_config_json_section = config.get('compression', {})
    compression_config_json_section = deepcopy(compression_config_json_section)

    hw_config_type = None
    target_device = config.get("target_device", "ANY")
    global_compression_lr_multiplier = config.get("compression_lr_multiplier", None)
    if target_device != 'TRIAL':
        hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

    if isinstance(compression_config_json_section, dict):
        compression_config_json_section = [compression_config_json_section]

    compression_algorithm_configs = []
    for algo_config in compression_config_json_section:
        algo_config = NNCFConfig(algo_config)
        algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
        algo_config["hw_config_type"] = hw_config_type
        if "compression_lr_multiplier" not in algo_config:
            algo_config["compression_lr_multiplier"] = global_compression_lr_multiplier
        compression_algorithm_configs.append(algo_config)

    return compression_algorithm_configs


def extract_bn_adaptation_init_params(config: NNCFConfig) -> Dict[str, object]:
    """
    Extracts parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`
    from the NNCF config.

    :param config: An instance of the NNCFConfig.
    :return: Parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`.
    """
    params = config.get('initializer', {}).get('batchnorm_adaptation', {})
    num_bn_adaptation_samples = params.get('num_bn_adaptation_samples', 2000)
    num_bn_forget_samples = params.get('num_bn_forget_samples', 1000)

    try:
        args = config.get_extra_struct(BNAdaptationInitArgs)
    except KeyError:
        raise RuntimeError(
            'There is no possibility to create the batch-norm statistics adaptation algorithm '
            'because the data loader is not provided as an extra struct. Refer to the '
            '`NNCFConfig.register_extra_structs` method and the `BNAdaptationInitArgs` class.') from None

    params = {
        'num_bn_adaptation_samples': num_bn_adaptation_samples,
        'num_bn_forget_samples': num_bn_forget_samples,
        'data_loader': args.data_loader,
        'device': args.device
    }

    return params
