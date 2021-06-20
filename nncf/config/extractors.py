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
from typing import Dict, List, Optional

from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.utils.logger import logger
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs


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
    target_device = config.get('target_device', 'ANY')
    if target_device != 'TRIAL':
        hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])
    global_compression_lr_multiplier = config.get('compression_lr_multiplier', None)
    global_ignored_scopes = config.get("ignored_scopes")
    global_target_scopes = config.get("target_scopes")

    if isinstance(compression_config_json_section, dict):
        compression_config_json_section = [compression_config_json_section]

    compression_algorithm_configs = []
    for algo_config in compression_config_json_section:
        algo_config = NNCFConfig(algo_config)
        algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
        if hw_config_type is not None:
            algo_config["hw_config_type"] = hw_config_type
        if global_ignored_scopes is not None:
            if "ignored_scopes" in algo_config:
                algo_config["ignored_scopes"].extend(global_ignored_scopes)
            else:
                algo_config["ignored_scopes"] = global_ignored_scopes
        if global_target_scopes is not None:
            if "target_scopes" in algo_config:
                algo_config["target_scopes"].extend(global_target_scopes)
            else:
                algo_config["target_scopes"] = global_target_scopes
        if 'compression_lr_multiplier' not in algo_config:
            algo_config['compression_lr_multiplier'] = global_compression_lr_multiplier
        algo_config['target_device'] = target_device
        compression_algorithm_configs.append(algo_config)

    return compression_algorithm_configs


def extract_range_init_params(config: NNCFConfig) -> Optional[Dict[str, object]]:
    """
    Extracts parameters of the quantization range initialization algorithm from the
    compression algorithm NNCFconfig.

    :param config: An instance of the NNCFConfig.
    :return: Parameters of the quantization range initialization algorithm.
    """
    init_range_config_dict_or_list = config.get('initializer', {}).get('range', {})

    range_init_args = None
    try:
        range_init_args = config.get_extra_struct(QuantizationRangeInitArgs)
    except KeyError as e:
        if not init_range_config_dict_or_list:
            logger.warning('Initializer section not specified for quantization algorithm in NNCF config and '
                           'quantization init args not supplied - the necessary parameters are not specified '
                           'to run the quantizer range initialization algorithm')
            return None

    if not init_range_config_dict_or_list:
        logger.warning('Enabling quantization range initialization with default parameters.')
        init_range_config_dict_or_list = {'num_init_samples': 256}

    max_num_init_samples = 0
    global_range_init_config = None
    scope_overrides = []  # type: List[PerLayerRangeInitConfig]
    if isinstance(init_range_config_dict_or_list, dict):
        global_range_init_config = RangeInitConfig.from_dict(init_range_config_dict_or_list)
        max_num_init_samples = global_range_init_config.num_init_samples
    else:
        for sub_init_range_config_dict in init_range_config_dict_or_list:
            scope_overrides.append(PerLayerRangeInitConfig.from_dict(sub_init_range_config_dict))
            max_num_init_samples_config = max(scope_overrides, key=lambda x: x.num_init_samples)
            max_num_init_samples = max_num_init_samples_config.num_init_samples

    if max_num_init_samples == 0:
        return None
    if range_init_args is None:
        raise ValueError(
            'Should run range initialization as specified via config,'
            'but the initializing data loader is not provided as an extra struct. '
            'Refer to `NNCFConfig.register_extra_structs` and the `QuantizationRangeInitArgs` class') from e

    params = {
        'init_range_data_loader': range_init_args.data_loader,
        'device': range_init_args.device,
        'global_init_config': global_range_init_config,
        'per_layer_range_init_configs': scope_overrides
    }

    return params


def extract_bn_adaptation_init_params(config: NNCFConfig) -> Dict[str, object]:
    """
    Extracts parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`
    from the compression algorithm NNCFconfig.

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
