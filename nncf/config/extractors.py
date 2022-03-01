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

from typing import Dict
from typing import List
from typing import Optional

from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.utils.logger import logger
from nncf.config.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs


def extract_algorithm_names(config: NNCFConfig) -> List[str]:
    retval = []
    compression_config_json_section = config.get('compression', [])
    if isinstance(compression_config_json_section, dict):
        compression_config_json_section = [compression_config_json_section]
    for algo_config in compression_config_json_section:
        retval.append(algo_config['algorithm'])
    return retval


def extract_algo_specific_config(config: NNCFConfig, algo_name_to_match: str) -> Dict:
    """
    Extracts a .json sub-dictionary for a given compression algorithm from the
    common NNCFConfig.

    :param config: An instance of the NNCFConfig.
    :param algo_name_to_match: The name of the algorithm for which the algorithm-specific section
      should be extracted.
    :return: The sub-dictionary, exactly as it is specified in the NNCF configuration of the .json file,
    that corresponds to the algorithm-specific data (i.e. {"algorithm": "quantization", ... })
    """
    compression_section = config.get('compression', [])
    if isinstance(compression_section, list):
        algo_list = compression_section
    else:
        assert isinstance(compression_section, dict)
        algo_list = [compression_section]

    from nncf.common.compression import NO_COMPRESSION_ALGORITHM_NAME
    if algo_name_to_match == NO_COMPRESSION_ALGORITHM_NAME:
        if len(algo_list) > 0:
            raise RuntimeError(f'No algorithm configuration should be specified '
                               f'when you try to extract {algo_name_to_match} from the NNCF config!')
        return {}

    matches = []
    for compression_algo_dict in algo_list:
        algo_name = compression_algo_dict['algorithm']
        if algo_name == algo_name_to_match:
            matches.append(compression_algo_dict)

    if len(matches) > 1:
        raise RuntimeError(f'Multiple algorithm configurations specified for the same '
                           f'algo {algo_name_to_match} in the NNCF config!')
    if not matches:
        raise RuntimeError(f'Did not find an algorithm configuration for '
                           f'algo {algo_name_to_match} in the NNCF config!')
    return next(iter(matches))


def extract_range_init_params(config: NNCFConfig, algorithm_name: str = 'quantization') -> Optional[Dict[str, object]]:
    """
    Extracts parameters of the quantization range initialization algorithm from the
    compression algorithm NNCFconfig.

    :param config: An instance of the NNCFConfig.
    :param algorithm_name: Name of the compression algorithm. Should be
        one of the following: `quantization`, `experimental_quantization`.
    :return: Parameters of the quantization range initialization algorithm.
    """
    algo_config = extract_algo_specific_config(config, algorithm_name)
    init_range_config_dict_or_list = algo_config.get('initializer', {}).get('range', {})

    range_init_args = None
    try:
        range_init_args = config.get_extra_struct(QuantizationRangeInitArgs)
    except KeyError:
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
            'Refer to `NNCFConfig.register_extra_structs` and the `QuantizationRangeInitArgs` class')

    params = {
        'init_range_data_loader': range_init_args.data_loader,
        'device': range_init_args.device,
        'global_init_config': global_range_init_config,
        'per_layer_range_init_configs': scope_overrides
    }

    return params


def extract_bn_adaptation_init_params(config: NNCFConfig, algo_name: str) -> Optional[Dict[str, object]]:
    """
    Extracts parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`
    from the compression algorithm NNCFconfig.

    :param config: An instance of the NNCFConfig.
    :param algo_name: The name of the algorithm for which the params have to be extracted.
    :return: Parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm` specific
      to the supplied algorithm, or None if the config specified not to perform any batchnorm adaptation.
    """
    algo_config = extract_algo_specific_config(config, algo_name)
    params = algo_config.get('initializer', {}).get('batchnorm_adaptation', {})
    num_bn_adaptation_samples = params.get('num_bn_adaptation_samples', 2000)

    if num_bn_adaptation_samples == 0:
        return None

    try:
        args = config.get_extra_struct(BNAdaptationInitArgs)
    except KeyError:
        raise RuntimeError(
            'Unable to create the batch-norm statistics adaptation algorithm '
            'because the data loader is not provided as an extra struct. Refer to the '
            '`NNCFConfig.register_extra_structs` method and the `BNAdaptationInitArgs` class.') from None

    params = {
        'num_bn_adaptation_samples': num_bn_adaptation_samples,
        'data_loader': args.data_loader,
        'device': args.device
    }

    return params


def extract_accuracy_aware_training_params(config: NNCFConfig) -> Dict[str, object]:
    """
    Extracts accuracy aware training parameters from NNCFConfig.

    :param: config: An instance of the NNCFConfig.
    :return: Accuracy aware training parameters.
    """
    class NNCFAlgorithmNames:
        QUANTIZATION = 'quantization'
        FILTER_PRUNING = 'filter_pruning'
        SPARSITY = ['rb_sparsity', 'magnitude_sparsity', 'const_sparsity']

    def validate_accuracy_aware_schema(config: NNCFConfig, params: Dict[str, object]):
        from nncf.common.accuracy_aware_training import AccuracyAwareTrainingMode
        if params["mode"] == AccuracyAwareTrainingMode.EARLY_EXIT:
            return
        if params["mode"] == AccuracyAwareTrainingMode.ADAPTIVE_COMPRESSION_LEVEL:
            algorithms = extract_algorithm_names(config)
            if NNCFAlgorithmNames.FILTER_PRUNING in algorithms and \
                    any(algo in NNCFAlgorithmNames.SPARSITY for algo in algorithms):
                raise RuntimeError("adaptive_compression_level mode supports filter_pruning or sparsity algorithms"
                                   "separately. Please, choose only one algorithm with adaptive compression level. "
                                   "Take a note that you still can use it combined with quantization.")
            if len(algorithms) == 1 and algorithms[0] == NNCFAlgorithmNames.QUANTIZATION:
                raise RuntimeError("adaptive_compression_level mode doesn't support quantization")

    accuracy_aware_training_config = config.get("accuracy_aware_training", None)

    mode = accuracy_aware_training_config.get("mode")
    params = {"mode": mode}

    if accuracy_aware_training_config.get("params") is not None:
        for param_key, param_val in accuracy_aware_training_config.get("params").items():
            params[param_key] = param_val

    validate_accuracy_aware_schema(config, params)

    return params
