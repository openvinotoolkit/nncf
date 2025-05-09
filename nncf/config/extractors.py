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

from typing import Any, Optional

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.quantization.initialization.range import PerLayerRangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.config.config import NNCFConfig
from nncf.config.schemata.defaults import NUM_BN_ADAPTATION_SAMPLES
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs


def extract_algorithm_names(config: NNCFConfig) -> list[str]:
    retval = []
    compression_config_json_section = config.get("compression", [])
    if isinstance(compression_config_json_section, dict):
        compression_config_json_section = [compression_config_json_section]
    for algo_config in compression_config_json_section:
        retval.append(algo_config["algorithm"])
    return retval


def extract_algo_specific_config(config: NNCFConfig, algo_name_to_match: str) -> dict[str, Any]:
    """
    Extracts a .json sub-dictionary for a given compression algorithm from the
    common NNCFConfig.

    :param config: An instance of the NNCFConfig.
    :param algo_name_to_match: The name of the algorithm for which the algorithm-specific section
      should be extracted.
    :return: The sub-dictionary, exactly as it is specified in the NNCF configuration of the .json file,
    that corresponds to the algorithm-specific data (i.e. {"algorithm": "quantization", ... })
    """
    compression_section = config.get("compression", [])
    if isinstance(compression_section, list):
        algo_list = compression_section
    else:
        assert isinstance(compression_section, dict)
        algo_list = [compression_section]

    from nncf.common.compression import NO_COMPRESSION_ALGORITHM_NAME

    if algo_name_to_match == NO_COMPRESSION_ALGORITHM_NAME:
        if len(algo_list) > 0:
            msg = (
                f"No algorithm configuration should be specified "
                f"when you try to extract {algo_name_to_match} from the NNCF config!"
            )
            raise nncf.ValidationError(msg)
        return {}

    matches = []
    for compression_algo_dict in algo_list:
        algo_name = compression_algo_dict["algorithm"]
        if algo_name == algo_name_to_match:
            matches.append(compression_algo_dict)

    if len(matches) > 1:
        msg = f"Multiple algorithm configurations specified for the same algo {algo_name_to_match} in the NNCF config!"
        raise nncf.ValidationError(msg)
    if not matches:
        msg = f"Did not find an algorithm configuration for algo {algo_name_to_match} in the NNCF config!"
        raise nncf.InternalError(msg)
    return next(iter(matches))


def extract_range_init_params(config: NNCFConfig, algorithm_name: str = "quantization") -> Optional[dict[str, Any]]:
    """
    Extracts parameters of the quantization range initialization algorithm from the
    compression algorithm NNCFconfig.

    :param config: An instance of the NNCFConfig.
    :param algorithm_name: Name of the compression algorithm. Should be
        one of the following: `quantization`, `experimental_quantization`.
    :return: Parameters of the quantization range initialization algorithm.
    """
    algo_config = extract_algo_specific_config(config, algorithm_name)
    init_range_config_dict_or_list = algo_config.get("initializer", {}).get("range", {})

    try:
        range_init_args = config.get_extra_struct(QuantizationRangeInitArgs)
    except KeyError:
        if not init_range_config_dict_or_list:
            nncf_logger.warning(
                "Initializer section not specified for quantization algorithm in NNCF config and "
                "quantization init args not supplied - the quantizer range initialization algorithm "
                "cannot proceed."
            )
            return None

    if not init_range_config_dict_or_list:
        nncf_logger.warning("Enabling quantization range initialization with default parameters.")
        init_range_config_dict_or_list = {"num_init_samples": 256}

    max_num_init_samples = 0
    global_range_init_config = None
    scope_overrides: list[PerLayerRangeInitConfig] = []
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
    if not isinstance(range_init_args, QuantizationRangeInitArgs):
        msg = (
            "Should run range initialization as specified via config,"
            "but the initializing data loader is not provided as an extra struct. "
            "Refer to `NNCFConfig.register_extra_structs` and the `QuantizationRangeInitArgs` class"
        )
        raise ValueError(msg)

    params = {
        "init_range_data_loader": range_init_args.data_loader,
        "device": range_init_args.device,
        "global_init_config": global_range_init_config,
        "per_layer_range_init_configs": scope_overrides,
    }

    return params


def extract_bn_adaptation_init_params(config: NNCFConfig, algo_name: str) -> Optional[dict[str, object]]:
    """
    Extracts parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`
    from the compression algorithm NNCFconfig.

    :param config: An instance of the NNCFConfig.
    :param algo_name: The name of the algorithm for which the params have to be extracted.
    :return: Parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm` specific
      to the supplied algorithm, or None if the config specified not to perform any batchnorm adaptation.
    """
    algo_config = extract_algo_specific_config(config, algo_name)
    params = algo_config.get("initializer", {}).get("batchnorm_adaptation", {})
    return get_bn_adapt_algo_kwargs(config, params)


def has_bn_section(config: NNCFConfig, algo_name: str) -> bool:
    algo_config = extract_algo_specific_config(config, algo_name)
    return algo_config.get("initializer", {}).get("batchnorm_adaptation") is not None


class BNAdaptDataLoaderNotFoundError(RuntimeError):
    pass


def get_bn_adapt_algo_kwargs(nncf_config: NNCFConfig, params: dict[str, Any]) -> Optional[dict[str, Any]]:
    num_bn_adaptation_samples = params.get("num_bn_adaptation_samples", NUM_BN_ADAPTATION_SAMPLES)

    if num_bn_adaptation_samples == 0:
        return None

    try:
        args = nncf_config.get_extra_struct(BNAdaptationInitArgs)
    except KeyError:
        msg = (
            "Unable to create the batch-norm statistics adaptation algorithm "
            "because the data loader is not provided as an extra struct. Refer to the "
            "`NNCFConfig.register_extra_structs` method and the `BNAdaptationInitArgs` class."
        )
        raise BNAdaptDataLoaderNotFoundError(msg) from None

    if not isinstance(args, BNAdaptationInitArgs):
        msg = "The extra struct for batch-norm adaptation must be an instance of the BNAdaptationInitArgs class."
        raise BNAdaptDataLoaderNotFoundError(msg)
    params = {
        "num_bn_adaptation_samples": num_bn_adaptation_samples,
        "data_loader": args.data_loader,
        "device": args.device,
    }
    return params


def extract_accuracy_aware_training_params(config: NNCFConfig) -> dict[str, Any]:
    """
    Extracts accuracy aware training parameters from NNCFConfig.

    :param: config: An instance of the NNCFConfig.
    :return: Accuracy aware training parameters.
    """

    class NNCFAlgorithmNames:
        QUANTIZATION = "quantization"
        FILTER_PRUNING = "filter_pruning"
        SPARSITY = ["rb_sparsity", "magnitude_sparsity", "const_sparsity"]

    def validate_accuracy_aware_schema(config: NNCFConfig, params: dict[str, Any]) -> None:
        from nncf.common.accuracy_aware_training import AccuracyAwareTrainingMode

        if params["mode"] == AccuracyAwareTrainingMode.EARLY_EXIT:
            return
        if params["mode"] == AccuracyAwareTrainingMode.ADAPTIVE_COMPRESSION_LEVEL:
            algorithms = extract_algorithm_names(config)
            if NNCFAlgorithmNames.FILTER_PRUNING in algorithms and any(
                algo in NNCFAlgorithmNames.SPARSITY for algo in algorithms
            ):
                msg = (
                    "adaptive_compression_level mode supports filter_pruning or sparsity algorithms"
                    "separately. Please, choose only one algorithm with adaptive compression level. "
                    "Take a note that you still can use it combined with quantization."
                )
                raise nncf.ValidationError(msg)
            if len(algorithms) == 1 and algorithms[0] == NNCFAlgorithmNames.QUANTIZATION:
                msg = "adaptive_compression_level mode doesn't support quantization"
                raise nncf.ValidationError(msg)

    accuracy_aware_training_config = config.get("accuracy_aware_training", None)

    mode = accuracy_aware_training_config.get("mode")
    params = {"mode": mode}

    if accuracy_aware_training_config.get("params") is not None:
        for param_key, param_val in accuracy_aware_training_config.get("params").items():
            params[param_key] = param_val

    validate_accuracy_aware_schema(config, params)

    return params


def has_input_info_field(config: NNCFConfig) -> bool:
    return config.get("input_info") is not None
