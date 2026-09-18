# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict
from typing import Any

import openvino as ov

import nncf
from nncf.common.logging import nncf_logger
from nncf.quantization.advanced_parameters import CustomAnnotation
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.scopes import BaseScope


def exclude_empty_fields(value: dict[str, Any]) -> dict[str, Any]:
    """
    Remove keys where value is empty and key is `validate`

    :param value: IgnoredScope instance, based on the quantization parameters
    """
    value.pop("validate")
    keys = list(value.keys())
    for key in keys:
        if not value[key]:
            value.pop(key)
    return value


def compression_config_to_dict(config: WeightCompressionConfig) -> dict[str, Any]:
    """
    Converts a weight compression config into a dictionary suitable for dumping into Model's meta section.

    Codebook values are reduced to their number since the values themselves are too large to be dumped.

    :param config: Weight compression config.
    :return: Dictionary with the compression config parameters.
    """
    value: dict[str, Any] = {"mode": config.mode.value, "group_size": config.group_size}
    if config.codebook_values is not None:
        value["codebook_size"] = config.codebook_values.size
    return value


def dump_parameters(
    model: ov.Model, parameters: dict[str, Any], algo_name: str | None = "quantization", path: list[str] | None = None
) -> None:
    """
    Dumps the given parameters into Model's meta section.

    :param model: ov.Model instance.
    :param algo_name: Name of the algorithm, to which the parameters refer.
    :param parameters: Incoming dictionary with parameters to save.
    :param path: Optional list of the paths.
    """
    try:
        path = path if path else []
        for key, value in parameters.items():
            # Special condition for composed fields like IgnoredScope
            if isinstance(value, BaseScope):
                value = exclude_empty_fields(asdict(value))
                if bool(value):
                    dump_parameters(model, value, algo_name, path + [key])
                    continue
                # The default value in case empty ignored_scope parameter passed
                value = []
            # Special condition for the list of custom annotations
            elif isinstance(value, (list, tuple)) and any(isinstance(item, CustomAnnotation) for item in value):
                for i, annotation in enumerate(value):
                    dump_parameters(
                        model,
                        {"scope": annotation.scope, "config": compression_config_to_dict(annotation.config)},
                        algo_name,
                        path + [key, str(i)],
                    )
                continue

            rt_path = ["nncf", algo_name] + path + [key]
            model.set_rt_info(str(value), rt_path)
        model.set_rt_info(nncf.__version__, ["nncf", "version"])
    except RuntimeError as e:
        nncf_logger.debug(f"Unable to dump optimization parameters due to error: {e}")
