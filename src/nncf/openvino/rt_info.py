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
from nncf.quantization.advanced_parameters import ScopedConfig
from nncf.scopes import Scope


def exclude_empty_fields(value: dict[str, Any]) -> dict[str, Any]:
    """
    Remove keys where value is empty and key is `validate`

    :param value: Scope instance, based on the quantization parameters
    """
    value.pop("validate")
    keys = list(value.keys())
    for key in keys:
        if not value[key]:
            value.pop(key)
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
        path = path or []
        for key, value in parameters.items():
            # Special condition for composed fields like Scope
            if isinstance(value, Scope):
                value = exclude_empty_fields(asdict(value))
                if bool(value):
                    dump_parameters(model, value, algo_name, path + [key])
                    continue
                # The default value in case empty ignored_scope parameter passed
                value = []
            # Special condition for the list of scoped configs
            elif isinstance(value, (list, tuple)) and any(isinstance(item, ScopedConfig) for item in value):
                for i, scoped_config in enumerate(value):
                    # Index each scoped config so scoped configs do not overwrite each other,
                    # e.g. scoped_configs/0/{scope/patterns,compression_config}.
                    dump_parameters(
                        model,
                        {
                            "scope": scoped_config.scope,
                            "compression_config": scoped_config.compression_config,
                        },
                        algo_name,
                        path + [key, str(i)],
                    )
                continue

            rt_path = ["nncf", algo_name] + path + [key]
            model.set_rt_info(str(value), rt_path)
        model.set_rt_info(nncf.__version__, ["nncf", "version"])
    except RuntimeError as e:
        nncf_logger.debug(f"Unable to dump optimization parameters due to error: {e}")
