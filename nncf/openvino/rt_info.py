# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import openvino.runtime as ov

from nncf.common.logging import nncf_logger
from nncf.scopes import IgnoredScope


def dump_parameters(
    model: ov.Model, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
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
            if isinstance(value, IgnoredScope):
                dump_parameters(model, value.__dict__, algo_name, [key])
                continue
            rt_path = ["nncf", algo_name] + path + [key]
            model.set_rt_info(str(value), rt_path)
    except RuntimeError as e:
        nncf_logger.debug(f"Unable to dump optimization parameters due to error: {e}")
