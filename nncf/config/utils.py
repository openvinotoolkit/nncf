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

from typing import Dict

from nncf.config import NNCFConfig
from nncf.config.structure import BNAdaptationInitArgs


def extract_bn_adaptation_init_params(config: NNCFConfig) -> Dict[str, object]:
    """
    Extracts parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`
    from the NNCF config.

    :param config: NNCF config.
    :return: Parameters for initialization of an object of the class `BatchnormAdaptationAlgorithm`.
    """

    params = config.get('initializer', {}).get('batchnorm_adaptation', {})

    try:
        extra_config = config.get_extra_struct(BNAdaptationInitArgs)
    except KeyError:
        extra_config = None

    params['extra_config'] = extra_config
    return params
