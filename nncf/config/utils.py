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
        'device': args.device,
    }

    return params
