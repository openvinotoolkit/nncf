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

from typing import List, Dict, Optional

from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.quantization.structs import QuantizerGroup


class RangeInitConfig:
    """
    The `RangeInitConfig` class representing the quantization range initialization
    parameters.
    """

    def __init__(self, init_type: str, num_init_samples: int, init_type_specific_params: Dict = None):
        """
        Initializes the quantization range initialization parameters.

        :param init_type: Type of the initializer - determines which tensor
            statistics will be used to initialize quantization ranges.
        :param num_init_samples: The number of samples from the dataset to consume
            as sample model inputs to compute quantization ranges.
        :param init_type_specific_params: additional parameters specific to the type
            of the initializer
        """
        self.init_type = init_type
        self.num_init_samples = num_init_samples
        self.init_type_specific_params = init_type_specific_params
        if self.init_type_specific_params is None:
            self.init_type_specific_params = {}

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_dict(cls, dct: Dict) -> 'RangeInitConfig':
        num_init_samples = dct.get('num_init_samples', 256)
        if num_init_samples < 0:
            raise ValueError('Number of initialization samples must be >= 0')
        return cls(dct.get('type', 'mean_min_max'),
                   num_init_samples,
                   dct.get('params'))


class PerLayerRangeInitConfig(RangeInitConfig):
    """
    The `PerLayerRangeInitConfig` class representing the quantization range
    initialization parameters for layers which are specified using the target
    and ignored scopes and the target group of quantizers.
    """

    def __init__(self, range_init_config: RangeInitConfig,
                 target_scopes: Optional[List[str]],
                 ignored_scopes: Optional[List[str]],
                 target_quantizer_group: QuantizerGroup = None):
        """
        Initializes the quantization range initialization parameters.

        :param range_init_config: The quantization range initialization parameters.
        :param target_scopes: A list of model control flow graph node scopes
            to be considered for this operation - functions as a 'denylist'
        :param ignored_scopes: A list of model control flow graph node scopes
            to be ignored for this operation - functions as a 'allowlist'
        :param target_quantizer_group: The target group of quantizers for which
            specified type of range initialization will be applied. It can be
            quantizers group for activations or weights.
        """

        super().__init__(range_init_config.init_type, range_init_config.num_init_samples,
                         range_init_config.init_type_specific_params)
        if target_scopes is None and ignored_scopes is None:
            raise ValueError('At least one of the (target_scopes, ignored_scopes) should be specified'
                             ' for a per-layer range init config!')
        self.target_scopes = target_scopes
        self.ignored_scopes = ignored_scopes
        self.target_group = target_quantizer_group

    @classmethod
    def from_dict(cls, dct: Dict) -> 'PerLayerRangeInitConfig':
        base_config = RangeInitConfig.from_dict(dct)

        def get_list(dct: Dict, attr_name: str) -> Optional[List[str]]:
            str_or_list = dct.get(attr_name)
            if str_or_list is None:
                return None
            if isinstance(str_or_list, str):
                retval_list = [str_or_list]
            else:
                retval_list = str_or_list
            return retval_list
        target_scopes, ignored_scopes = get_list(dct, 'target_scopes'), get_list(dct, 'ignored_scopes')

        target_group_str = dct.get('target_quantizer_group')
        target_group = None
        if target_group_str is not None:
            target_group = QuantizerGroup.from_str(target_group_str)

        return cls(base_config, target_scopes, ignored_scopes, target_group)


class RangeInitParams:
    """
    The `RangeInitParams` class representing the initialization dataset and the
    quantization range initialization parameters for all model layers.
    """

    def __init__(self, init_range_data_loader: NNCFDataLoader,
                 device: str,
                 global_init_config: Optional[RangeInitConfig],
                 per_layer_range_init_configs: List[PerLayerRangeInitConfig]):
        """

        :param init_range_data_loader: Provides an iterable over the given dataset.
        :param device: Device to perform initialization. If `device` is `None`
            then the device of the model parameters will be used.
        :param global_init_config: The quantization range initialization parameters
            for a model
        :param per_layer_range_init_configs: The List of the quantization range
            initialization parameters for model layers
        """
        self.init_range_data_loader = init_range_data_loader
        self.device = device
        self.global_init_config = global_init_config
        self.per_layer_range_init_configs = per_layer_range_init_configs
