"""
 Copyright (c) 2019-2021 Intel Corporation
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
import warnings
from enum import Enum
from typing import Dict
from typing import List
from typing import Set

import re
from typing import Tuple

import torch

from nncf.common.utils.logger import logger as nncf_logger


def load_state(model: torch.nn.Module, state_dict_to_load: dict, is_resume: bool = False) -> int:
    """
    Used to load a checkpoint containing a compressed model into an NNCFNetwork object, but can
    be used for any PyTorch module as well. Will do matching of state_dict_to_load parameters to
    the model's state_dict parameters while discarding irrelevant prefixes added during wrapping
    in NNCFNetwork or DataParallel/DistributedDataParallel objects, and load the matched parameters
    from the state_dict_to_load into the model's state dict.
    :param model: The target module for the state_dict_to_load to be loaded to.
    :param state_dict_to_load: A state dict containing the parameters to be loaded into the model.
    :param is_resume: Determines the behavior when the function cannot do a successful parameter match
    when loading. If True, the function will raise an exception if it cannot match the state_dict_to_load
    parameters to the model's parameters (i.e. if some parameters required by model are missing in
    state_dict_to_load, or if state_dict_to_load has parameters that could not be matched to model parameters,
    or if the shape of parameters is not matching). If False, the exception won't be raised.
    Usually is_resume is specified as False when loading uncompressed model's weights into the model with
    compression algorithms already applied, and as True when loading a compressed model's weights into the model
    with compression algorithms applied to evaluate the model.
    :return: The number of state_dict_to_load entries successfully matched and loaded into model.
    """

    if 'state_dict' in state_dict_to_load:
        state_dict_to_load = state_dict_to_load['state_dict']
    model_state_dict = model.state_dict()

    key_matcher = KeyMatcher(is_resume, state_dict_to_load, model_state_dict)
    new_dict = key_matcher.run()
    num_loaded_params = len(new_dict)
    key_matcher.handle_problematic_keys()
    nncf_logger.info("Loaded {}/{} parameters".format(num_loaded_params, len(model_state_dict.items())))

    model.load_state_dict(new_dict, strict=False)
    return num_loaded_params


class OptionalParametersRegistry:
    """
    Provides an interface to register optional parameters and get access to all of them.
    If there is no optional parameters in a checkpoint, it can be loaded without an error in a strict mode.
    New parameters can be introduced for the model without breaking backward compatibility with old checkpoint.
    """

    def __init__(self):
        self._optional_parameters_names = set()

    def register(self, parameter_name: str):
        self._optional_parameters_names.add(parameter_name)

    def get_optional_parameters_names(self) -> Set[str]:
        return self._optional_parameters_names


OPTIONAL_PARAMETERS_REGISTRY = OptionalParametersRegistry()


class ProcessedKeyStatus(Enum):
    """ Status of matching checkpoint key with model keys """
    MATCHED = 'Matched'
    MISSING = 'Missing'
    UNEXPECTED = 'Unexpected'
    SIZE_MISMATCHED = 'Size mismatched'
    SKIPPED = 'Skipped'


class ProcessedKeys:
    """ Contains checkpoint keys with their status of matching with model keys """
    def __init__(self):
        self._keys = {}  # type: Dict[ProcessedKeyStatus, List[str]]
        for key_status in ProcessedKeyStatus:
            self._keys[key_status] = []

    def add_key(self, key: str, status: ProcessedKeyStatus):
        self._keys[status].append(key)

    def extend_keys(self, keys: List[str], status: ProcessedKeyStatus):
        self._keys[status].extend(keys)

    def add_skipped_and_missing_keys(self, model_state_dict: Dict[str, torch.Tensor]):
        all_processed_keys = []
        params_to_skip = tuple('.' + name for name in OPTIONAL_PARAMETERS_REGISTRY.get_optional_parameters_names())
        for keys in self._keys.values():
            all_processed_keys.extend(keys)

        for key in model_state_dict.keys():
            if key not in all_processed_keys:
                if key.endswith(params_to_skip):
                    self.add_key(key, ProcessedKeyStatus.SKIPPED)
                    nncf_logger.warning("The optional parameter {} is missed in the loaded state".format(key))
                else:
                    self.add_key(key, ProcessedKeyStatus.MISSING)

    def handle_problematic(self, is_resume: bool, are_all_loaded_params_matched: bool):
        """
        Reports about errors during the matching state_dict_to_load parameters to the model's state_dict ones.
        It raises an error if is_resume is True or prints warning when it's False. The report happens if
        state_dict_to_load has parameters that could not be matched to model parameters or if the shape of parameters is
        not matching. If some parameters required by model are missing in state_dict_to_load reporting occurs for
        non optional parameters only or when not all parameters from state_dict_to_load match.
        :param is_resume: Determines the behavior when the function cannot do a successful parameter match when loading.
        :param are_all_loaded_params_matched: whether all parameters to load match with model parameters
        """
        error_msgs = []

        def add_error_msg(name, keys_):
            error_msgs.insert(
                0, '{} key(s):\n{}. '.format(name,
                                             ',\n'.join('\t\t"{}"'.format(k) for k in keys_)))

        for key_status, keys in self._keys.items():
            is_missing = key_status == ProcessedKeyStatus.MISSING
            erroneous = key_status in (ProcessedKeyStatus.SIZE_MISMATCHED, ProcessedKeyStatus.UNEXPECTED)
            if keys and (erroneous or is_missing and (is_resume or not are_all_loaded_params_matched)):
                add_error_msg(key_status.value, keys)
        if error_msgs:
            error_msg = 'Error(s) when loading model parameters:\n\t{}'.format("\n\t".join(error_msgs))
            if is_resume:
                raise RuntimeError(error_msg)
            nncf_logger.warning(error_msg)


class KeyMatcher:
    """
    Matches state_dict_to_load parameters to the model's state_dict parameters while discarding irrelevant prefixes
    added during wrapping in NNCFNetwork or DataParallel/DistributedDataParallel objects, skipping registered optional
    parameters, and forms the model state dict with matched parameters.
    """

    def __init__(self, is_resume: bool,
                 state_dict_to_load: Dict[str, torch.Tensor], model_state_dict: Dict[str, torch.Tensor]):
        self._is_resume = is_resume
        self.state_dict_to_load = state_dict_to_load

        self.model_state_dict = model_state_dict
        self._processed_keys = ProcessedKeys()
        self._new_dict = {}
        self._num_params_to_load = len(state_dict_to_load.items())

    def run(self) -> Dict[str, torch.Tensor]:
        """
        :return: the model state dict with matched parameters
        """
        from nncf.nncf_network import MODEL_WRAPPED_BY_NNCF_ATTR_NAME
        clip_patterns = [MODEL_WRAPPED_BY_NNCF_ATTR_NAME + '.', 'module.']

        clipped_keys = list(self.model_state_dict.keys())
        for pattern in clip_patterns:
            for i, _ in enumerate(clipped_keys):
                clipped_keys[i] = clipped_keys[i].replace(pattern, '')

        clipped_key_to_model_key_dict = dict(zip(clipped_keys, self.model_state_dict.keys()))


        norm_clipped_keys = {}
        collisions = []
        for clipped_key, orig_key in clipped_key_to_model_key_dict.items():
            normalized_key = self._key_normalizer(clipped_key)
            if normalized_key in norm_clipped_keys:
                collisions.append(clipped_key)
            norm_clipped_keys[normalized_key] = orig_key

        has_legacy_storage_keys = False
        for (saved_key, saved_value) in self.state_dict_to_load.items():
            clipped_saved_key = saved_key
            for pattern in clip_patterns:
                clipped_saved_key = clipped_saved_key.replace(pattern, '')

            clipped_saved_key, did_replace = self._replace_legacy_act_quantizer_storage_name(
                clipped_saved_key
            )

            if did_replace:
                has_legacy_storage_keys = True

            if clipped_saved_key in clipped_key_to_model_key_dict:
                key = clipped_key_to_model_key_dict[clipped_saved_key]
                self._check_parameter_size(key, saved_value)
            else:
                norm_clipped_saved_key = self._key_normalizer(clipped_saved_key)
                in_norm_clipped = norm_clipped_saved_key in norm_clipped_keys
                not_in_collisions = clipped_saved_key not in collisions
                if in_norm_clipped and not_in_collisions and not self._is_resume:
                    key = norm_clipped_keys[norm_clipped_saved_key]
                    self._check_parameter_size(key, saved_value)
                else:
                    self._processed_keys.add_key(saved_key, ProcessedKeyStatus.UNEXPECTED)

        if has_legacy_storage_keys:
            warnings.warn('Legacy NNCF-enabled .pth checkpoint has been loaded! '
                          'The "activation_quantizers" storage key is replaced with '
                          '"external_quantizers" in newer versions of NNCF, and support '
                          'for the legacy storage key will be dropped in a future release. '
                          'This checkpoint will be loaded; update your checkpoint file by saving this model\'s'
                          'checkpoint file again.', category=DeprecationWarning)

        self._processed_keys.add_skipped_and_missing_keys(self.model_state_dict)
        return self._new_dict

    @staticmethod
    def _replace_legacy_act_quantizer_storage_name(checkpoint_key : str) -> Tuple[str, bool]:

        from nncf.nncf_network import LEGACY_ACT_STORAGE_NAME
        from nncf.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
        did_replace = False
        splits = checkpoint_key.split('.')
        if splits[0] == LEGACY_ACT_STORAGE_NAME:
            did_replace = True
            splits[0] = EXTERNAL_QUANTIZERS_STORAGE_NAME
        reconstructed_key = '.'.join(splits)
        return reconstructed_key, did_replace

    def handle_problematic_keys(self):
        """
        Reports about errors during the matching state_dict_to_load parameters to the model's state_dict ones.
        """
        num_matched_params = len(self._new_dict)
        self._processed_keys.handle_problematic(self._is_resume, num_matched_params == self._num_params_to_load)

    @staticmethod
    def _key_normalizer(key: str) -> str:
        new_key = key
        match = re.search('(pre_ops|post_ops)\\.(\\d+?)\\.op', key)
        return new_key if not match else new_key.replace(match.group(), 'operation')

    def _check_parameter_size(self, key: str, value_to_load: torch.Tensor):
        size_of_value_to_load = value_to_load.size()
        size = self.model_state_dict[key].size()
        if size_of_value_to_load == size:
            self._new_dict[key] = value_to_load
            self._processed_keys.add_key(key, ProcessedKeyStatus.MATCHED)
        else:
            nncf_logger.warning("Different size of value of '{}' in resuming dictionary ({}) and in model ({})"
                                .format(key, size_of_value_to_load, size, ))
            self._processed_keys.add_key(key, ProcessedKeyStatus.SIZE_MISMATCHED)
