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
import re
from enum import Enum
from typing import Dict, List, Set, Tuple

import torch

import nncf
from nncf.common.deprecation import warning_deprecated
from nncf.common.logging import nncf_logger
from nncf.common.utils.api_marker import api


@api(canonical_alias="nncf.torch.load_state")
def load_state(
    model: torch.nn.Module, state_dict_to_load: dict, is_resume: bool = False, keys_to_ignore: List[str] = None
) -> int:
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
    :param keys_to_ignore: A list of parameter names that should be skipped from matching process.
    :return: The number of state_dict_to_load entries successfully matched and loaded into model.
    """

    model_state_dict = model.state_dict()

    from nncf.torch.utils import maybe_convert_legacy_names_in_model_state

    maybe_convert_legacy_names_in_model_state(state_dict_to_load)
    key_matcher = KeyMatcher(is_resume, state_dict_to_load, model_state_dict, keys_to_ignore)
    new_dict = key_matcher.run()
    num_loaded_params = len(new_dict)
    key_matcher.handle_problematic_keys()
    nncf_logger.info(f"Loaded {num_loaded_params}/{len(model_state_dict.items())} parameters")

    model.load_state_dict(new_dict, strict=False)
    return num_loaded_params


class ParametersRegistry:
    """
    Provides an interface to register parameters and get access to all of them.
    """

    def __init__(self):
        self._parameters_names = set()

    def register(self, parameter_name: str):
        self._parameters_names.add(parameter_name)

    def get_parameters_names(self) -> Set[str]:
        return self._parameters_names


# If optional parameter is missed in a checkpoint, it can be loaded without an error in a strict mode.
# New parameters can be introduced for the model without breaking backward compatibility with old checkpoint.
OPTIONAL_PARAMETERS_REGISTRY = ParametersRegistry()


class ProcessedKeyStatus(Enum):
    """Status of matching checkpoint key with model keys"""

    MATCHED = "Matched"
    MISSING = "Missing"
    UNEXPECTED = "Unexpected"
    SIZE_MISMATCHED = "Size mismatched"
    SKIPPED = "Skipped"


class ProcessedKeys:
    """Contains checkpoint keys with their status of matching with model keys"""

    def __init__(self):
        self._keys: Dict[ProcessedKeyStatus, Set[str]] = {}
        for key_status in ProcessedKeyStatus:
            self._keys[key_status] = set()

    def add_key(self, key: str, status: ProcessedKeyStatus):
        self._keys[status].add(key)

    def extend_keys(self, keys: List[str], status: ProcessedKeyStatus):
        self._keys[status].update(keys)

    def add_skipped_and_missing_keys(self, model_state_dict: Dict[str, torch.Tensor]):
        all_processed_keys = []
        optional_param_names = OPTIONAL_PARAMETERS_REGISTRY.get_parameters_names()
        params_to_skip = tuple("." + name for name in optional_param_names)
        for keys in self._keys.values():
            all_processed_keys.extend(keys)

        for key in model_state_dict:
            if key not in all_processed_keys:
                if key.endswith(params_to_skip) or key in optional_param_names:
                    self.add_key(key, ProcessedKeyStatus.SKIPPED)
                    nncf_logger.warning(f"The optional parameter {key} is missing in the loaded state.")
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
            error_msgs.insert(0, "{} key(s):\n{}. ".format(name, ",\n".join('\t\t"{}"'.format(k) for k in keys_)))

        for key_status, keys in self._keys.items():
            is_missing = key_status == ProcessedKeyStatus.MISSING
            erroneous = key_status in (ProcessedKeyStatus.SIZE_MISMATCHED, ProcessedKeyStatus.UNEXPECTED)
            if keys and (erroneous or is_missing and (is_resume or not are_all_loaded_params_matched)):
                add_error_msg(key_status.value, keys)
        if error_msgs:
            error_msg = "Error(s) when loading model parameters:\n\t{}".format("\n\t".join(error_msgs))
            if is_resume:
                raise nncf.InternalError(error_msg)
            nncf_logger.error(error_msg)


class NormalizedKeys:
    """
    Contains normalized form of parameters. It helps to discard irrelevant prefixes added during wrapping in
    NNCFNetwork or DataParallel/DistributedDataParallel objects, to handle legacy parameters' names and to match
    unified compression parameters from the separate ones.
    """

    def __init__(self, keys: List[str], keys_to_ignore: List[str]):
        self._unique_normalized_key_vs_orig_key_map = {}
        self.is_unified_group_detected = False
        unique_clipped_key_vs_orig_key_map, ignored_keys = self._clip_keys_without_collisions(keys, keys_to_ignore)
        self.ignored_orig_keys = ignored_keys
        ignored_keys = self._normalize_keys_without_collisions(unique_clipped_key_vs_orig_key_map, keys_to_ignore)
        self.ignored_orig_keys.extend(ignored_keys)

    def __contains__(self, key: str):
        return key in self._unique_normalized_key_vs_orig_key_map

    def __iter__(self):
        return iter(self._unique_normalized_key_vs_orig_key_map)

    def get_orig_key(self, normalized_key: str):
        return self._unique_normalized_key_vs_orig_key_map[normalized_key]

    def _normalize_keys_without_collisions(
        self, unique_clipped_key_vs_orig_key_map: Dict[str, str], keys_to_ignore: List[str]
    ) -> List[str]:
        ignored_keys = []
        normalized_key_vs_clipped_key_list_map = {}
        for clipped_key in unique_clipped_key_vs_orig_key_map:
            replaced_keys = self._key_replacer(clipped_key)
            if len(replaced_keys) > 1:
                self.is_unified_group_detected = True

            for replaced_key in replaced_keys:
                if replaced_key in keys_to_ignore:
                    orig_key = unique_clipped_key_vs_orig_key_map[clipped_key]
                    ignored_keys.append(orig_key)
                    continue
                if replaced_key in normalized_key_vs_clipped_key_list_map:
                    normalized_key_vs_clipped_key_list_map[replaced_key].append(clipped_key)
                else:
                    normalized_key_vs_clipped_key_list_map[replaced_key] = [clipped_key]
        # keep clipped keys if their normalization led to a collisions
        for normalized_key, value in normalized_key_vs_clipped_key_list_map.items():
            list_clipped_keys = value
            if len(list_clipped_keys) == 1:
                clipped_key = list_clipped_keys[0]
                orig_key = unique_clipped_key_vs_orig_key_map[clipped_key]
                self._unique_normalized_key_vs_orig_key_map[normalized_key] = orig_key
            else:
                for clipped_key in list_clipped_keys:
                    orig_key = unique_clipped_key_vs_orig_key_map[clipped_key]
                    self._unique_normalized_key_vs_orig_key_map[clipped_key] = orig_key
        return ignored_keys

    @staticmethod
    def _clip_keys_without_collisions(keys: List[str], keys_to_ignore: List[str]) -> Tuple[Dict[str, str], List[str]]:
        clipped_key_vs_orig_key_list_map = {}
        ignored_keys = []
        for orig_key in keys:
            clipped_key = NormalizedKeys._key_clipper(orig_key)
            if clipped_key in keys_to_ignore:
                ignored_keys.append(orig_key)
                continue
            if clipped_key in clipped_key_vs_orig_key_list_map:
                clipped_key_vs_orig_key_list_map[clipped_key].append(orig_key)
            else:
                clipped_key_vs_orig_key_list_map[clipped_key] = [orig_key]
        # keep original keys if their clipping led to a collisions
        unique_clipped_key_vs_orig_key_map = {}
        for clipped_key, val in clipped_key_vs_orig_key_list_map.items():
            list_orig_keys = val
            if len(list_orig_keys) == 1:
                unique_clipped_key_vs_orig_key_map[clipped_key] = list_orig_keys[0]
            else:
                for orig_key in list_orig_keys:
                    unique_clipped_key_vs_orig_key_map[orig_key] = orig_key
        return unique_clipped_key_vs_orig_key_map, ignored_keys

    @staticmethod
    def _key_clipper(key: str) -> str:
        new_key = key

        clip_patterns = ["module.", "|OUTPUT", "|INPUT"]
        for pattern in clip_patterns:
            new_key = new_key.replace(pattern, "")
        return new_key

    def _key_replacer(self, key: str) -> List[str]:
        new_key = key

        match = re.search("(pre_ops|post_ops)\\.(\\d+?)\\.op", key)
        new_key = new_key if not match else new_key.replace(match.group(), "operation")
        result = self._split_unified_parameters(new_key)
        if len(result) > 1:
            self.is_unified_group_detected = True
        return result

    @staticmethod
    def _split_unified_parameters(new_key: str) -> List[str]:
        """covers unified activation quantizers case, e.g.
            external_quantizers.RELU_0;RELU_1;RELU_2.op
        Result of this function is full names of individual parameters:
            external_quantizers.RELU_2.op
            external_quantizers.RELU_1.op
            external_quantizers.RELU_0.op
        It's utilized to match parameters from checkpoints without unified operations to not start training
        compression from scratch, but instead initialize group of parameters by one of the matched individual one.
        Returns original key if there's no ';' and operation doesn't start with EXTERNAL_QUANTIZERS_STORAGE_NAME
        """
        result = [new_key]
        from nncf.torch.quantization.external_quantizer import EXTERNAL_QUANTIZERS_STORAGE_PREFIX

        if ";" in new_key and new_key.startswith(EXTERNAL_QUANTIZERS_STORAGE_PREFIX):
            group_of_keys = new_key.split(";")
            last_key = group_of_keys[-1]
            common_op = last_key.split(".")[-1]
            result = [group_of_keys[0] + "." + common_op, EXTERNAL_QUANTIZERS_STORAGE_PREFIX + "." + last_key]
            for key in group_of_keys[1:-1]:
                result.append(EXTERNAL_QUANTIZERS_STORAGE_PREFIX + "." + key + "." + common_op)
        return result


class KeyMatcher:
    """
    Matches state_dict_to_load parameters to the model's state_dict parameters while discarding irrelevant prefixes
    added during wrapping in NNCFNetwork or DataParallel/DistributedDataParallel objects, skipping registered optional
    parameters, handling legacy parameters' names, ignoring the order of pre/post operations, matching unified
    compression parameters from the separate ones and forms the model state dict with matched parameters.
    """

    def __init__(
        self,
        is_resume: bool,
        state_dict_to_load: Dict[str, torch.Tensor],
        model_state_dict: Dict[str, torch.Tensor],
        ignored_keys: List[str] = None,
    ):
        """
        :param state_dict_to_load: A state dict containing the parameters to be loaded into the model.
        :param ignored_keys: list of parameters to skip from matching process on loading.
        """
        self._is_resume = is_resume
        self.state_dict_to_load = state_dict_to_load

        self.model_state_dict = model_state_dict
        self._processed_keys = ProcessedKeys()
        self._new_dict = {}
        self._num_params_to_load = len(state_dict_to_load.items())
        self.ignored_keys = ignored_keys if ignored_keys else []

    def run(self) -> Dict[str, torch.Tensor]:
        """
        :return: the model state dict with matched parameters
        """
        normalized_model_keys = NormalizedKeys(list(self.model_state_dict.keys()), keys_to_ignore=self.ignored_keys)
        normalized_keys_to_load = NormalizedKeys(list(self.state_dict_to_load.keys()), keys_to_ignore=self.ignored_keys)

        has_version_agnostic_names = False
        cross_match_key_map = self._cross_match_version_agnostic_names(
            list(normalized_keys_to_load), list(normalized_model_keys)
        )

        for matched_checkpoint_key, matched_model_key in cross_match_key_map.items():
            if matched_checkpoint_key != matched_model_key:
                has_version_agnostic_names = True

        if has_version_agnostic_names:
            warning_deprecated(
                "Legacy NNCF-enabled .pth checkpoint has been loaded! "
                "The version-agnostic `RELU` operator name entries in the state dict "
                "have been deprecated. "
                "The loader will try to match these entries to the corresponding `relu` and `relu_` op "
                "names. The newly exported checkpoints will be adjusted to the new format."
            )

        if normalized_model_keys.is_unified_group_detected and not normalized_keys_to_load.is_unified_group_detected:
            nncf_logger.warning(
                "Unified parameters are detected in the compressed model, but all parameters are independent "
                "and separate in the loading checkpoint. The unified parameters will be initialized by one of"
                "the corresponding separate parameter in the checkpoint. That may slightly degrade the "
                "accuracy, but should allow to not start training compression from scratch with unified "
                "params."
            )
        ignored_keys = normalized_model_keys.ignored_orig_keys + normalized_keys_to_load.ignored_orig_keys
        self._processed_keys.extend_keys(ignored_keys, ProcessedKeyStatus.SKIPPED)
        if ignored_keys:
            ignored_keys_str = "\n".join(set(ignored_keys))
            nncf_logger.warning(
                f"Following parameters were skipped from matching checkpoint's keys:\n{ignored_keys_str}"
            )

        loaded_prefixless_keys = False
        for normalized_key_to_load in normalized_keys_to_load:
            key_to_load = normalized_keys_to_load.get_orig_key(normalized_key_to_load)
            normalized_key_to_load = cross_match_key_map.get(normalized_key_to_load, normalized_key_to_load)
            if normalized_key_to_load in normalized_model_keys:
                model_key = normalized_model_keys.get_orig_key(normalized_key_to_load)
                if "_nncf." + key_to_load == model_key:
                    loaded_prefixless_keys = True
                value_to_load = self.state_dict_to_load[key_to_load]
                size_of_value_to_load = value_to_load.size()
                size_of_model_value = self.model_state_dict[model_key].size()
                if size_of_value_to_load == size_of_model_value:
                    self._new_dict[model_key] = value_to_load
                    self._processed_keys.add_key(model_key, ProcessedKeyStatus.MATCHED)
                else:
                    nncf_logger.warning(
                        f"Different size of value of '{model_key}' "
                        f"in resuming dictionary ({size_of_value_to_load}) and in model ({size_of_model_value})"
                    )
                    self._processed_keys.add_key(model_key, ProcessedKeyStatus.SIZE_MISMATCHED)
            else:
                self._processed_keys.add_key(key_to_load, ProcessedKeyStatus.UNEXPECTED)
        self._processed_keys.add_skipped_and_missing_keys(self.model_state_dict)
        if loaded_prefixless_keys:
            warning_deprecated(
                "Legacy NNCF-enabled .pth checkpoint has been loaded! "
                'Some storage keys in the loaded checkpoint should now have a "_nncf." prefix,'
                "support for the legacy storage key will be dropped in a future release. "
                "This checkpoint will be loaded; update your checkpoint file by saving this model's"
                "checkpoint file again."
            )
        return self._new_dict

    @staticmethod
    def _cross_match_version_agnostic_names(
        normalized_keys_to_load: List[str], normalized_model_keys: List[str]
    ) -> Dict[str, str]:
        """
        Handles the situation where the normalized_keys_to_load contain legacy version-agnostic names
        of operations, such as `RELU`.

        :param normalized_keys_to_load: A list of keys in the checkpoint, potentially with version-agnostic names
        :param normalized_model_keys: A list of keys in the model, without version-agnostic names.
        :return: A mapping of the checkpoint key to a model key that matches version-agnostic names with their
            torch-specific counterparts.
        """
        version_agnostic_to_specific_names = {"RELU": {"relu", "relu_"}}
        retval = {}
        processed_keys_to_load = normalized_keys_to_load

        for model_key in normalized_model_keys:
            for agnostic_op_name, specific_op_name_set in version_agnostic_to_specific_names.items():
                matches_for_curr_agnostic_op_name = []
                has_specific_op_name = False
                for specific_op_name in specific_op_name_set:
                    # Have to take care not to replace the matches to the class names
                    # The op names in existing checkpoint can only appear in external quantizers,
                    # i.e. external_quantizers.ResNet/ReLU[relu]/relu_0.signed_tensor, so composing a regex to match
                    # for that
                    slash_split_str = model_key.split("/")
                    last_portion = slash_split_str[-1]
                    if specific_op_name in last_portion:
                        last_portion = last_portion.replace(specific_op_name, agnostic_op_name, 1)
                        has_specific_op_name = True
                    slash_split_str[-1] = last_portion
                    agnostic_version_of_model_key = "/".join(slash_split_str)
                    processed_agnostic_version_of_model_key = agnostic_version_of_model_key
                    if processed_agnostic_version_of_model_key in processed_keys_to_load:
                        idx = processed_keys_to_load.index(processed_agnostic_version_of_model_key)
                        matches_for_curr_agnostic_op_name.append(normalized_keys_to_load[idx])

                if not has_specific_op_name:
                    if matches_for_curr_agnostic_op_name:
                        checkpoint_matched_key = next(iter(matches_for_curr_agnostic_op_name))
                        retval[checkpoint_matched_key] = model_key
                elif len(matches_for_curr_agnostic_op_name) == 1:
                    checkpoint_matched_key = next(iter(matches_for_curr_agnostic_op_name))
                    retval[checkpoint_matched_key] = model_key
                elif len(matches_for_curr_agnostic_op_name) == 0:
                    nncf_logger.debug(f"Failed to match a version-specific key: {model_key}")
                elif len(matches_for_curr_agnostic_op_name) > 1:
                    nncf_logger.debug(
                        f"More than one match for the version specific key: {model_key}\n"
                        f"Matches:\n"
                        f"{', '.join(matches_for_curr_agnostic_op_name)}"
                    )

        return retval

    def handle_problematic_keys(self):
        """
        Reports about errors during the matching state_dict_to_load parameters to the model's state_dict ones.
        """
        num_matched_params = len(self._new_dict)
        self._processed_keys.handle_problematic(self._is_resume, num_matched_params == self._num_params_to_load)
