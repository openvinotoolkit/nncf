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
from copy import deepcopy
from typing import Dict, List

from nncf.common.graph import NNCFNode
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.scopes import matches_any


def get_scoped_quantizer_config(
    base_config: QuantizerConfig, scope_str: str, scope_overrides: Dict = None
) -> QuantizerConfig:
    """
    Returns a QuantizerConfig which is based on a given config, which will have overrides
    applied on top of it according to the dictionary of per-scope overrides.

    :param base_config: The base quantizer configuration - corresponding parameters will
      be used in the returned qconfig if no override for this parameter is given.
    :param scope_str: A string identifier of the spot in the model that will be associated with
      this quantizer.
    :param scope_overrides: A dictionary of scope strings vs. dict of overrides for the corresponding
      scope.
    :return: The base configuration with overrides applied on top of it.
    """
    qconfig = deepcopy(base_config)
    if scope_overrides is None:
        scope_overrides = {}
    for overridden_scope in scope_overrides:
        if matches_any(scope_str, overridden_scope):
            config_overrides = scope_overrides[overridden_scope]
            if config_overrides.get("bits") is not None:
                qconfig.num_bits = config_overrides["bits"]
            if config_overrides.get("mode") is not None:
                qconfig.mode = config_overrides["mode"]
            if config_overrides.get("per_channel") is not None:
                qconfig.per_channel = config_overrides["per_channel"]
            if config_overrides.get("signed") is not None:
                qconfig.signedness_to_force = config_overrides["signed"]
    return qconfig


def assign_qconfig_lists_to_modules(
    nodes_with_weights: List[NNCFNode],
    default_weight_qconfig: QuantizerConfig,
    global_weight_constraints: QuantizationConstraints = None,
    scope_overrides_dict: Dict = None,
    hw_config: HWConfig = None,
) -> Dict[NNCFNode, List[QuantizerConfig]]:
    """
    Assigns a list of possible quantizer configurations (as determined by HW config, defaults and overrides)
    to each weighted node that was passed.

    :param nodes_with_weights: The nodes in NNCFGraph that correspond to weighted operations.
    :param default_weight_qconfig: The default quantizer configuration for weights, to be used if
      no other information is given.
    :param global_weight_constraints: The constraints imposed on all weights by the NNCFConfig .json file, such
      as "all symmetric" or "all per-channel" etc.
    :param scope_overrides_dict: The dictionary of strings vs dict of overrides for a given weight quantizer.
      The strings are matched against the name of the NNCFNodes in nodes_with_weights.
    :param hw_config: The HWConfig object to be used for device-specific constraints on allowed weights.
    :return: A dict of each weighted node vs. the list of quantizer configs allowed for quantizing the associated
      weights
    """
    retval: Dict[NNCFNode, List[QuantizerConfig]] = {}
    default_qconfig = deepcopy(default_weight_qconfig)
    if global_weight_constraints is not None:
        default_qconfig = global_weight_constraints.apply_constraints_to(default_qconfig)
    if scope_overrides_dict is None:
        scope_overrides_dict = {}
    weight_scope_overrides_dict = scope_overrides_dict.get("weights")
    if hw_config is not None:
        meta_vs_qconfig_map = hw_config.get_metatype_vs_quantizer_configs_map(for_weights=True)
    for node in nodes_with_weights:
        qconfig_for_current_scope = get_scoped_quantizer_config(
            default_qconfig, node.node_name, weight_scope_overrides_dict
        )
        if hw_config is None:
            qconfig_list = [qconfig_for_current_scope]
        else:
            metatype = node.metatype
            qconfig_list = meta_vs_qconfig_map[metatype]
            if HWConfig.is_wildcard_quantization(qconfig_list):  # Empty list = wildcard quantization
                qconfig_list = [default_qconfig]
            elif HWConfig.is_qconf_list_corresponding_to_unspecified_op(qconfig_list):
                continue  # The module will not have its weights quantized

            local_constraints = global_weight_constraints
            for overridden_scope, scoped_override_dict in scope_overrides_dict.items():
                if matches_any(node.node_name, overridden_scope):
                    scope_constraints = QuantizationConstraints.from_config_dict(scoped_override_dict)
                    local_constraints = local_constraints.get_updated_constraints(scope_constraints)
            qconfig_list = local_constraints.constrain_qconfig_list(
                node.node_name, hw_config.target_device, qconfig_list
            )

        retval[node] = qconfig_list
    return retval
