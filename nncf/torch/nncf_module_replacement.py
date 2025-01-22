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
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

import torch
from torch import nn

import nncf
from nncf.common.logging import nncf_logger
from nncf.common.scopes import matches_any
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.layers import NNCF_MODULES
from nncf.torch.layers import NNCF_MODULES_DICT
from nncf.torch.layers import NNCF_MODULES_MAP
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.layers import UNWRAPPED_USER_MODULES
from nncf.torch.layers import add_nncf_functionality_to_user_module
from nncf.torch.utils import get_model_device


def is_nncf_module(module: nn.Module) -> bool:
    """
    Checks whether given module has been extended with NNCF-enabling functionality.

    :param module: Module to check.
    :returns: True if module is an NNCF-extended module.
    """
    for nncf_module_name in NNCF_MODULES:
        if module.__class__.__name__ == nncf_module_name:
            return True
    for nncf_user_wrapped_class in NNCF_WRAPPED_USER_MODULES_DICT.values():
        if module.__class__.__name__ == nncf_user_wrapped_class.__name__:
            return True

    return False


def collect_all_scopes_for_extendable_and_extended_modules(
    model: nn.Module, predicate: Callable = None
) -> Dict[nn.Module, Set[Scope]]:
    """
    Collects all ranges for all modules in the model that match the condition from predicate.

    :param module: The model.
    :param predicate: A predicate function that can be used to filter modules.
    By default, the predicate function filters all NNCF modules and modules that can be replaced with NNCF modules.
    :return: A dictionary mapping modules to sets of scopes.
    """
    retval = {}
    if predicate is None:
        predicate = lambda x: _can_extend(x) or is_nncf_module(x)
    return _collect_modules_and_scopes_recursive_helper(model, Scope(), predicate, retval)


def collect_modules_and_scopes_by_predicate(
    module: nn.Module, predicate: Callable[[torch.nn.Module], bool]
) -> Dict[nn.Module, Set[Scope]]:
    retval = {}
    return _collect_modules_and_scopes_recursive_helper(module, Scope(), predicate, retval)


def _collect_modules_and_scopes_recursive_helper(
    current_module: nn.Module,
    current_scope: Scope,
    collect_predicate: Callable[[torch.nn.Module], bool],
    retval: Dict[nn.Module, Set[Scope]],
    visited_scopes: Set[Scope] = None,
) -> Dict[nn.Module, Set[Scope]]:
    if visited_scopes is None:
        visited_scopes = set()
        current_scope = Scope()
        current_scope.push(ScopeElement(current_module.__class__.__name__))

    if current_scope in visited_scopes:
        return retval

    visited_scopes.add(current_scope)

    for name, child_module in current_module.named_children():
        child_scope_element = ScopeElement(child_module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)

        if collect_predicate(child_module):
            if child_module not in retval:
                retval[child_module] = {child_scope}
            else:
                retval[child_module].add(child_scope)
        _ = _collect_modules_and_scopes_recursive_helper(
            child_module, child_scope, collect_predicate, retval, visited_scopes
        )

    return retval


def _can_extend(module: nn.Module) -> bool:
    """
    Whether the module can be replaced by an NNCF-extended version to enable compression via NNCF.
    :param module: Candidate module for replacement
    :return: Whether the module should be replaced.
    """
    return (
        module.__class__ in NNCF_MODULES_DICT.values()
        or module.__class__ in UNWRAPPED_USER_MODULES.registry_dict.values()
    )


def nncf_module_from(module: nn.Module) -> nn.Module:
    """
    Returns an NNCF-extended module from a given module that NNCF knows how to extend.

    :param module: The module to be replaced. Must be registered in NNCF as replaceable.
    :returns: The module extended with NNCF functionality.
    """
    assert _can_extend(module)
    for nncf_module_class, original_module_class in NNCF_MODULES_DICT.items():
        if module.__class__ == original_module_class:
            return nncf_module_class.from_module(module)
    for user_module_class in UNWRAPPED_USER_MODULES.registry_dict.values():
        if module.__class__ == user_module_class:
            nncf_module = deepcopy(module)
            nncf_module = add_nncf_functionality_to_user_module(nncf_module)
            return nncf_module
    raise nncf.InternalError(f"Could not extend module {module} with NNCF functionality!")


def replace_modules_by_nncf_modules(
    model: nn.Module,
    ignored_scopes: Optional[List[str]] = None,
    target_scopes: Optional[List[str]] = None,
    eval_op_scopes: Optional[List[Scope]] = None,
    custom_replacer: Callable[[nn.Module], None] = None,
    predicate_fn: Optional[Callable] = None,
) -> Tuple[nn.Module, Dict[torch.nn.Module, List[Scope]]]:
    """
    Replaces certain modules in the model hierarchy with NNCF-wrapped versions of the same modules.
    The modules to be replaced are statically defined in `nncf.torch.layers.NNCF_MODULES_DICT` and dynamically
    extended if the user utilized the `nncf.torch.layers.register_module` decorator to enable their custom
    weighted modules.
    Does not replace sub-modules of an already replaced module. Modules targeted via "target_scopes" will not
    be replaced if they are not already in another module targeted by "target_scopes".
    Note: for `ignored_scopes`, `target_scopes` and `eval_op_scopes`, the scopes
    in this list must correspond to the *storage* scope, i.e. reflect the hierarchy of the modules in the Python
    object,
    rather than correspond to the way that the underlying forward operations were executed in, which is different
    from the way other `ignored_scopes` and `target_scopes` are written in NNCF.

    :param model: The model in which the modules should be replaced.
    :param ignored_scopes: The list of string representations of the scopes that should be ignored during replacement,
     i.e. in which the replacement should not occur (corresponds to a "denylist").
    :param target_scopes: The list of string representations of the scopes so that the replacement should occur only
     in these scopes (corresponds to an "allowlist")
    :param eval_op_scopes: The list of the scopes for the modules that are executed in evaluation mode (i.e. the modules
    that end up having a scope not in this list will be considered train-only and will not be replaced).
    :param custom_replacer: The function to be used instead of the regular approach to replace a module with NNCF-
      extended counterpart.
    :param predicate_fn: The function to find modules that can be replaced.
    :return: The model with the modules replaced and the dictionary of all extended modules vs list of scopes through
    which the module is accessible. The list of scope shall be sorted lexicographically w.r.t. the string representation
    of the Scope objects.
    The dictionary will also include the extended modules that have already been present in the model.
    """
    modules_vs_scopes_dict = collect_all_scopes_for_extendable_and_extended_modules(model, predicate=predicate_fn)
    inter_dict: Dict[nn.Module, Set[Scope]] = {}
    ret_dict: Dict[nn.Module, List[Scope]] = {}
    for module, scope_set in modules_vs_scopes_dict.items():
        if is_nncf_module(module):
            # The module has already been extended, track it in the return value
            ret_dict[module] = list(sorted(scope_set, key=str))
            continue
        should_process = _is_scopes_allow_replacement(
            scope_set, ignored_scopes, target_scopes, eval_op_scopes
        ) and not _is_module_only_in_user_module(scope_set)
        if should_process:
            device = get_model_device(module)

            if custom_replacer is not None:
                replaced_module = custom_replacer(module)
            else:
                replaced_module = nncf_module_from(module)

            replaced_module.to(device)
            inter_dict[replaced_module] = scope_set

            new_scope_set = set()
            for scope in scope_set:
                # Adjust the returned scope so that it has the "NNCF..." version of the module
                new_scope = scope.copy()
                new_scope[-1].calling_module_class_name = replaced_module.__class__.__name__
                new_scope_set.add(new_scope)
                ret_dict[replaced_module] = list(sorted(new_scope_set, key=str))

    for replaced_module, old_scope_set in inter_dict.items():
        for old_scope in old_scope_set:
            # Should replace by all available scopes, otherwise some sub-containers may keep references to old
            # modules
            _replace_module_by_scope(model, old_scope, replaced_module)
    return model, ret_dict


def get_original_module_scope_from_nncf_module_scope(nncf_module_scope: Scope) -> Scope:
    original_module_scope = nncf_module_scope.copy()
    elt = original_module_scope.pop()
    assert elt.calling_module_class_name in NNCF_MODULES_MAP
    elt.calling_module_class_name = NNCF_MODULES_MAP[elt.calling_module_class_name]
    original_module_scope.push(elt)
    return original_module_scope


def _is_module_only_in_user_module(scope_set: Set[Scope]) -> bool:
    for scope in scope_set:
        has_at_least_one_user_module = False
        for user_module_class in UNWRAPPED_USER_MODULES.registry_dict.values():
            if _has_user_module(scope, user_module_class) or _has_user_module(
                scope, NNCF_WRAPPED_USER_MODULES_DICT[user_module_class]
            ):
                has_at_least_one_user_module = True
        if not has_at_least_one_user_module:
            # At least one scope from the input scope_set is located outside user module
            return False
    return True


def _has_user_module(scope: Scope, user_module_class: Type[torch.nn.Module]) -> bool:
    for scope_element in scope.scope_elements[:-1]:
        if scope_element.calling_module_class_name == user_module_class.__name__:
            return True
    return False


def _replace_module_by_scope(base_model: torch.nn.Module, scope: Scope, replaced_module: torch.nn.Module):
    """
    Accesses the module pointed to by scope in the base model and replaces it with an NNCF-extended module.

    :param base_model: The model relative to which the scope is built.
    :param scope: Scope of the module in base_module to be extended with NNCF functionality.
    :param custom_extender: The module that will replace the one pointed to by scope.
    :param reset: Whether the module should be reset.
    """
    curr_module = base_model
    owning_module = base_model
    for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
        child_module = curr_module._modules.get(scope_element.calling_field_name)
        if child_module is None:
            raise nncf.InternalError(
                "Could not find a {} module member in {} module of scope {} during module replacement".format(
                    scope_element.calling_field_name, scope_element.calling_module_class_name, str(scope)
                )
            )
        owning_module = curr_module
        curr_module = child_module

    if is_nncf_module(curr_module):
        # Already replaced, possibly via another scope alias
        nncf_logger.debug(f"Module at {str(scope)} was already extended.")
        return

    nncf_logger.debug(f"Extending module at {str(scope)}...")
    last_calling_field_name = scope[-1].calling_field_name
    if isinstance(owning_module, nn.Sequential):
        owning_module._modules[last_calling_field_name] = replaced_module
    else:
        setattr(owning_module, last_calling_field_name, replaced_module)


def _is_scopes_allow_replacement(
    scope_set_for_module: Set[Scope],
    ignored_scopes: Optional[List[str]] = None,
    target_scopes: Optional[List[str]] = None,
    eval_op_scopes: Optional[List[Scope]] = None,
) -> bool:
    used_in_eval = False
    for scope in scope_set_for_module:
        if matches_any(str(scope), ignored_scopes):
            nncf_logger.info(
                f"Not processing a module that matched to an ignored scope in config; module scope = {str(scope)}"
            )
            return False
        if eval_op_scopes is not None:
            for eval_op_scope in eval_op_scopes:
                # child_scope isn't ignored, if there's at least a single operation or a module called in eval mode
                # inside it
                if eval_op_scope in scope:
                    used_in_eval = True
                    break

    if eval_op_scopes is not None and not used_in_eval:
        return False

    if target_scopes is not None:
        is_any_scope_in_target = False
        for scope in scope_set_for_module:
            if matches_any(str(scope), target_scopes):
                is_any_scope_in_target = True
                break
        if not is_any_scope_in_target:
            nncf_logger.info(
                f"Not processing a module outside target scope specified in config; "
                f"module known under scope(s) = {';'.join([str(x) for x in scope_set_for_module])}"
            )
            return False

    return True
