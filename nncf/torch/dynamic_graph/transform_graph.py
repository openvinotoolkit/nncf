"""
 Copyright (c) 2023 Intel Corporation
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

from copy import deepcopy
from functools import partial
from typing import Callable
from typing import Set
from typing import Tuple

from typing import List
from typing import Optional

from torch import nn

from nncf.common.logging import nncf_logger
from nncf.common.scopes import matches_any
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.layers import NNCF_MODULES
from nncf.torch.layers import NNCF_MODULES_DICT
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.layers import add_nncf_functionality_to_user_module


def is_nncf_module(module: nn.Module) -> bool:
    """
    Checks weather given module is an instance of
    a nncf layer or not.

    :param module: Module to check.
    :returns: True if module is an instance of a nncf layer
        otherwise False.
    """
    for nncf_module_name in NNCF_MODULES:
        if module.__class__.__name__ == nncf_module_name:
            return True
    for nncf_user_wrapped_class in NNCF_WRAPPED_USER_MODULES_DICT.values():
        if module.__class__.__name__ == nncf_user_wrapped_class.__name__:
            return True

    return False


def replace_module_by_nncf_module(module: nn.Module) -> nn.Module:
    """
    Returns updated modules for modules that could be replaced by a nncf layer.

    :param module: Candidate module for the replacement
    :returns: A correspondent nncf layer if it is possible
        and given module otherwise.
    """
    for nncf_module_type, module_type in NNCF_MODULES_DICT.items():
        if module.__class__.__name__ == module_type.__name__:
            nncf_module = module
            if not module.__class__.__name__ == nncf_module_type.__name__:
                nncf_module = nncf_module_type.from_module(module)
            return nncf_module
    from nncf.torch.layers import UNWRAPPED_USER_MODULES  # pylint: disable=cyclic-import
    for _, user_module_type in UNWRAPPED_USER_MODULES.registry_dict.items():
        if module.__class__ == user_module_type:
            nncf_module = deepcopy(module)
            nncf_module = add_nncf_functionality_to_user_module(nncf_module)
            return nncf_module
    return module


def replace_modules_by_nncf_modules(model: nn.Module,
                                    ignored_scopes: Optional[List[str]] = None,
                                    target_scopes: Optional[List[str]] = None,
                                    eval_op_scopes: Optional[List[Scope]] = None,
                                    reset: Optional[bool] = False) -> Tuple[nn.Module, List[Scope]]:
    """
    Replaces certain modules in the model hierarchy with NNCF-wrapped versions of the same modules.
    The modules to be replaced are statically defined in `nncf.torch.layers.NNCF_MODULES_DICT` and dynamically
    extended if the user utilized the `nncf.torch.layers.register_module` decorator to enable their custom
    weighted modules.
    Does not replace sub-modules of an already replaced module. Modules targeted via "target_scopes" will not
    be replaced if they are not already in another module targeted by "target_scopes".
    Note: for `ignored_scopes`, `target_scopes` and `eval_op_scopes`, the scopes
    in this list must correspond to the *storage* scope, i.e. reflect the hierarchy of the modules in the Python object,
    rather than correspond to the way that the underlying forward operations were executed in, which is different
    from the way other `ignored_scopes` and `target_scopes` are written in NNCF.

    :param model: The model in which the modules should be replaced.
    :param ignored_scopes: The list of string representations of the scopes that should be ignored during replacement,
     i.e. in which the replacement should not occur (corresponds to a "denylist").
    :param target_scopes: The list of string representations of the scopes so that the replacement should occur only
     in these scopes (corresponds to an "allowlist")
    :param eval_op_scopes: The list of the scopes for the modules that are executed in evaluation mode (i.e. the modules
    that end up having a scope not in this list will be considered train-only and will not be replaced).
    :param reset: Whether to reset the NNCF-wrapped modules as a result of this function if the model already contains
     those.
    :return: The model with the modules replaced and the list of storage scopes for the modules that were replaced.
    """

    replace_fn = partial(replace_module_by_nncf_module)
    affected_scopes = []  # type: List
    return replace_modules(model, replace_fn, is_nncf_module, affected_scopes,
                           ignored_scopes=ignored_scopes, target_scopes=target_scopes,
                           eval_op_scopes=eval_op_scopes, reset=reset)


def set_replaced_module_by_name(model: nn.Module,
                                name: str, replaced_module: nn.Module) -> None:
    """
    Replaces `model` nested module with name `name` by the `replaced_module`.

    :param model: Module to replace nested module.
    :param name: Name of the target nested module to replace.
    :param replaced_module: Module to be placed instead of replaced one.
    """

    if isinstance(model, nn.Sequential):
        # pylint: disable=protected-access
        model._modules[name] = replaced_module
    else:
        setattr(model, name, replaced_module)


# pylint: disable=too-many-branches
def replace_modules(model: nn.Module,
                    replace_fn: Callable[[nn.Module], Optional[nn.Module]],
                    stop_branching_fn: Callable[[nn.Module], bool],
                    affected_scopes: List[ScopeElement],
                    ignored_scopes: Optional[List[str]] = None,
                    target_scopes: Optional[List[str]] = None,
                    memo: Optional[Set[ScopeElement]] = None,
                    current_scope: Optional[List[ScopeElement]] = None,
                    eval_op_scopes: Optional[List[Scope]] = None,
                    reset: Optional[bool] = False) -> Tuple[nn.Module, List[Scope]]:
    """
    Recursive helper for the `replace_modules_by_nncf_modules`. See docstring of
    `replace_modules_by_nncf_modules` for a description of the majority of the parameters.
    :param model: Model where the replacing will take place.
    :param replace_fn: Callable that returns updated modules for modules
        that should be replaced.
    :param stop_branching_fn: Predicate for when to stop recursion call for given model.
    """

    if memo is None:
        memo = set()
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))

    if current_scope in memo:
        return model, affected_scopes

    memo.add(current_scope)
    for name, module in model.named_children():
        if module is None:
            continue

        child_scope_element = ScopeElement(module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)
        replaced_module = replace_fn(module)

        if replaced_module is not None:
            replaced_scope_element = ScopeElement(replaced_module.__class__.__name__, name)
            replaced_scope = current_scope.copy()
            replaced_scope.push(replaced_scope_element)
            if module is not replaced_module:
                if matches_any(str(child_scope), ignored_scopes):
                    nncf_logger.info(f"Not processing a module that matched to ignored scope in config: {child_scope}")
                    continue
                if eval_op_scopes is None:
                    eval_op_scopes = []
                is_ignored = True
                for eval_op_scope in eval_op_scopes:
                    # child_scope isn't ignored, if there's at least a single operation or a module called in eval mode
                    # inside it
                    if eval_op_scope in child_scope:
                        is_ignored = False
                        break
                if is_ignored and eval_op_scopes:
                    nncf_logger.info(f"Not processing a module not called in eval mode: {child_scope}")
                    continue

                if target_scopes is None or matches_any(str(child_scope), target_scopes):
                    nncf_logger.debug(f"Wrapping module {str(child_scope)} by {str(replaced_scope)}")
                    set_replaced_module_by_name(model, name, replaced_module)
                    affected_scopes.append(replaced_scope)
            elif is_nncf_module(replaced_module):
                # Got an NNCF-wrapped module from previous compression stage, track its scope as well
                affected_scopes.append(replaced_scope)
                if reset:
                    replaced_module.reset()

        if stop_branching_fn(module):
            continue

        # Prevent recursive call for replaced modules
        if replaced_module is None or module is replaced_module:
            _, affected_scopes = replace_modules(module, replace_fn, stop_branching_fn,
                                                 affected_scopes, ignored_scopes,
                                                 target_scopes, memo, child_scope, eval_op_scopes, reset=reset)
    return model, affected_scopes
