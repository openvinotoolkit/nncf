"""
 Copyright (c) 2019 Intel Corporation
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
from torch import nn
from typing import List

from nncf.torch.layers import NNCF_MODULES_DICT, NNCF_MODULES, \
    add_nncf_functionality_to_user_module, NNCF_WRAPPED_USER_MODULES_DICT
from nncf.common.utils.helpers import matches_any
from nncf.torch.dynamic_graph.scope import ScopeElement
from nncf.torch.dynamic_graph.scope import Scope

from nncf.common.utils.logger import logger as nncf_logger


def is_nncf_module(module):
    for nncf_module_name in NNCF_MODULES:
        if module.__class__.__name__ == nncf_module_name:
            return True
    for nncf_user_wrapped_class in NNCF_WRAPPED_USER_MODULES_DICT.values():
        if module.__class__.__name__ == nncf_user_wrapped_class.__name__:
            return True

    return False


def replace_module_by_nncf_module(module: nn.Module):
    for nncf_module_type, module_type in NNCF_MODULES_DICT.items():
        if module.__class__.__name__ == module_type.__name__:
            nncf_module = module
            if not module.__class__.__name__ == nncf_module_type.__name__:
                nncf_module = nncf_module_type.from_module(module)
            return nncf_module
    from nncf.torch.layers import UNWRAPPED_USER_MODULES
    for _, user_module_type in UNWRAPPED_USER_MODULES.registry_dict.items():
        if module.__class__ == user_module_type:
            nncf_module = deepcopy(module)
            nncf_module = add_nncf_functionality_to_user_module(nncf_module)
            return nncf_module
    return module


def replace_modules_by_nncf_modules(model: nn.Module, ignored_scopes=None, target_scopes=None,
                                    eval_op_scopes: List[Scope] = None,
                                    reset: bool = False) -> (nn.Module, List[Scope]):
    replace_fn = partial(replace_module_by_nncf_module)
    affected_scopes = []  # type: List
    return replace_modules(model, replace_fn, affected_scopes,
                           ignored_scopes=ignored_scopes, target_scopes=target_scopes,
                           eval_op_scopes=eval_op_scopes, reset=reset)


def set_replaced_module_by_name(model, name, replaced_module):
    if isinstance(model, nn.Sequential):
        # pylint: disable=protected-access
        model._modules[name] = replaced_module
    else:
        setattr(model, name, replaced_module)


# pylint: disable=too-many-branches
def replace_modules(model: nn.Module, replace_fn, affected_scopes, ignored_scopes=None, target_scopes=None, memo=None,
                    current_scope=None, eval_op_scopes: List[Scope] = None, reset: bool = False):
    if memo is None:
        memo = set()
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))

    if model in memo:
        return model, affected_scopes

    memo.add(model)
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
                    nncf_logger.info("Ignored wrapping modules specified in scope: {}".format(child_scope))
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
                    nncf_logger.info(
                        "Ignored wrapping modules not called in eval mode in scope: {}".format(child_scope))
                    continue

                if target_scopes is None or matches_any(str(child_scope), target_scopes):
                    nncf_logger.info("Wrapping module {} by {}".format(str(child_scope),
                                                                       str(replaced_scope)))
                    set_replaced_module_by_name(model, name, replaced_module)
                    affected_scopes.append(replaced_scope)
            elif is_nncf_module(replaced_module):
                # Got an NNCF-wrapped module from previous compression stage, track its scope as well
                affected_scopes.append(replaced_scope)
                if reset:
                    replaced_module.reset()
        _, affected_scopes = replace_modules(module, replace_fn, affected_scopes, ignored_scopes, target_scopes,
                                             memo, child_scope, eval_op_scopes, reset=reset)
    return model, affected_scopes
