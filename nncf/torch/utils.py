"""
 Copyright (c) 2019-2020 Intel Corporation
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
from collections import OrderedDict
from typing import Dict, Callable, Any, Mapping, Sequence, Set, List
from typing import Tuple
from typing import Type

import numpy as np
import random
import torch
from torch import distributed as dist, nn
from torch.nn import Module, Parameter

from nncf.common.graph import NNCFNodeName
from nncf.common.utils.helpers import matches_any
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.layer_utils import _NNCFModuleMixin
from contextlib import contextmanager


def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(prefix=prefix, cls=module.__class__.__name__, name=module_name)


def get_all_modules(model, prefix=None):
    found = OrderedDict()
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        found[full_node_name] = module
        sub_found = get_all_modules(module, prefix=full_node_name)
        if sub_found:
            found.update(sub_found)
    return found


def get_all_modules_by_type(model, module_types=None, current_scope=None,
                            ignored_scopes=None, target_scopes=None) -> Dict['Scope', Module]:
    if isinstance(module_types, str):
        module_types = [module_types]
    found = OrderedDict()
    from nncf.torch.dynamic_graph.scope import Scope
    from nncf.torch.dynamic_graph.scope import ScopeElement
    if current_scope is None:
        current_scope = Scope()
        current_scope.push(ScopeElement(model.__class__.__name__))
    for name, module in model.named_children():
        child_scope_element = ScopeElement(module.__class__.__name__, name)
        child_scope = current_scope.copy()
        child_scope.push(child_scope_element)

        if matches_any(str(child_scope), ignored_scopes):
            continue

        if target_scopes is None or matches_any(str(child_scope), target_scopes):
            if module_types is None or module_types.count(str(type(module).__name__)) != 0:
                found[child_scope] = module
            sub_found = get_all_modules_by_type(module, module_types,
                                                current_scope=child_scope,
                                                ignored_scopes=ignored_scopes,
                                                target_scopes=target_scopes)
            if sub_found:
                found.update(sub_found)
    return found


def get_state_dict_names_with_modules(model, str_types=None, prefix=''):
    found = OrderedDict()
    for name, module in model.named_children():
        full_node_name = "{}{}".format(prefix, name)
        if str_types is not None and type(module).__name__ in str_types:
            found[full_node_name] = module
        sub_found = get_state_dict_names_with_modules(module, str_types, prefix=full_node_name + '.')
        if sub_found:
            found.update(sub_found)
    return found


def get_filters_num(module):
    if isinstance(module, _NNCFModuleMixin):
        return module.weight.size(module.target_weight_dim_for_compression)
    return module.weight.size(0)


def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def is_tracing_state():
    # pylint: disable=protected-access
    return torch._C._get_tracing_state()


class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def sum_like(tensor_to_sum, ref_tensor):
    """Warning: may modify tensor_to_sum"""
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            if isinstance(tensor_to_sum, np.ndarray):
                tensor_to_sum = tensor_to_sum.sum(dim, keepdims=True)
            else:
                tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


def get_per_channel_scale_shape(input_shape, is_weights):
    scale_shape = [1 for _ in input_shape]
    if is_weights:
        scale_shape[0] = input_shape[0]  # Per weight channel scales
    else:
        scale_shape[1] = input_shape[1]  # Per activation channel scales

    return scale_shape


def get_scale_shape(input_shape: List[int], is_weights: bool, per_channel: bool) -> List[int]:
    if not per_channel:
        return [1]
    return get_per_channel_scale_shape(input_shape, is_weights)


def get_flat_tensor_contents_string(input_tensor):
    retval = "["
    for idx, el in enumerate(input_tensor.view(-1)):
        if idx >= 10:
            retval += "... (first 10/{} elements shown only) ".format(len(input_tensor.view(-1)))
            break
        retval += "{:.4f}, ".format(el.item())
    retval += "]"
    return retval


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def safe_thread_call(main_call_fn, after_barrier_call_fn=None):
    result = None
    if is_dist_avail_and_initialized():
        if is_main_process():
            result = main_call_fn()
        dist.barrier()
        if not is_main_process():
            result = after_barrier_call_fn() if after_barrier_call_fn else main_call_fn()
    else:
        result = main_call_fn()
    return result


string_types = (str, bytes)
iteritems = lambda mapping: getattr(mapping, 'iteritems', mapping.items)()


def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

def is_traced_tensor(obj):
    return isinstance(obj, TracedTensor)

def maybe_get_iterator(obj):
    it = None
        # pylint:disable=isinstance-second-argument-not-valid-type
    if isinstance(obj, Mapping):
        it = iteritems
        # pylint:disable=isinstance-second-argument-not-valid-type
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
        it = enumerate
    return it


def to_tuple(lst: List,
             named_tuple_class: Type = None,
             named_tuple_fields: List[str] = None) -> Tuple:
    # Able to produce namedtuples if a corresponding parameter is given
    if named_tuple_fields is None:
        return tuple(lst)
    return named_tuple_class(*lst)


def is_tuple(obj) -> bool:
    return isinstance(obj, tuple)


def is_named_tuple(obj) -> bool:
    return is_tuple(obj) and (obj.__class__ != tuple)


def objwalk(obj, unary_predicate: Callable[[Any], bool], apply_fn: Callable, memo=None):
    """
    Walks through the indexable container hierarchy of obj and replaces all sub-objects matching a criterion
    with the result of a given function application.
    """
    #pylint:disable=too-many-nested-blocks
    #pylint:disable=too-many-branches
    if memo is None:
        memo = set()

    named_tuple_class = None
    named_tuple_fields = None
    if is_named_tuple(obj):
        named_tuple_class = obj.__class__
        #pylint:disable=protected-access
        named_tuple_fields = obj._fields

    was_tuple = is_tuple(obj)
    if was_tuple:
        obj = list(obj)

    iterator = maybe_get_iterator(obj)

    if iterator is not None:
        if id(obj) not in memo:
            memo.add(id(obj))
            indices_to_apply_fn_to = set()
            indices_vs_named_tuple_data = {}  # type: Dict[Any, Tuple[list, Type, List[str]]]
            for idx, value in iterator(obj):
                next_level_it = maybe_get_iterator(value)
                if next_level_it is None:
                    if unary_predicate(value):
                        indices_to_apply_fn_to.add(idx)
                else:
                    if is_tuple(value):
                        processed_tuple = objwalk(value, unary_predicate, apply_fn, memo)
                        if is_named_tuple(value):
                            indices_vs_named_tuple_data[idx] = processed_tuple, value.__class__, value._fields
                        else:
                            indices_vs_named_tuple_data[idx] = processed_tuple, None, None
                    else:
                        objwalk(value, unary_predicate, apply_fn)
            for idx in indices_to_apply_fn_to:
                obj[idx] = apply_fn(obj[idx])
            for idx, tpl_data in indices_vs_named_tuple_data.items():
                tpl, n_tpl_class, n_tpl_fields = tpl_data
                obj[idx] = to_tuple(tpl, n_tpl_class, n_tpl_fields)

            memo.remove(id(obj))
    else:
        if unary_predicate(obj):
            return apply_fn(obj)

    if was_tuple:
        return to_tuple(obj, named_tuple_class, named_tuple_fields)

    return obj


class _ModuleState:
    def __init__(self, module: Module = None):
        self._training_state = {}
        self._requires_grad_state = {}
        if module is not None:
            for ch in module.modules():
                self.training_state[ch] = ch.training

            for p in module.parameters():
                self.requires_grad_state[p] = p.requires_grad

    @property
    def training_state(self) -> Dict[Module, bool]:
        return self._training_state

    @property
    def requires_grad_state(self) -> Dict[Parameter, bool]:
        return self._requires_grad_state


def save_module_state(module: Module) -> _ModuleState:
    return _ModuleState(module)


def load_module_state(module: Module, state: _ModuleState, strict=False) -> None:
    for ch in module.modules():
        try:
            ch.train(state.training_state[ch])
        except KeyError as err:
            # KeyError could happen if the modules name were changed during forward
            # (e.g. LSTM block in NNCF examples)
            nncf_logger.warning(err)
            if strict:
                nncf_logger.error(err)
                return

    for p in module.parameters():
        p.requires_grad = state.requires_grad_state[p]


@contextmanager
def training_mode_switcher(model: Module, is_training: bool = True):
    saved_state = save_module_state(model)
    model.train(is_training)
    try:
        yield
    finally:
        load_module_state(model, saved_state)


def compute_FLOPs_hook(module, input_, output, dict_to_save, module_node_name: NNCFNodeName):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d,
                           nn.ConvTranspose3d)):
        ks = module.weight.data.shape
        mac_count = np.prod(ks) * np.prod(output.shape[2:])
    elif isinstance(module, nn.Linear):
        if len(input_[0].shape) == 1:
            # In some test cases input tensor could have dimension [N]
            mac_count = input_[0].shape[0] * output.shape[-1]
        else:
            mac_count = np.prod(input_[0].shape[1:]) * output.shape[-1]
    else:
        return
    dict_to_save[module_node_name] = 2 * mac_count


def add_domain(name_operator: str) -> str:
    from nncf.torch.compression_method_api import DOMAIN_CUSTOM_OPS_NAME
    return DOMAIN_CUSTOM_OPS_NAME + "::" + name_operator
