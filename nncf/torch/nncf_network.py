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

import inspect
from collections import OrderedDict
from enum import Enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import functools
import torch
from copy import deepcopy
from torch import nn

from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.torch.debug import CombinedDebugInterface
from nncf.torch.debug import debuggable_forward
from nncf.common.utils.debug import is_debug
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph import ShapeIgnoringTensorMetaComparator
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.graph_tracer import PostGraphBuildActing
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import InputInfoWrapManager
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.patch_pytorch import ignore_scope
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.transform_graph import replace_modules_by_nncf_modules
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.graph_builder import GraphConverter
from nncf.torch.graph.operator_metatypes import SplitMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.knowledge_distillation.knowledge_distillation_handler import KnowledgeDistillationLossHandler
from nncf.torch.layers import NNCF_MODULES
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.utils import compute_FLOPs_hook
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.utils import get_state_dict_names_with_modules
from nncf.torch.nested_objects_traversal import objwalk

MODEL_WRAPPED_BY_NNCF_ATTR_NAME = 'nncf_module'
LEGACY_ACT_STORAGE_NAME = "activation_quantizers"
EXTERNAL_QUANTIZERS_STORAGE_NAME = "external_quantizers"

Module = TypeVar('Module', bound=nn.Module)


class ExtraCompressionModuleType(Enum):
    EXTERNAL_QUANTIZER = 0


class LoadStateListener:
    """
        Resets the initialization flags (`initialized`) for all quantization modules on `load_state_dict` call.
        These flags are used to update not loaded params (from checkpoint or model's state)
        on initialization stage of algorithm.
        Flags reset is required on each call of `load_state_dict`, because internal method (`build_graph`)
        restores model state by calling this method.
    """

    def __init__(self, model, all_quantizations):
        # pylint: disable=protected-access
        self.hook = model._register_load_state_dict_pre_hook(
            functools.partial(self.hook_fn, quantize_modules=all_quantizations.values()))

    def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                quantize_modules):
        for module in quantize_modules:
            module.initialized = False

    def close(self):
        self.hook.remove()


class PTInsertionType(OrderedEnum):
    NNCF_MODULE_PRE_OP = 0
    NNCF_MODULE_POST_OP = 1
    OPERATOR_PRE_HOOK = 2
    OPERATOR_POST_HOOK = 3


class PTInsertionPoint:
    TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT = {
        TargetType.PRE_LAYER_OPERATION: PTInsertionType.NNCF_MODULE_PRE_OP,
        TargetType.POST_LAYER_OPERATION: PTInsertionType.NNCF_MODULE_POST_OP,
        TargetType.OPERATION_WITH_WEIGHTS: PTInsertionType.NNCF_MODULE_PRE_OP,
        TargetType.OPERATOR_PRE_HOOK: PTInsertionType.OPERATOR_PRE_HOOK,
        TargetType.OPERATOR_POST_HOOK: PTInsertionType.OPERATOR_POST_HOOK
    }

    def _get_pt_insertion_type(self, target_type: TargetType) -> PTInsertionType:
        if target_type not in PTInsertionPoint.TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT:
            raise RuntimeError("Unsupported target type for PyTorch: {}".format(target_type))
        return PTInsertionPoint.TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT[target_type]

    def __init__(self, target_type: TargetType, op_address: OperationAddress,
                 input_port_id: int = None):
        self.insertion_type = self._get_pt_insertion_type(target_type)
        self.op_address = op_address
        self.module_scope = op_address.scope_in_model
        self.input_port_id = input_port_id

    def __eq__(self, other: 'PTInsertionPoint'):
        return self.insertion_type == other.insertion_type and \
               self.op_address == other.op_address and \
               self.module_scope == other.module_scope and \
               self.input_port_id == other.input_port_id

    def __str__(self):
        return ' '.join([str(v) for v in self.__dict__.values()])

    def __hash__(self):
        return hash(str(self))

# pylint: disable=too-many-public-methods


@ignore_scope
class NNCFNetwork(nn.Module, PostGraphBuildActing):
    MODEL_STATE_VERSION_ATTR = '_nncf_model_state_version'
    MODEL_STATE_VERSION = 1

    def __init__(self, module, input_infos: List[ModelInputInfo],
                 dummy_forward_fn=None, wrap_inputs_fn=None, scopes_without_shape_matching=None,
                 ignored_scopes=None, target_scopes=None, reset: bool = False, wrap_outputs_fn=None,
                 original_model_accuracy=None):
        super().__init__()
        self._set_nncf_wrapped_model(module)
        self._forward_signature = inspect.signature(module.forward)
        self.input_infos = input_infos

        self._original_model_accuracy = original_model_accuracy

        self.ignored_scopes = ignored_scopes
        self.target_scopes = target_scopes
        self._user_dummy_forward_fn = dummy_forward_fn
        self._kd_loss_handler = None

        try:
            device = next(module.parameters()).device
        except StopIteration:
            # Param-less model, assume CPU
            device = 'cpu'

        if wrap_inputs_fn is not None:
            self._wrap_inputs_fn = wrap_inputs_fn
        else:
            self.__input_infos_based_input_wrapper = InputInfoWrapManager(self.input_infos,
                                                                          self._forward_signature,
                                                                          module_ref_for_device=self)
            self._wrap_inputs_fn = self.__input_infos_based_input_wrapper.wrap_inputs

        if wrap_outputs_fn is not None:
            self._wrap_outputs_fn = wrap_outputs_fn
        else:
            self._wrap_outputs_fn = wrap_nncf_model_outputs_with_objwalk

        self._nncf_module_scopes = []  # type: List[Scope]
        self.scopes_without_shape_matching = scopes_without_shape_matching
        self.debug_interface = CombinedDebugInterface() if is_debug() else None
        self._extra_module_types = []  # type: List[ExtraCompressionModuleType]
        # pylint:disable=line-too-long
        self._insertions_into_original_graph = {}  # type: Dict[PTTargetPoint, List[Tuple[Callable, TransformationPriority]]]

        _orig_graph_build_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=True,
                                                                                     with_output_tracing=True)

        nncf_wrapped_model = self.get_nncf_wrapped_model()
        eval_only_op_scopes = self._collect_eval_only_op_scopes(nncf_wrapped_model,
                                                                _orig_graph_build_forward_fn)

        # all modules called in eval mode should be replaced prior to graph building
        self._replace_modules_by_nncf_modules(device, eval_only_op_scopes, reset)

        _orig_context = TracingContext()

        _orig_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        _orig_context.add_node_comparators([MODEL_OUTPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self.scopes_without_shape_matching:
            _orig_context.add_node_comparators(scopes_without_shape_matching,
                                               ShapeIgnoringTensorMetaComparator())

        self._original_dynamic_graph = GraphTracer(_orig_graph_build_forward_fn).trace_graph(nncf_wrapped_model,
                                                                                             _orig_context,
                                                                                             as_eval=True)
        self._original_graph = GraphConverter.convert(self._original_dynamic_graph,
                                                      input_infos=self.input_infos)
        self._compressed_graph = None  # type: PTNNCFGraph

        self._compressed_context = TracingContext()

        self._dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=False,
                                                                               with_output_tracing=False)
        self._in_user_dummy_forward = False

        self._compressed_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        self._compressed_context.add_node_comparators([MODEL_OUTPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self.scopes_without_shape_matching:
            self._compressed_context.add_node_comparators(scopes_without_shape_matching,
                                                          ShapeIgnoringTensorMetaComparator())
        self._load_listener = None

    @debuggable_forward
    def forward(self, *args, **kwargs):
        with self._compressed_context as ctx:  # type: TracingContext
            ctx.base_module_thread_local_replica = self
            args, kwargs = replicate_same_tensors((args, kwargs))
            if not self._in_user_dummy_forward:
                # If a user supplies own dummy forward, he is responsible for
                # correctly wrapping inputs inside it as well.
                args, kwargs = self._strip_traced_tensors(args, kwargs)
                args, kwargs = self._wrap_inputs_fn(args, kwargs)
            retval = self.get_nncf_wrapped_model()(*args, **kwargs)
            retval = replicate_same_tensors(retval)
            if not self._in_user_dummy_forward:
                retval = self._wrap_outputs_fn(retval)

        if self._kd_loss_handler is not None and self.get_nncf_wrapped_model().training:
            self._kd_loss_handler(retval, *args, **kwargs)
        return retval

    def _strip_traced_tensors(self, args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
        """
            Required to guard against new forward calls on tensors that have already passed
            through NNCF's forward once and got turned into TracedTensors by reference access.
        """
        is_traced_tensor_predicate = lambda x: isinstance(x, TracedTensor)

        def strip_fn(tensor: TracedTensor) -> torch.Tensor:
            if hasattr(torch.Tensor, 'as_subclass'):
                return torch.Tensor.as_subclass(tensor, torch.Tensor)
            # Torch < 1.7.0 fallback
            return torch.tensor(tensor, device=tensor.device, requires_grad=tensor.requires_grad)

        args = objwalk(args, is_traced_tensor_predicate, strip_fn)
        kwargs = objwalk(kwargs, is_traced_tensor_predicate, strip_fn)
        return args, kwargs

    def create_knowledge_distillation_loss_handler(self, kd_original_model: nn.Module, calculate_fn)\
            -> KnowledgeDistillationLossHandler:
        """
        Creates KnowledgeDistillationLossHandler instance for enabling Knowledge Distillation feature.
            Also returns created KnowledgeDistillationLossHandler for control over Knowledge Distillation logic.

        :param kd_original_model: original non compressed model used for distillation
        :param calculate_fn: function used to parse model outputs and calculate knowledge distillation loss
        :return: KnowledgeDistillationLossHandler instance
        """
        device = next(self.get_nncf_wrapped_model().parameters()).device
        self._kd_loss_handler = KnowledgeDistillationLossHandler(self._compressed_context,
                                                                 kd_original_model,
                                                                 calculate_fn,
                                                                 device)
        return self._kd_loss_handler

    # Cannnot use property syntax here, otherwise the wrapped module will end up
    # being twice in the same checkpoint with different prefixes
    def get_nncf_wrapped_model(self):
        return getattr(self, MODEL_WRAPPED_BY_NNCF_ATTR_NAME)

    def _set_nncf_wrapped_model(self, value):
        setattr(self, MODEL_WRAPPED_BY_NNCF_ATTR_NAME, value)

    def get_clean_shallow_copy(self) -> 'NNCFNetwork':
        # WARNING: Will reset pre- and post-ops of the underlying model. Use save_nncf_module_additions
        # and load_nncf_module_additions to preserve these, or temporary_clean_view().
        from nncf.torch.utils import save_module_state, load_module_state
        saved_state = save_module_state(self)
        model_copy = NNCFNetwork(self.get_nncf_wrapped_model(), self.input_infos,
                                 self._user_dummy_forward_fn, self._wrap_inputs_fn,
                                 self.scopes_without_shape_matching, self.ignored_scopes, self.target_scopes,
                                 reset=True)
        load_module_state(model_copy, saved_state)
        return model_copy

    def get_modules_in_nncf_modules_by_type(self, types) -> Dict[Scope, nn.Module]:
        nncf_modules = self.get_nncf_modules()
        retval = {}
        for nncf_module_scope, nncf_module in nncf_modules.items():
            nncf_module_scope.pop()
            for relative_scope, target_module in get_all_modules_by_type(nncf_module, types).items():
                retval[nncf_module_scope + relative_scope] = target_module
        return retval

    def insert_at_point(self, point: PTInsertionPoint, fn_list: List[Callable]):
        if point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            self._compressed_context.register_pre_hooks(fn_list, point.op_address, point.input_port_id)
        elif point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            self._compressed_context.register_post_hooks(fn_list, point.op_address)
        elif point.insertion_type in [PTInsertionType.NNCF_MODULE_PRE_OP,
                                      PTInsertionType.NNCF_MODULE_POST_OP]:
            norm_target_scope = self._normalize_variable_recurrent_scope(point.module_scope)
            norm_nncf_scopes = [self._normalize_variable_recurrent_scope(x) for x in self._nncf_module_scopes]
            assert norm_target_scope in norm_nncf_scopes  # Required for proper Recurrent/VariableRecurrent addressing
            nncf_module = self.get_module_by_scope(point.module_scope)
            if point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
                for fn in fn_list:
                    nncf_module.register_pre_forward_operation(fn)
            elif point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
                for fn in fn_list:
                    nncf_module.register_post_forward_operation(fn)
        else:
            raise RuntimeError("Unsupported insertion type: {}".format(point.insertion_type))

    def __getattr__(self, name):
        wrapped_module = super().__getattr__(MODEL_WRAPPED_BY_NNCF_ATTR_NAME)
        if hasattr(wrapped_module, name):
            return getattr(wrapped_module, name)
        return super().__getattr__(name)

    def get_graph(self) -> PTNNCFGraph:
        if self._compressed_context.graph.get_nodes_count() == 0 or self._compressed_graph is None:
            self.rebuild_graph()
        return self._compressed_graph

    def get_dynamic_graph(self) -> DynamicGraph:
        return self._compressed_context.graph

    def get_original_graph(self) -> PTNNCFGraph:
        return self._original_graph

    def get_tracing_context(self) -> TracingContext:
        return self._compressed_context

    def enable_dynamic_graph_building(self):
        self._compressed_context.enable_node_additions()

    def disable_dynamic_graph_building(self):
        self._compressed_context.disable_node_additions()

    def _get_dummy_forward_fn_for_graph_building(self, with_input_tracing, with_output_tracing):
        if self._user_dummy_forward_fn is None:
            return create_dummy_forward_fn(self.input_infos,
                                           with_input_tracing=with_input_tracing,
                                           wrap_inputs_fn=self._wrap_inputs_fn,
                                           wrap_outputs_fn=self._wrap_outputs_fn,
                                           with_output_tracing=with_output_tracing)

        def wrapped_user_dummy_forward_fn(*args, **kwargs):
            self._in_user_dummy_forward = True
            retval = self._user_dummy_forward_fn(*args, **kwargs)
            self._in_user_dummy_forward = False
            return retval

        return wrapped_user_dummy_forward_fn

    def _replace_modules_by_nncf_modules(self, device, eval_only_op_scopes: List[Scope] = None,
                                         reset: bool = False):
        module, self._nncf_module_scopes = replace_modules_by_nncf_modules(
            self.get_nncf_wrapped_model(), ignored_scopes=self.ignored_scopes,
            target_scopes=self.target_scopes, eval_op_scopes=eval_only_op_scopes,
            reset=reset)
        self._set_nncf_wrapped_model(module.to(device))

    def get_nncf_module_scopes(self) -> List[Scope]:
        return self._nncf_module_scopes

    def get_nncf_modules(self) -> Dict[Scope, torch.nn.Module]:
        nncf_module_names_list = NNCF_MODULES + [x.__name__ for x in NNCF_WRAPPED_USER_MODULES_DICT.values()]
        return get_all_modules_by_type(self.get_nncf_wrapped_model(), nncf_module_names_list)

    def get_weighted_original_graph_nodes(self, nncf_module_names: List[str] = None) -> List[NNCFNode]:
        retval = []
        for nncf_module_scope in self._nncf_module_scopes:
            if nncf_module_names is not None:
                module_name = nncf_module_scope[-1].calling_module_class_name
                if module_name not in nncf_module_names:
                    continue
            nodes_in_scope = self._original_graph.get_op_nodes_in_scope(nncf_module_scope)
            for node in nodes_in_scope:
                if node.layer_attributes is not None:  # TODO(vshampor): implement more explicit filtering
                    retval.append(node)
        return retval

    def get_nncf_modules_by_module_names(self, nncf_module_names_list: List[str]) -> Dict["Scope", torch.nn.Module]:
        return get_all_modules_by_type(self.get_nncf_wrapped_model(), nncf_module_names_list)

    def rebuild_graph(self, *input_args):
        self._compressed_context.reset_graph()
        dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(with_input_tracing=False,
                                                                         with_output_tracing=False)
        builder = GraphBuilder(dummy_forward_fn)
        self._compressed_graph = builder.build_graph(self, self._compressed_context,
                                                     input_infos=self.input_infos)

    def post_build_graph_actions(self):
        # Reset initialization flags (`initialized`) for all quantization modules
        # after dummy `load_state_dict` call.
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        all_quantizations = get_state_dict_names_with_modules(self, quantization_types)
        for module in all_quantizations.values():
            module.initialized = False

    def is_scope_in_nncf_module_scope(self, scope: Scope):
        # TODO: optimize
        norm_nncf_scopes = [self._normalize_variable_recurrent_scope(x) for x in self._nncf_module_scopes]
        norm_op_scope = self._normalize_variable_recurrent_scope(scope)
        for nncf_scope in norm_nncf_scopes:
            if norm_op_scope in nncf_scope:
                return True
        return False

    def register_compression_module_type(self, compression_module_type: ExtraCompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type in self._extra_module_types:
            raise RuntimeError("Module type {} is already registered".format(compression_module_type))
        self.__setattr__(attr_name, nn.ModuleDict())
        self._extra_module_types.append(compression_module_type)

    def add_compression_module(self, module_key: str, module: nn.Module,
                               compression_module_type: ExtraCompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        storage = self.__getattr__(attr_name)
        if module_key in storage:
            raise RuntimeError("Module {} is already registered under {}".format(module_key, attr_name))
        storage[module_key] = module

    def get_compression_modules_by_type(self, compression_module_type: ExtraCompressionModuleType) -> nn.ModuleDict:
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        return self.__getattr__(attr_name)

    @staticmethod
    def _compression_module_type_to_attr_name(compression_module_type: ExtraCompressionModuleType):
        """
        Required for backward compatibility with checkpoints that store function and activation
        quantizers directly under corresponding attributes of NNCFNetwork.
        """
        if compression_module_type == ExtraCompressionModuleType.EXTERNAL_QUANTIZER:
            return EXTERNAL_QUANTIZERS_STORAGE_NAME
        raise RuntimeError("Unknown extra module type")

    def sort_compression_modules(self, compression_module_type: ExtraCompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError("Module type {} was not registered".format(compression_module_type))
        module_dict = self.__getattr__(attr_name)
        # pylint: disable=protected-access
        module_dict._modules = OrderedDict(sorted(module_dict._modules.items()))
        self.__setattr__(attr_name, module_dict)

    @staticmethod
    def _normalize_variable_recurrent_scope(scope: Scope):
        """
        Two scopes pointing to an NNCF module that only differ in a Recurrent/VariableRecurrent/VariableRecurrentReverse
        scope node actually point to one and the same module.
        """
        ret_scope = scope.copy()
        for scope_element in ret_scope:
            if scope_element.calling_module_class_name in ["Recurrent", "VariableRecurrent",
                                                           "VariableRecurrentReverse"]:
                scope_element.calling_module_class_name = "NormalizedName_Recurrent"
        return ret_scope

    def do_dummy_forward(self, force_eval=False):
        """
        Attention: If run with force_eval=False, this may spoil the batchnorm statistics,
        and an eval run of the model will perform much worse than the train run.
        """
        if force_eval:
            train_mode = self.training
            self.eval()
        with torch.no_grad():
            with self._compressed_context as ctx:
                ctx.base_module_thread_local_replica = self
                self._dummy_forward_fn(self)
        if force_eval:
            if train_mode:
                self.train()

    def get_insertion_point_graph(self) -> InsertionPointGraph:
        # Set up a pre- and post-hooks on almost every op in PyTorch
        nncf_graph = self.get_original_graph()
        pre_hooks = []  # type: List[PreHookInsertionPoint]
        post_hooks = []  # type: List[PostHookInsertionPoint]
        for node in nncf_graph.get_all_nodes():
            # Pre-hook insertion point nodes
            # Will insert a pre-hook IP for each input edge. The input edge must be marked with
            # a port ID attribute.
            in_edges = nncf_graph.get_input_edges(node)
            for edge in in_edges:
                port_id = edge.input_port_id
                pre_hook_ip = PreHookInsertionPoint(target_node_name=node.node_name,
                                                    input_port_id=port_id)
                pre_hooks.append(pre_hook_ip)

            if issubclass(node.metatype, SplitMetatype):
                # chunk returns a tuple of tensors, which can only be handled in NNCF
                # once post-hook ports are enabled. Work around it for now by disallowing post-hook
                # insertion for chunks
                # TODO: enable post-hook ports and remove this
                continue

            # Post-hook insertion point nodes
            post_hook_ip = PostHookInsertionPoint(node.node_name)
            post_hooks.append(post_hook_ip)

        weighted_nodes = self.get_weighted_original_graph_nodes()
        weighted_node_names = [weighted_node.node_name for weighted_node in weighted_nodes]

        ip_graph = InsertionPointGraph(self._original_graph, weight_modifiable_node_names=weighted_node_names,
                                       allowed_pre_hook_insertion_points=pre_hooks,
                                       allowed_post_hook_insertion_points=post_hooks)
        return ip_graph

    def get_module_by_scope(self, scope: Scope) -> Optional[torch.nn.Module]:
        curr_module = self.get_nncf_wrapped_model()
        for scope_element in scope[1:]:  # omit first scope element which corresponds to base module
            if scope_element.calling_field_name is None:
                # The module used is being created in-place every time and never stored in the model,
                # happens for nn.Softmax in BERT implementations.
                return None
            # pylint: disable=protected-access
            next_module = curr_module._modules.get(scope_element.calling_field_name)
            if next_module is None:
                raise RuntimeError("Could not find a {} module member in {} module of scope {} during node search"
                                   .format(scope_element.calling_field_name,
                                           scope_element.calling_module_class_name,
                                           str(scope)))
            curr_module = next_module
        return curr_module

    def get_containing_module(self, node_name: NNCFNodeName) -> torch.nn.Module:
        if self._compressed_graph is not None:
            try:
                scope = self._compressed_graph.get_scope_by_node_name(node_name)
            except RuntimeError:
                nncf_logger.debug("Node {} not found in compressed graph when trying to determine containing module, "
                                  "trying the original graph to see if the node was present there "
                                  "during graph building")
                scope = self._original_graph.get_scope_by_node_name(node_name)
        else:
            scope = self._original_graph.get_scope_by_node_name(node_name)
        return self.get_module_by_scope(scope)

    def get_parameters_count_in_model(self):
        """
        Return total amount of model parameters.
        """
        count = 0
        for param in self.parameters():
            count = count + param.numel()
        return count

    def get_flops_per_module(self) -> Dict[NNCFNodeName, int]:
        """
        Calculates FLOPS count for modules.
        """
        model = self
        flops_count_dict = {}

        def get_hook(name):
            return functools.partial(compute_FLOPs_hook, dict_to_save=flops_count_dict,
                                     module_node_name=name)

        hook_list = []
        for nncf_node in self._original_graph.get_all_nodes():
            node_module = self.get_containing_module(nncf_node.node_name)
            hook_list.append(node_module.register_forward_hook(get_hook(nncf_node.node_name)))
        model.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()
        return flops_count_dict

    def get_MACs_in_model(self):
        """
            Calculates MAC units count for model.
        """
        flops_count_dict = self.get_flops_per_module()
        total_MACs_count = sum(v // 2 for v in flops_count_dict.values())
        return total_MACs_count

    def get_input_infos(self) -> List[ModelInputInfo]:
        return deepcopy(self.input_infos)

    def save_nncf_module_additions(self) -> Dict[Scope, Tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]]:
        retval = {}
        for module_scope, nncf_module in self.get_nncf_modules().items():
            retval[module_scope] = (deepcopy(nncf_module.pre_ops), deepcopy(nncf_module.post_ops))
        return retval

    def load_nncf_module_additions(self,
                                   scope_vs_pre_post_ops_dict: Dict[Scope, Tuple[torch.nn.ModuleDict,
                                                                                   torch.nn.ModuleDict]]):
        for module_scope, nncf_module in self.get_nncf_modules().items():
            nncf_module.pre_ops = scope_vs_pre_post_ops_dict[module_scope][0]
            nncf_module.post_ops = scope_vs_pre_post_ops_dict[module_scope][1]

    def temporary_clean_view(self):
        class Mgr:
            def __init__(self, model: NNCFNetwork):
                self.model = model
                self.storage_dict = {}

            def __enter__(self):
                self.storage_dict = self.model.save_nncf_module_additions()
                clean_model = self.model.get_clean_shallow_copy()
                return clean_model

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model.load_nncf_module_additions(self.storage_dict)

        return Mgr(self)

    def _collect_eval_only_op_scopes(self, model: nn.Module, dummy_forward_fn: Callable) -> List[Scope]:
        """
        Returns scopes of the modules which are executed in evaluation mode only.
        """

        tracer = GraphTracer(dummy_forward_fn)
        result = []
        eval_graph = tracer.trace_graph(model, as_eval=True)
        for dyn_graph_node in eval_graph.get_all_nodes():
            result.append(dyn_graph_node.op_exec_context.scope_in_model)
        return result

    @property
    def original_model_accuracy(self):
        return self._original_model_accuracy

    def get_node_to_op_address_mapping(self) -> Dict[NNCFNodeName, OperationAddress]:
        # The IDs of corresponding nodes of the original dynamic graph and original NNCF graph
        # must be equal for this to work.
        retval = {}
        for node in self._original_dynamic_graph.get_all_nodes():
            node_id = node.node_id
            op_address = node.op_exec_context.op_address
            nncf_node = self._original_graph.get_node_by_id(node_id)
            retval[nncf_node.node_name] = op_address
        return retval


class PTModelTransformer(ModelTransformer):
    def __init__(self, model: NNCFNetwork):
        super().__init__(model)
        self._node_to_op_address_mapping = model.get_node_to_op_address_mapping()

    def transform(self, transformation_layout: PTTransformationLayout) -> NNCFNetwork:
        fns_grouped_by_points = {}  # type: Dict[PTInsertionPoint, List[Tuple[Callable, TransformationPriority]]]
        for transformation_command in transformation_layout.transformations:  # type: PTInsertionCommand
            target_point = transformation_command.target_point  # type: PTTargetPoint
            target_node_name = target_point.target_node_name
            pt_ip = PTInsertionPoint(target_type=target_point.target_type,
                                     op_address=self._node_to_op_address_mapping[target_node_name],
                                     input_port_id=target_point.input_port_id)
            fn = transformation_command.fn
            if target_point.type is TargetType.OPERATION_WITH_WEIGHTS:
                fn = UpdateWeight(fn)
            tup = (fn, transformation_command.priority)
            if pt_ip not in fns_grouped_by_points:
                fns_grouped_by_points[pt_ip] = [tup]
            else:
                fns_grouped_by_points[pt_ip].append(tup)

        for pt_ip, fn_list_with_priority in fns_grouped_by_points.items():
            fn_list_with_priority = sorted(fn_list_with_priority, key=lambda x: x[1])
            self._model.insert_at_point(pt_ip, [x[0] for x in fn_list_with_priority])
        return self._model
