# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-lines
import functools
import inspect
import types
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from enum import IntEnum
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

import torch
from torch import nn

from nncf import nncf_logger
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.utils.debug import is_debug
from nncf.torch.debug import CombinedDebugInterface
from nncf.torch.debug import debuggable_forward
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.dynamic_graph.graph import DynamicGraph
from nncf.torch.dynamic_graph.graph import ShapeIgnoringTensorMetaComparator
from nncf.torch.dynamic_graph.graph_tracer import GraphTracer
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import InputInfoWrapManager
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.patch_pytorch import ORIGINAL_CALL
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.dynamic_graph.scope_access import get_module_by_scope
from nncf.torch.dynamic_graph.trace_tensor import TracedTensor
from nncf.torch.dynamic_graph.wrappers import wrap_module_call
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.graph_builder import GraphConverter
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_WEIGHTS_METATYPES
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.knowledge_distillation.knowledge_distillation_handler import KnowledgeDistillationLossHandler
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules
from nncf.torch.utils import compute_FLOPs_hook
from nncf.torch.utils import get_all_modules_by_type
from nncf.torch.utils import get_model_device
from nncf.torch.utils import training_mode_switcher

LEGACY_MODEL_WRAPPED_BY_NNCF_ATTR_NAME = "nncf_module"
LEGACY_EXTERNAL_QUANTIZERS_STORAGE_PREFIX = "external_quantizers"

EXTERNAL_QUANTIZERS_STORAGE_NAME = "external_quantizers"
CURRENT_EXTERNAL_QUANTIZERS_STORAGE_PREFIX = "_nncf." + EXTERNAL_QUANTIZERS_STORAGE_NAME

Module = TypeVar("Module", bound=nn.Module)


class PTInsertionType(IntEnum):
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
        TargetType.OPERATOR_POST_HOOK: PTInsertionType.OPERATOR_POST_HOOK,
    }

    def _get_pt_insertion_type(self, target_type: TargetType) -> PTInsertionType:
        if (
            not isinstance(target_type, TargetType)
            or target_type not in PTInsertionPoint.TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT
        ):
            raise RuntimeError("Unsupported target type for PyTorch: {}".format(target_type))
        return PTInsertionPoint.TARGET_TYPE_VS_PT_INSERTION_TYPE_DICT[target_type]

    def __init__(self, target_type: TargetType, op_address: OperationAddress, input_port_id: int = None):
        self.insertion_type = self._get_pt_insertion_type(target_type)
        self.op_address = op_address
        self.module_scope = op_address.scope_in_model
        self.input_port_id = input_port_id

    def __eq__(self, other: "PTInsertionPoint"):
        return (
            self.insertion_type == other.insertion_type
            and self.op_address == other.op_address
            and self.module_scope == other.module_scope
            and self.input_port_id == other.input_port_id
        )

    def __str__(self):
        return " ".join([str(v) for v in self.__dict__.values()])

    def __hash__(self):
        return hash(str(self))


class ExtraCompressionModuleType(Enum):
    EXTERNAL_QUANTIZER = 0


class NNCFNetworkInterface(torch.nn.Module):
    """
    The single object that is added to the original model object as an attribute to provide a namespace for
    NNCF-specific method calls and a torch.nn.Module-like storage for compression parameters. Since this is a
    Module stored in a Module, all trainable parameters of the NNCFInterface will be registered for optimization
    in the same manner as the original model parameters, and will also be eligible for state_dict-powered persistence
    when saving/loading checkpoints
    """

    # pylint:disable=too-many-public-methods
    MODEL_STATE_VERSION_ATTR = "_nncf_model_state_version"
    MODEL_STATE_VERSION = 1

    def forward(self):
        """
        The module only serves a storage and namespacing purpose, forward functionality is not implemented.
        """
        raise NotImplementedError("Calling `forward` on NNCFInterface is prohibited.")

    def get_original_forward(self) -> Callable:
        """
        Returns the forward function of the original model, unmodified by NNCF. The returned function will
        have its 0-th implicit `self` argument bound to the model object.
        """
        if self._original_instance_forward is not None:
            return functools.partial(self._original_instance_forward, self._model_ref)
        return functools.partial(self._original_unbound_forward, self._model_ref)

    @contextmanager
    def temporary_bound_original_forward(self, bound_forward: Callable):
        """
        Context manager for temporary replacement of the underlying original model forward function. NNCF
        works by doing additional operations before and after the original object's forward call, and this context
        manager allows to temporarily run the compressed model object as if it had another original forward method.
        The signature of the new forward method must be the same w.r.t. the original forward method in terms of
        activation tensors.
        :param bound_forward: A callable which will be used to temporary replace the original forward call. Must be
        a bound method, e.g. the `self` argument had already been set to the same model object where the forward call
        must be replaced.
        """
        prev_bound_forward = self._bound_original_forward
        self._bound_original_forward = bound_forward
        yield
        self._bound_original_forward = prev_bound_forward

    def get_original_unbound_forward(self) -> Callable:
        """
        Returns the forward function of the original model, unmodified by NNCF. The returned function will preserve
        its 0-th implicit `self` argument without binding it to the model object.
        """
        return self._original_unbound_forward

    def set_original_unbound_forward(self, fwd_fn: Callable):
        """
        Allows to set the function that is treated by NNCF as "original" model forward to another function.
        :param fwd_fn: The new original forward function. The signature w.r.t. activation tensors must be the same,
        and the function must leave its 0-th `self` argument unbound.
        """
        self._custom_original_unbound_forward = fwd_fn

    def reset_original_unbound_forward(self):
        """
        Reset the forward which was set with set_original_unbound_forward() method.
        After this NNCF will fall back to the unbound forward of the original model.
        """
        self._custom_original_unbound_forward = None

    def __init__(
        self,
        model: torch.nn.Module,
        input_infos: List[ModelInputInfo] = None,
        dummy_forward_fn: Callable = None,
        wrap_inputs_fn: Callable = None,
        scopes_without_shape_matching: List[str] = None,
        ignored_scopes: List[str] = None,
        target_scopes: List[str] = None,
        wrap_outputs_fn: Callable = None,
    ):
        super().__init__()

        # Need this in order not to register owning module as sub-module of NNCFInterface and thus
        # avoid circular references
        object.__setattr__(self, "__model_ref", model)

        if isinstance(model, NNCFNetwork):
            # Got an NNCFNetwork already, probably during shallow copying.
            self._original_class = model.nncf._original_class
            self._bound_original_forward = model.nncf._bound_original_forward
            self._custom_original_unbound_forward = model.nncf._custom_original_unbound_forward
            self._original_instance_forward = model.nncf._original_instance_forward
        else:
            self._original_class = model.__class__
            self._bound_original_forward = None
            self._custom_original_unbound_forward = None
            self._original_instance_forward = model.__dict__.get("forward")

        self._forward_signature = inspect.signature(self.get_original_forward())
        self._input_infos = input_infos

        self._ignored_scopes = ignored_scopes
        self._target_scopes = target_scopes
        self._user_dummy_forward_fn = dummy_forward_fn
        self._kd_loss_handler = None

        device = get_model_device(model)

        if wrap_inputs_fn is not None:
            self._wrap_inputs_fn = wrap_inputs_fn
        elif self._input_infos is not None:
            self.__input_infos_based_input_wrapper = InputInfoWrapManager(
                self._input_infos, self._forward_signature, module_ref_for_device=model
            )
            self._wrap_inputs_fn = self.__input_infos_based_input_wrapper.wrap_inputs
        else:
            raise ValueError("wrap_inputs_fn or input_infos should be passed.")

        if wrap_outputs_fn is not None:
            self._wrap_outputs_fn = wrap_outputs_fn
        else:
            self._wrap_outputs_fn = wrap_nncf_model_outputs_with_objwalk

        self._nncf_replaced_modules = {}  # type: Dict[torch.nn.Module, List[Scope]]
        self._scopes_without_shape_matching = scopes_without_shape_matching
        self.debug_interface = CombinedDebugInterface() if is_debug() else None
        self._extra_module_types = []  # type: List[ExtraCompressionModuleType]
        self._insertions_into_original_graph = (
            {}
        )  # type: Dict[PTTargetPoint, List[Tuple[Callable, TransformationPriority]]]

        _orig_graph_build_forward_fn = self._get_dummy_forward_fn_for_graph_building(
            with_input_tracing=True, with_output_tracing=True
        )

        eval_op_scopes = self._collect_eval_op_scopes(model, _orig_graph_build_forward_fn)

        # all modules called in eval mode should be replaced prior to graph building
        self._replace_modules_by_nncf_modules(model, device, eval_op_scopes)

        _orig_context = TracingContext()

        _orig_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        _orig_context.add_node_comparators([MODEL_OUTPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self._scopes_without_shape_matching:
            _orig_context.add_node_comparators(scopes_without_shape_matching, ShapeIgnoringTensorMetaComparator())

        if isinstance(model, NNCFNetwork):
            self._original_dynamic_graph = model.nncf._original_dynamic_graph
            self._original_graph = model.nncf._original_graph
        else:
            self._original_dynamic_graph = GraphTracer(_orig_graph_build_forward_fn).trace_graph(
                model, _orig_context, as_eval=True
            )
            self._original_graph = GraphConverter.convert(self._original_dynamic_graph, input_infos=self._input_infos)
        self._compressed_graph = None  # type: PTNNCFGraph

        self._compressed_context = TracingContext()

        self._dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(
            with_input_tracing=False, with_output_tracing=False
        )
        self._in_user_dummy_forward = False

        self._compressed_context.add_node_comparators([MODEL_INPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        self._compressed_context.add_node_comparators([MODEL_OUTPUT_OP_NAME], ShapeIgnoringTensorMetaComparator())
        if self._scopes_without_shape_matching:
            self._compressed_context.add_node_comparators(
                scopes_without_shape_matching, ShapeIgnoringTensorMetaComparator()
            )
        self._load_listener = None

        self.compression_controller = None  # type: PTCompressionAlgorithmController

    @property
    def _original_unbound_forward(self):
        # Notes:
        # (1) We rely on an "unbound" forward which is the version of the method that has the
        #   `self` parameter not set, otherwise we will be indirectly capturing a reference to the
        #   model object in NNCFInterface - this will lead to failures in DataParallel
        #   because the bound original forward call during NNCFNetwork.forward
        #   would then call forward on the original non-replica module even if NNCFNetwork itself was
        #   replicated.
        # (2) We access the unbound forward from a reference to the original model class instead
        #   of storing the reference to the unbound forward itself because the original class forward
        #   may be overridden by some 3rd party logic. For example, during export of mm-based models to ONNX
        #   using mmdeploy library, the original forward method of the model is temporarily replaced
        #   during export. Moreover, in such case the forward signature needs to be hidden by a user
        #   beforehand by wrapping it with a function with (*args, **kwargs) as its arguments.
        custom_unbound_forward = self._custom_original_unbound_forward
        return self._original_class.forward if custom_unbound_forward is None else custom_unbound_forward

    @property
    def _model_ref(self) -> "NNCFNetwork":
        return object.__getattribute__(self, "__model_ref")

    @property
    def input_infos(self) -> List[ModelInputInfo]:
        return deepcopy(self._input_infos)

    def _strip_traced_tensors(self, args: Tuple, kwargs: Dict) -> Tuple[Tuple, Dict]:
        """
        Required to guard against new forward calls on tensors that have already passed
        through NNCF's forward once and got turned into TracedTensors by reference access.
        """
        is_traced_tensor_predicate = lambda x: isinstance(x, TracedTensor)

        def strip_fn(tensor: TracedTensor) -> torch.Tensor:
            if hasattr(torch.Tensor, "as_subclass"):
                return torch.Tensor.as_subclass(tensor, torch.Tensor)
            # Torch < 1.7.0 fallback
            return torch.tensor(tensor, device=tensor.device, requires_grad=tensor.requires_grad)

        args = objwalk(args, is_traced_tensor_predicate, strip_fn)
        kwargs = objwalk(kwargs, is_traced_tensor_predicate, strip_fn)
        return args, kwargs

    def create_knowledge_distillation_loss_handler(
        self, kd_original_model: nn.Module, calculate_fn
    ) -> KnowledgeDistillationLossHandler:
        """
        Creates KnowledgeDistillationLossHandler instance for enabling Knowledge Distillation feature.
            Also returns created KnowledgeDistillationLossHandler for control over Knowledge Distillation logic.

        :param kd_original_model: original non compressed model used for distillation
        :param calculate_fn: function used to parse model outputs and calculate knowledge distillation loss
        :return: KnowledgeDistillationLossHandler instance
        """
        device = get_model_device(self._model_ref)
        self._kd_loss_handler = KnowledgeDistillationLossHandler(
            self._compressed_context, kd_original_model, calculate_fn, device
        )
        return self._kd_loss_handler

    def reset_nncf_modules(self):
        for scope_list in self.get_nncf_module_scopes():
            # Can pick any access scope since they all should
            # point to the same object
            some_scope = scope_list[0]
            module = self.get_module_by_scope(some_scope)
            module.reset()

    def get_clean_shallow_copy(self) -> "NNCFNetwork":
        # WARNING: Will reset pre- and post-ops of the underlying model. Use save_nncf_module_additions
        # and load_nncf_module_additions to preserve these, or temporary_clean_view().
        from nncf.torch.utils import load_module_state  # pylint: disable=cyclic-import
        from nncf.torch.utils import save_module_state  # pylint: disable=cyclic-import

        saved_state = save_module_state(self._model_ref)
        new_interface = NNCFNetworkInterface(
            self._model_ref,
            self._input_infos,
            self._user_dummy_forward_fn,
            self._wrap_inputs_fn,
            self._scopes_without_shape_matching,
            self._ignored_scopes,
            self._target_scopes,
            wrap_outputs_fn=self._wrap_outputs_fn,
        )
        self._model_ref._nncf = new_interface  # pylint:disable=protected-access
        self._model_ref.nncf.reset_nncf_modules()
        load_module_state(self._model_ref, saved_state)
        return self._model_ref

    def get_modules_in_nncf_modules_by_type(self, class_names: List[str]) -> Dict[Scope, nn.Module]:
        nncf_modules = self.get_nncf_modules()
        retval = {}
        for nncf_module, nncf_module_scope in nncf_modules.items():
            nncf_module_scope.pop()
            for relative_scope, target_module in get_all_modules_by_type(nncf_module, class_names).items():
                retval[nncf_module_scope + relative_scope] = target_module
        return retval

    def insert_at_point(self, point: PTInsertionPoint, fn_list: List[Callable]):
        if point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            self._compressed_context.register_pre_hooks(fn_list, point.op_address, point.input_port_id)
        elif point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            self._compressed_context.register_post_hooks(fn_list, point.op_address)
        elif point.insertion_type in [PTInsertionType.NNCF_MODULE_PRE_OP, PTInsertionType.NNCF_MODULE_POST_OP]:
            nncf_module = self.get_module_by_scope(point.module_scope)
            if not isinstance(nncf_module, _NNCFModuleMixin):
                raise RuntimeError(
                    f"Failed to insert pre/post op for not registered custom module {point.module_scope}. NNCF only "
                    f"supports native PyTorch modules with respect to trainable parameter (weight) compressed, such "
                    f"as `torch.nn.Conv2d`. If your model contains a custom, non-PyTorch standard module with trainable"
                    f" weights that should be compressed, you can register it using the "
                    f"`@nncf.register_module` decorator. Please refer to `Compression of custom modules` section in "
                    f"docs/Usage.md for more details."
                )

            norm_target_scope = self._normalize_variable_recurrent_scope(point.module_scope)
            norm_nncf_scopes = []
            for scope_list_for_module in self.get_nncf_module_scopes():
                norm_nncf_scopes.extend([self._normalize_variable_recurrent_scope(x) for x in scope_list_for_module])
            assert norm_target_scope in norm_nncf_scopes  # Required for proper Recurrent/VariableRecurrent addressing
            if point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
                for fn in fn_list:
                    nncf_module.register_pre_forward_operation(fn)
            elif point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
                for fn in fn_list:
                    nncf_module.register_post_forward_operation(fn)
        else:
            raise RuntimeError("Unsupported insertion type: {}".format(point.insertion_type))

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
            return create_dummy_forward_fn(
                self._input_infos,
                with_input_tracing=with_input_tracing,
                wrap_inputs_fn=self._wrap_inputs_fn,
                wrap_outputs_fn=self._wrap_outputs_fn,
                with_output_tracing=with_output_tracing,
            )

        def wrapped_user_dummy_forward_fn(*args, **kwargs):
            self._in_user_dummy_forward = True
            retval = self._user_dummy_forward_fn(*args, **kwargs)
            self._in_user_dummy_forward = False
            return retval

        return wrapped_user_dummy_forward_fn

    def _replace_modules_by_nncf_modules(
        self, model: torch.nn.Module, device: torch.device, eval_op_scopes: List[Scope] = None
    ):
        _, self._nncf_replaced_modules = replace_modules_by_nncf_modules(
            model, ignored_scopes=self._ignored_scopes, target_scopes=self._target_scopes, eval_op_scopes=eval_op_scopes
        )
        model.to(device)
        return model

    def get_nncf_module_scopes(self) -> List[List[Scope]]:
        return list(self._nncf_replaced_modules.values())

    def get_nncf_modules(self) -> Dict[torch.nn.Module, Scope]:
        retval = {}
        for module, scope_set in self._nncf_replaced_modules.items():
            canonical_scope = next(iter(scope_set))
            retval[module] = canonical_scope.copy()
        return retval

    def get_weighted_original_graph_nodes(self, nncf_module_names: List[str] = None) -> List[NNCFNode]:
        retval = set()
        for scope_list in self.get_nncf_module_scopes():
            for nncf_module_scope in scope_list:
                if nncf_module_names is not None:
                    module_name = nncf_module_scope[-1].calling_module_class_name
                    if module_name not in nncf_module_names:
                        continue
                nodes_in_scope = self._original_graph.get_op_nodes_in_scope(nncf_module_scope)
                for node in nodes_in_scope:
                    if node.metatype in OPERATORS_WITH_WEIGHTS_METATYPES:
                        retval.add(node)

        return list(sorted(retval, key=str))

    def rebuild_graph(self, *input_args):
        self._compressed_context.reset_graph()
        dummy_forward_fn = self._get_dummy_forward_fn_for_graph_building(
            with_input_tracing=False, with_output_tracing=False
        )
        builder = GraphBuilder(dummy_forward_fn)

        with training_mode_switcher(self._model_ref, is_training=False):
            self._compressed_graph = builder.build_graph(
                self._model_ref, self._compressed_context, input_infos=self._input_infos
            )

    def is_scope_in_nncf_module_scope(self, scope: Scope) -> bool:
        norm_nncf_scopes = []
        for scope_list_for_module in self.get_nncf_module_scopes():
            norm_nncf_scopes.extend([self._normalize_variable_recurrent_scope(x) for x in scope_list_for_module])
        norm_op_scope = self._normalize_variable_recurrent_scope(scope)
        for nncf_scope in norm_nncf_scopes:
            if norm_op_scope in nncf_scope:
                return True
        return False

    def register_compression_module_type(self, compression_module_type: ExtraCompressionModuleType):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type in self._extra_module_types:
            raise RuntimeError(f"Module type {compression_module_type} is already registered")

        self.__setattr__(attr_name, nn.ModuleDict())
        self._extra_module_types.append(compression_module_type)

    def add_compression_module(
        self, module_key: str, module: nn.Module, compression_module_type: ExtraCompressionModuleType
    ):
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError(f"Module type {compression_module_type} was not registered")
        storage = self.__getattr__(attr_name)
        if module_key in storage:
            raise RuntimeError(f"Module {module_key} is already registered under {attr_name}")
        storage[module_key] = module

    def get_compression_modules_by_type(self, compression_module_type: ExtraCompressionModuleType) -> nn.ModuleDict:
        attr_name = self._compression_module_type_to_attr_name(compression_module_type)
        if compression_module_type not in self._extra_module_types:
            raise RuntimeError(f"Module type {compression_module_type} was not registered")
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
            if scope_element.calling_module_class_name in [
                "Recurrent",
                "VariableRecurrent",
                "VariableRecurrentReverse",
            ]:
                scope_element.calling_module_class_name = "NormalizedName_Recurrent"
        return ret_scope

    def do_dummy_forward(self, force_eval: bool = False):
        """
        Attention: If run with force_eval=False, this may spoil the batchnorm statistics,
        and an eval run of the model will perform much worse than the train run.
        """
        if force_eval:
            train_mode = self._model_ref.training
            self._model_ref.eval()
        with torch.no_grad():
            with self._compressed_context as ctx:
                ctx.base_module_thread_local_replica = self._model_ref
                self._dummy_forward_fn(self._model_ref)
        if force_eval:
            if train_mode:
                self._model_ref.train()

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
                for port_id in [
                    edge.input_port_id,
                ] + edge.parallel_input_port_ids:
                    pre_hook_ip = PreHookInsertionPoint(target_node_name=node.node_name, input_port_id=port_id)
                    pre_hooks.append(pre_hook_ip)

            if issubclass(node.metatype, PTSplitMetatype):
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

        ip_graph = InsertionPointGraph(
            self._original_graph,
            weight_modifiable_node_names=weighted_node_names,
            allowed_pre_hook_insertion_points=pre_hooks,
            allowed_post_hook_insertion_points=post_hooks,
        )
        return ip_graph

    def get_module_by_scope(self, scope: Scope) -> Optional[torch.nn.Module]:
        curr_module = self._model_ref
        return get_module_by_scope(curr_module, scope)

    def get_containing_module(self, node_name: NNCFNodeName) -> torch.nn.Module:
        if self._compressed_graph is not None:
            try:
                scope = self._compressed_graph.get_scope_by_node_name(node_name)
            except RuntimeError:
                nncf_logger.debug(
                    f"Node {node_name} not found in compressed graph when trying to determine "
                    f"the containing module, trying the original graph to see if the node was "
                    f"present there during graph building"
                )
                scope = self._original_graph.get_scope_by_node_name(node_name)
        else:
            scope = self._original_graph.get_scope_by_node_name(node_name)
        return self.get_module_by_scope(scope)

    def get_flops_per_module(self) -> Dict[NNCFNodeName, int]:
        """
        Calculates FLOPS count for modules.
        """
        model = self._model_ref
        flops_count_dict = {}

        def get_hook(name):
            return functools.partial(compute_FLOPs_hook, dict_to_save=flops_count_dict, module_node_name=name)

        hook_list = []
        for nncf_node in self._original_graph.get_all_nodes():
            node_module = self.get_containing_module(nncf_node.node_name)
            hook_list.append(node_module.register_forward_hook(get_hook(nncf_node.node_name)))
        model.nncf.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()
        return flops_count_dict

    def get_MACs_in_model(self):
        """
        Calculates MAC units count for model.
        """
        flops_count_dict = self.nncf.get_flops_per_module()
        total_MACs_count = sum(v // 2 for v in flops_count_dict.values())
        return total_MACs_count

    def save_nncf_module_additions(self) -> Dict[Scope, Tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]]:
        retval = {}
        for nncf_module, module_scope in self.get_nncf_modules().items():
            retval[module_scope] = (deepcopy(nncf_module.pre_ops), deepcopy(nncf_module.post_ops))
        return retval

    def load_nncf_module_additions(
        self, scope_vs_pre_post_ops_dict: Dict[Scope, Tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]]
    ):
        for nncf_module, module_scope in self.get_nncf_modules().items():
            nncf_module.pre_ops = scope_vs_pre_post_ops_dict[module_scope][0]
            nncf_module.post_ops = scope_vs_pre_post_ops_dict[module_scope][1]

    def temporary_clean_view(self):
        class Mgr:
            def __init__(self, model: NNCFNetwork):
                self.model = model
                self.nncf_module_state_dicts = {}
                self.nncf_interface_state_dict = None
                self.nncf_compression_module_types = []

            def __enter__(self):
                self.nncf_module_state_dicts = self.model.nncf.save_nncf_module_additions()
                self.nncf_interface = self.model.nncf
                clean_model = self.model.nncf.get_clean_shallow_copy()
                return clean_model

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.model._nncf = self.nncf_interface
                self.model.nncf.load_nncf_module_additions(self.nncf_module_state_dicts)

        return Mgr(self._model_ref)

    def _collect_eval_op_scopes(self, model: nn.Module, dummy_forward_fn: Callable) -> List[Scope]:
        """
        Returns scopes of the operations in the graph which are executed in evaluation mode.
        """

        tracer = GraphTracer(dummy_forward_fn)
        result = []
        eval_graph = tracer.trace_graph(model, as_eval=True)
        root_scope = Scope()
        for dyn_graph_node in eval_graph.get_all_nodes():
            scope_in_model = dyn_graph_node.op_exec_context.scope_in_model
            if scope_in_model != root_scope:  # happens for ops such as /nncf_model_input_* and /nncf_model_output_*
                result.append(scope_in_model)
        return result

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

    def set_compression_controller(self, ctrl: "PTCompressionAlgorithmController"):
        self.compression_controller = ctrl

    def strip(self, do_copy: bool = True) -> "NNCFNetwork":
        """
        Returns the model object with as much custom NNCF additions as possible removed
        while still preserving the functioning of the model object as a compressed model.
        :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
          will return the currently associated model object "stripped" in-place.
        :return: The stripped model.
        """
        if self.compression_controller is None:
            # PTQ algorithm does not set compressed controller
            from nncf.torch.quantization.strip import strip_quantized_model

            return strip_quantized_model(self._model_ref)
        return self.compression_controller.strip(do_copy)


class NNCFNetworkMeta(type):
    """
    Metaclass for the NNCFNetwork mixin. Has magic methods defined so that the original model object could be
    extended with NNCF-related functionality via a conventional `nncf_network = NNCFNetwork(original_model, ...)`
    syntax and at the same time retain its original class so that downstream class-based checks for the original
    model type don't fail.
    """

    def __call__(
        cls,
        original_model: torch.nn.Module,
        input_infos: List[ModelInputInfo] = None,
        dummy_forward_fn: Callable = None,
        wrap_inputs_fn: Callable[..., None] = None,
        scopes_without_shape_matching: List[str] = None,
        ignored_scopes: List[str] = None,
        target_scopes: List[str] = None,
        wrap_outputs_fn: Callable[..., None] = None,
    ) -> "NNCFNetwork":
        """
        This function plays the role of a "constructor" call in the `nncf_network = NNCFNetwork(original_model, ...)`
        syntax. *_scopes arguments are to be passed as string representation of either
        `nncf.common.graph.graph.NNCFNodeName` or `nncf.torch.dynamic_graph.scope.Scope` objects.
        :param original_model: The original model object to be extended with NNCF functionality.
        :param input_infos: A list of descriptors of each tensor input to the model. Will be used to properly generate
        dummy inputs during internal forward calls of the original model for purposes of control flow graph building.
        :param dummy_forward_fn: A function to be called instead of the model's original forward function during
        control flow graph building.
        :param wrap_inputs_fn: A user-defined function that will be called with the model's forward arguments at each
        call of the NNCFNetwork object and within which the `nncf.torch.dynamic_graph.io_handling.nncf_model_input`
        function is expected to be called upon each tensor among the arguments that is to be treated as an input tensor
        to the model, thus overriding `input_infos`.
        :param scopes_without_shape_matching: A list of scopes in the model in which the activation tensor shapes will
        not be considered for purposes of scope matching - this helps handle RNN-like cases.
        :param ignored_scopes: A list of scopes in the model for which NNCF handling should not be applied. Functions as
        a "denylist". If left unspecified, nothing will be ignored.
        :param target_scopes: A list of scopes in the model for which NNCF handling should be applied. Functions as
        an "allowlist". If left unspecified, everything will be targeted.
        :param wrap_outputs_fn: Same as `wrap_inputs_fn`, but for marking model outputs with
        `nncf.torch.dynamic_graph.io_handling.nncf_model_output` calls.
        :return: The same object as passed in `original_model`, but with internal modules extended/replaced for
        purposes of further NNCF compression, and its class dynamically extended with the `NNCFNetwork` as a base class.
        The object will pass both isinstance(retval, original_model.__class__) and isinstance(retval, NNCFNetwork)
        checks.
        """
        original_class = original_model.__class__
        original_model._nncf = NNCFNetworkInterface(
            original_model,
            input_infos,
            dummy_forward_fn,
            wrap_inputs_fn,
            scopes_without_shape_matching,
            ignored_scopes,
            target_scopes,
            wrap_outputs_fn,
        )  # pylint:disable=protected-access
        # The new class will also have an adjusted metaclass to avoid a "metaclass conflict" upon
        # class creation
        original_metaclass = type(original_model.__class__)
        class_creation_kwds = {}
        if original_metaclass is not type:
            new_metaclass = types.new_class(
                name=original_metaclass.__name__, bases=(NNCFNetworkMeta, original_metaclass)
            )
            class_creation_kwds["metaclass"] = new_metaclass
        new_class = types.new_class(
            name=original_model.__class__.__name__,
            bases=(NNCFNetwork, original_model.__class__),
            kwds=class_creation_kwds,
        )
        # Make the signature of the forward on the resulting object same as for
        # the original forward.
        new_class.forward = _get_nncf_forward_function_with_signature(inspect.signature(original_class.forward))

        # In case of overriding forward by code like `model.forward = wrapper(model.forward)`
        forward_inst_attr_fn = original_model.__dict__.get("forward")
        if forward_inst_attr_fn is not None:
            new_inst_forward = _get_nncf_forward_function_with_signature(inspect.signature(forward_inst_attr_fn))
            original_model.__dict__["forward"] = functools.partial(new_inst_forward, original_model)

        # Make resulting class keep __module__ attributes of the original class,
        # otherwise these will point to NNCF
        new_class.__module__ = original_class.__module__
        original_model.__class__ = new_class

        if isinstance(original_model, torch.nn.Sequential):
            # If the top-level module is Sequential, then the addition of the ._nncf module will result in
            # the NNCFInterface module being iterated over during the torch.nn.Sequential call, with an attempt to call
            # its forward method, which it effectively doesn't have. Employing a special iterator allows to hide the
            # NNCFInterface object during iteration.
            def nncf_safe_iter(self: torch.nn.Sequential):
                return NNCFSkippingIter(iter(self._modules.values()))  # pylint:disable=protected-access

            original_model.__class__.__iter__ = nncf_safe_iter
        return original_model

    def __hash__(cls):
        """
        Makes the dynamically created class object for the processed original model object return the same value when
        hashed as the original class. This allows to gracefully handle the situation when the downstream training
        pipeline checks that the model's class is registered in some registry and determines a training approach
        based on that.
        """
        # expected from a compressed model object to have a NNCFNetwork as 0-th base
        # and original class as 1-st
        if len(cls.__bases__) == 2:
            original_class = cls.__bases__[1]
            return hash(original_class)
        return id(NNCFNetwork)  # conforms to a default hashing behavior in Python for cls objects

    def __eq__(cls, other):
        """
        Makes the dynamically created class object for the processed original model object compare equal with the
        original class object. This allows to gracefully handle the situation when the downstream training
        pipeline checks that the model's class is registered in some registry and determines a training approach
        based on that.
        """
        if len(cls.__bases__) == 2:
            original_class = cls.__bases__[1]
            return original_class == other
        return other is NNCFNetwork


def _get_nncf_forward_function_with_signature(signature: inspect.Signature):
    """
    Create forward function with copy signature of forward function.
    :param signature: Signature of function that will used for forward function.
    :return: New copy of function NNCFNetwork.forward with specified signature.
    """
    fn = NNCFNetwork.forward
    new_forward = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    new_forward.__dict__.update(fn.__dict__)
    new_forward.__signature__ = signature
    if is_debug():
        new_forward = debuggable_forward(new_forward)
    return new_forward


class NNCFNetwork(torch.nn.Module, metaclass=NNCFNetworkMeta):
    """
    A mixin-like class to dynamically extend the original model object's class with.
    """

    def __init__(self, *args, **kwargs):
        """
        In normal situations, the __init__ of the NNCFNetwork will never be called. The constructor-like syntax is
        achieved by a __call__ method defined in the metaclass `NNCFNetworkMeta`.
        """
        super().__init__()
        raise RuntimeError("Direct instantiation of NNCFNetwork objects using __init__ is prohibited.")

    def __call__(self, *args, **kwargs):
        """
        Ensures that functor-like calls of the processed model object will directly trigger the NNCF-specific
        forward call.
        """
        return ORIGINAL_CALL(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Wraps the original forward call, doing additional actions before and after the call to facilitate model
        graph tracing and calling compression-related hooks.
        """
        # pylint:disable=protected-access
        with self.nncf._compressed_context as ctx:  # type: TracingContext
            ctx.base_module_thread_local_replica = self
            args, kwargs = replicate_same_tensors((args, kwargs))
            if not self.nncf._in_user_dummy_forward:
                # If a user supplies own dummy forward, he is responsible for
                # correctly wrapping inputs inside it as well.
                args, kwargs = self.nncf._strip_traced_tensors(args, kwargs)
                args, kwargs = self.nncf._wrap_inputs_fn(args, kwargs)

            # For purposes of scope tracking, need the original forward call to occur as if it were
            # a module call of the corresponding object.
            if self.nncf._original_instance_forward is not None:

                def _unbound_like_original_instance_forward(_self, *args, **kwargs):
                    return self.nncf._original_instance_forward(*args, **kwargs)

                retval = wrap_module_call(_unbound_like_original_instance_forward)(self, *args, **kwargs)

            elif self.nncf._bound_original_forward is None:
                retval = wrap_module_call(self.nncf._original_unbound_forward)(self, *args, **kwargs)
            else:

                def _unbound_like_original_forward(_self, *args, **kwargs):
                    return self.nncf._bound_original_forward(*args, **kwargs)

                retval = wrap_module_call(_unbound_like_original_forward)(self, *args, **kwargs)

            retval = replicate_same_tensors(retval)
            if not self.nncf._in_user_dummy_forward:
                retval = self.nncf._wrap_outputs_fn(retval)

        if self.nncf._kd_loss_handler is not None and self.training:
            self.nncf._kd_loss_handler(retval, *args, **kwargs)
        return retval

    @property
    def nncf(self) -> NNCFNetworkInterface:
        """
        Accessor for all NNCF-specific methods and attributes of the compressed model object.
        """
        # self._nncf is being set in the creation function defined in the NNCFNetworkMeta metaclass
        return self._nncf

    def __setattr__(self, key, value):
        # If setting `forward`, set it on the original model.
        if key == "forward":
            nncf_logger.warning(
                "You are setting `forward` on an NNCF-processed model object.\n"
                "NNCF relies on custom-wrapping the `forward` call in order to function properly.\n"
                "Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behavior.\n"
                "If you need to replace the underlying forward function of the original model so that "
                "NNCF should be using that instead of the original forward function that NNCF saved "
                "during the compressed model creation, you can do this by calling:\n"
                "model.nncf.set_original_unbound_forward(fn)\n"
                "if `fn` has an unbound 0-th `self` argument, or\n"
                "with model.nncf.temporary_bound_original_forward(fn): ...\n"
                "if `fn` already had 0-th `self` argument bound or never had it in the first place."
            )
        super().__setattr__(key, value)


class NNCFSkippingIter:
    """
    An iterator over the regular torch.nn.Module iterator that will skip NNCFInterface objects if they come up.
    """

    def __init__(self, iter_to_wrap: Iterator[Module]):
        self._iter_to_wrap = iter_to_wrap

    def __next__(self):
        item = next(self._iter_to_wrap)
        if isinstance(item, NNCFNetworkInterface):
            item = next(self._iter_to_wrap)
        return item


class LoadStateListener:
    """
    Resets the initialization flags (`initialized`) for all quantization modules on `load_state_dict` call.
    These flags are used to update not loaded params (from checkpoint or model's state)
    on initialization stage of algorithm.
    Flags reset is required on each call of `load_state_dict`, because internal method (`build_graph`)
    restores model state by calling this method.
    """

    def __init__(self, model: "NNCFNetwork", all_quantizations: Dict[str, torch.nn.Module]):
        # pylint: disable=protected-access
        self.hook = model._register_load_state_dict_pre_hook(
            functools.partial(self.hook_fn, quantize_modules=list(all_quantizations.values()))
        )

    def hook_fn(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        quantize_modules: List[torch.nn.Module],
    ):
        for module in quantize_modules:
            module.initialized = False

    def close(self):
        self.hook.remove()
