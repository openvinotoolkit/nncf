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

import functools
import inspect
import itertools
from typing import List
from typing import Dict

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.tensorflow.nncf_operation import NNCFOperation
from nncf.experimental.tensorflow.graph.transformations.commands import TFTargetPoint
from nncf.experimental.tensorflow.graph.argprovider import TF_ARG_PROVIDERS
from nncf.experimental.tensorflow.nncf_context import get_nncf_context
from nncf.experimental.tensorflow.scope import get_op_name
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
from nncf.experimental.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.experimental.tensorflow.graph.argprovider import replace


class Hook:
    """
    Contains the NNCF operations and target point where
    these operations should be applied.
    """

    def __init__(self,
                 operations: List[NNCFOperation],
                 target_point: TFTargetPoint):
        """
        Initializes the hook.

        :param operations: List of the NNCF operations in the correct order.
            The operation at index 0 is applied first, with index -1 last.
        :param target_point: A target point. Contains information on where the
            operations should be applied.
        """
        self._operations = operations
        self._target_point = target_point

        arg_provider_cls = TF_ARG_PROVIDERS.registry_dict.get(self._target_point.op_type_name)
        if arg_provider_cls is None:
            pass
            # raise ValueError(f'Unexpected type of the TensorFlow operation: {self._target_point.op_type_name}'
            #                  'Register an `ArgProvider` instance for this type in the '
            #                  '`TF_ARG_PROVIDERS` registry, please.')

        if arg_provider_cls:
            self._arg_provider = arg_provider_cls()

    @property
    def operations(self) -> List[NNCFOperation]:
        return self._operations

    @property
    def target_point(self) -> TFTargetPoint:
        return self._target_point

    @property
    def is_pre_hook(self) -> bool:
        """
        Returns the boolean flag that specified whether the hook should be
        applied before or after target operation.

        :return: `True` if hook should be applied before target operation,
            `False` otherwise.
        """
        return self._target_point.type == TargetType.OPERATOR_PRE_HOOK

    def __call__(self, *args, **kwargs):
        """
        Applies this hook.

        :return: A tuple (args, kwargs)
        """
        if self.is_pre_hook:
            get_fn = self._arg_provider.get_input
            set_fn = self._arg_provider.set_input
        else:
            get_fn = self._arg_provider.get_output
            set_fn = self._arg_provider.set_output

        x = get_fn(self.target_point.port_id, args, kwargs)
        for op in self.operations:
            x = op(*(x,))
        args_, kwargs_ = set_fn(self.target_point.port_id, x, args, kwargs)

        return args_, kwargs_


class TFPatcher:
    """
    """

    @staticmethod
    def apply(pre_hooks: Dict[str, List[Hook]], post_hooks: Dict[str, List[Hook]]):
        """
        :param pre_hooks:
        :param post_hooks:
        """
        unique_op_type_names = set()
        for hook in itertools.chain(*pre_hooks.values(), *post_hooks.values()):
            unique_op_type_names.add(hook.target_point.op_type_name)

        op_type_name_to_op_info_map = TFPatcher._get_ops_info(unique_op_type_names)
        for op_type_name in unique_op_type_names:
            fn, fn_name, module = op_type_name_to_op_info_map[op_type_name]
            setattr(module, fn_name, TFPatcher._wrap_tf_operation(fn, op_type_name, pre_hooks, post_hooks))

        ops.name_scope = TFPatcher._wrap_name_scope_internal_fn(ops.name_scope)
        ops.name_scope_v2.__enter__ = TFPatcher._wrap_name_scope_v2_enter_fn(ops.name_scope_v2.__enter__)
        tf.name_scope.__enter__ = TFPatcher._wrap_name_scope_v2_enter_fn(tf.name_scope.__enter__)

    @staticmethod
    def _get_ops_info(op_type_names):
        """
        :param op_type_names:
        :return:
        """
        raw_ops = inspect.getmembers(tf.raw_ops, predicate=inspect.isfunction)
        op_type_name_to_fn_map = dict(raw_ops)

        op_type_name_to_op_info_map = {}
        for op_type_name in op_type_names:
            original_fn = op_type_name_to_fn_map[op_type_name]

            module = inspect.getmodule(original_fn)
            fn_name = original_fn.__name__
            fn = getattr(module, fn_name)

            op_type_name_to_op_info_map[op_type_name] = (fn, fn_name, module)

        return op_type_name_to_op_info_map

    @staticmethod
    def _apply_hooks(hooks: List[Hook], args, kwargs):
        """
        Applies hooks.

        :param hooks: A list of hooks.
        :return: A tuple (args, kwargs).
        """
        for hook in hooks:
            args, kwargs = hook(*args, **kwargs)
        return args, kwargs

    @staticmethod
    def _wrap_tf_operation(fn,
                           op_type_name: str,
                           pre_hooks: Dict[str, List[Hook]],
                           post_hooks: Dict[str, List[Hook]]):
        """
        Wraps python method which represents TensorFlow operation.

        :param fn: TensorFlow op i.e. function from the `tf.raw_ops` module.
        :param op_type_name: Operation type name (name of the function
            from the `tf.raw_ops` module)
        :param pre_hooks:
        :param post_hooks:
        :return: Wrapped function.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nncf_context = get_nncf_context()

            # Should we wrap current operation?
            if not nncf_context.wrap_ops:
                return fn(*args, **kwargs)

            op_name = get_op_name(op_type_name, kwargs.get('name'))
            with nncf_context.enter(wrap_ops=False):
                # Apply pre-hooks
                args, kwargs = TFPatcher._apply_hooks(pre_hooks.get(op_name, []), args, kwargs)
                # Apply TensorFlow operation
                outputs = fn(*args, **kwargs)
                # Apply post-hooks
                (outputs,), _ = TFPatcher._apply_hooks(post_hooks.get(op_name, []), (outputs,), {})
            return outputs

        return wrapper

    @staticmethod
    def _wrap_name_scope_internal_fn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 4:
                args = replace(args, 3, False)
            else:
                kwargs['skip_on_eager'] = False
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def _wrap_name_scope_v2_enter_fn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nncf_context = get_nncf_context()

            if tf.executing_eagerly() and nncf_context.in_call:
                obj, = args  # self
                eager_context = context.context()
                old_name = eager_context.scope_name
                name = obj._name

                if not name:
                    scope_name = ''
                elif name[-1] == '/':
                    scope_name = name
                elif old_name:
                    scope_name = nncf_context.unique_name(old_name + name) + '/'
                else:
                    scope_name = name + '/'
                eager_context.scope_name = scope_name

                def _restore_name_scope(*_):
                    eager_context.scope_name = old_name

                obj._exit_fns.append(_restore_name_scope)
            else:
                scope_name = func(*args, **kwargs)
            return scope_name

        return wrapper


class TFModelTransformer(ModelTransformer):
    """
    Applies transformations to the NNCF network.
    """

    def __init__(self, model: NNCFNetwork):
        """
        Initializes the model transformer.

        :param model: NNCF network.
        """
        super().__init__(model)

    def transform(self, transformation_layout: TFTransformationLayout) -> NNCFNetwork:
        """
        Applies transformations to the model.

        :param transformation_layout: An instance of `TransformationLayout` that
            includes a list of transformations to be applied to the NNCF network.
        :return: The transformed NNCF network.
        """

        model_hooks = getattr(self._model, '_hooks')
        for command in transformation_layout.transformations:
            if command.type == TransformationType.INSERT:
                hook = Hook(command.insertion_objects, command.target_point)
                model_hooks.append(hook)
            elif command.type == TransformationType.REMOVE:
                # TODO(andrey-churkin): Add support
                pass
            else:
                raise ValueError(f'Transformation type {command.type} does not support.')

        pre_hooks = {}  # Dict[str, List[Hook]]
        post_hooks = {}  # Dict[str, List[Hook]]
        for hook in model_hooks:
            hooks = pre_hooks if hook.is_pre_hook else post_hooks
            hooks.setdefault(hook.target_point.op_name, []).append(hook)

        TFPatcher.apply(pre_hooks, post_hooks)

        with self._model.distribute_strategy.scope():
            for op in self._model.nncf_operations:
                op.build(self._model)

        return self._model
