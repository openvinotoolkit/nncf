# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass

# from functools import partial
# from typing import Callable, Dict, List, Optional, Tuple
from typing import Callable, List, Optional

import torch
import torch.fx

# from torch import Tensor
# from torch import nn
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType

# from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
# from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand

# from nncf.torch.graph.transformations.commands import PTTargetPoint
# from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout

# from torch.nn.parameter import Parameter
# from nncf.torch.model_graph_manager import update_fused_bias
# from nncf.torch.nncf_network import PTInsertionPoint
# from nncf.torch.nncf_network import compression_module_type_to_attr_name
# from nncf.torch.utils import get_model_device
# from nncf.torch.utils import is_multidevice


class FXModelTransformer(ModelTransformer):
    """
    Applies transformations upon Torch FX model.
    """

    def __init__(self, model: torch.fx.GraphModule):
        super().__init__(model)

        self._command_transformation_ordered_pairs = [
            (PTInsertionCommand, self._apply_insertion_transformations),
            (PTSharedFnInsertionCommand, self._apply_shared_nodes_insertion),
        ]

    def transform(self, transformation_layout: PTTransformationLayout) -> torch.fx.GraphModule:
        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        return model

    @staticmethod
    def _apply_insertion_transformations(
        model: torch.fx.GraphModule, transformations: List[PTInsertionCommand]
    ) -> torch.fx.GraphModule:
        """
        Applies insertion transformations to the model.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :param device: Target device for the insertion functions. Applies only to
            functions which are subclassed from torch.nn.Module. Do nothing in case device is None.
        :return: A modified torch.fx.GraphModule.
        """
        node_type = "output"
        graph = model.graph
        outputs = []
        for transformation in transformations:
            for node in graph.nodes:
                if node.name == transformation.target_point.target_node_name:
                    target_node = node
                    break
            target_type = transformation.target_point.target_type
            if target_type == TargetType.OPERATOR_PRE_HOOK:
                ctx = graph.inserting_before(target_node)
                target_nodes = [target_node]
            elif target_type == TargetType.OPERATOR_POST_HOOK:
                ctx = graph.inserting_after(target_node)
                target_nodes = [target_node]
            elif target_type == TargetType.OPERATION_WITH_WEIGHTS:
                # TODO: make it common
                ctx = graph.inserting_after(target_node)
                target_nodes = []
                for input in target_node.all_input_nodes:
                    if input.op == "get_attr":
                        target_nodes.append(input)
            else:
                raise RuntimeError(f"Unsupported target type: {target_type} for transformation: {transformation}")

            with ctx:
                for target_node in target_nodes:
                    outputs.append(
                        graph.create_node(node_type, "", (target_node,), {}, name=target_node.name + "_nncf_output")
                    )
        return model

    @staticmethod
    def _apply_shared_nodes_insertion(
        model: torch.fx.GraphModule,
        transformations: List[PTSharedFnInsertionCommand],
    ) -> torch.fx.GraphModule:
        """
        Applies insertion of PTSharedFnInsertionCommand commands. For each command method inserts
        a torch module to the torch.fx.GraphModule and inserts call hooks for each command target points.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :param device: Target device for the insertion functions. Applies only to
            functions which are subclassed from torch.nn.Module. Do nothing in case device is None.
        :return: A modified torch.fx.GraphModule.
        """
        for transformation in transformations:
            a = 6
            del a
            pass


@dataclass
class QPARAMSPerTensor:
    scale: float
    zero_point: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


@dataclass
class QPARAMPerChannel:
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    axis: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


def insert_qdq_to_model(model: torch.fx.GraphModule, qsetup) -> torch.fx.GraphModule:
    # from prepare
    _fuse_conv_bn_(model)

    # from convert
    original_graph_meta = model.meta
    _insert_qdq_to_model(model, qsetup)

    # Magic. Without this call compiled model
    # is not preformant
    model = GraphModule(model, model.graph)

    model = _fold_conv_bn_qat(model)
    pm = PassManager([DuplicateDQPass()])

    model = pm(model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model


def _insert_qdq_to_model(model: torch.fx.GraphModule, qsetup) -> torch.fx.GraphModule:
    for idx, node in enumerate(list(model.graph.nodes)):
        if node.name not in qsetup:
            continue
        # 1. extract information for inserting q/dq node from activation_post_process
        params = qsetup[node.name]
        node_type = "call_function"
        quantize_op: Optional[Callable] = None
        # scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if isinstance(params, QPARAMPerChannel):
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
            qparams = {
                "_scale_": params.scales,
                "_zero_point_": params.zero_points,
                "_axis_": params.axis,
                "_quant_min_": params.quant_min,
                "_quant_max_": params.quant_max,
                "_dtype_": params.dtype,
            }
        elif isinstance(params, QPARAMSPerTensor):
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            qparams = {
                "_scale_": params.scale,
                "_zero_point_": params.zero_point,
                "_quant_min_": params.quant_min,
                "_quant_max_": params.quant_max,
                "_dtype_": params.dtype,
            }

        else:
            raise RuntimeError(f"params {params} are unknown")
        # 2. replace activation_post_process node with quantize and dequantize
        graph = model.graph

        # TODO: use metatype to get correct input_port_id
        # Do not quantize already quantized nodes
        # inserting_before handle only order in the graph generated code.
        # so, inserting quantize-dequantize and all constant nodes before the usage of the nodes
        with graph.inserting_before(node):
            quantize_op_inputs = [node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"] and (not isinstance(value_or_node, (float, int))):
                    # For scale and zero_point values we register them as buffers in the root module.
                    # However, note that when the values are not tensors, as in the case of
                    # per_tensor quantization, they will be treated as literals.
                    # However, registering them as a node seems to cause issue with dynamo
                    # tracing where it may consider tensor overload as opposed to default.
                    # With extra check of scale and zero_point being scalar, it makes
                    # sure that the default overload can be used.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(model, graph, str(idx) + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store
                    # them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

        with graph.inserting_after(node):
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # use the same qparams from quantize op
        dq_inputs = [quantized_node] + quantize_op_inputs[1:]
        user_dq_nodes = []
        with graph.inserting_after(quantized_node):
            for user in node.users:
                if user is quantized_node:
                    continue
                user_dq_nodes.append((user, graph.call_function(dequantize_op, tuple(dq_inputs), {})))

        for user, dq_node in user_dq_nodes:
            user.replace_input_with(node, dq_node)

        # node.replace_all_uses_with(dequantized_node)
        # graph.erase_node(node)
        from torch.fx.passes.graph_drawer import FxGraphDrawer

        g = FxGraphDrawer(model, "model_after_qdq_insertion")
        g.get_dot_graph().write_svg("model_after_qdq_insertion.svg")
