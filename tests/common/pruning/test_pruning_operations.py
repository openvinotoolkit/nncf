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

from functools import partial

import numpy as np
import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.operations import BasePruningOp
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from tests.common.pruning import dummy_types
from tests.common.pruning.tensor import NPNNCFTensor
from tests.common.pruning.tensor import NPNNCFTensorProcessor


@pytest.mark.parametrize(
    "pruning_op,metatype,accept_pruned_input",
    [
        (dummy_types.DummyInputPruningOp, dummy_types.DummyInputMetatype, False),
        (dummy_types.DummyOutputPruningOp, dummy_types.DummyOutputMetatype, True),
        (dummy_types.DummyStopMaskForwardPruningOp, dummy_types.DummyStopMaskForwardMetatype, False),
    ],
)
def test_stop_propagate_ops(pruning_op, metatype, accept_pruned_input):
    graph = NNCFGraph()
    node = graph.add_nncf_node("conv_op", metatype.name, metatype)
    assert pruning_op.accept_pruned_input(node) == accept_pruned_input
    pruning_op.mask_propagation(node, graph, NPNNCFTensorProcessor)
    assert node.attributes["output_mask"] is None


@pytest.mark.parametrize("dummy_op_class", [dummy_types.DummyIdentityMaskForward, dummy_types.DummyBatchNormPruningOp])
def test_identity_mask_propogation_prune_ops(dummy_op_class):
    assert dummy_op_class.accept_pruned_input(None)
    graph = NNCFGraph()
    conv_op = graph.add_nncf_node("conv_op", "conv", dummy_types.DummyConvMetatype)
    identity_ops = []
    for alias in dummy_op_class.get_all_op_aliases():
        identity_op = graph.add_nncf_node("identity", alias, dummy_types.DummyIdentityMaskForwardMetatype)
        graph.add_edge_between_nncf_nodes(
            from_node_id=conv_op.node_id,
            to_node_id=identity_op.node_id,
            tensor_shape=[10] * 4,
            input_port_id=0,
            output_port_id=0,
            dtype=Dtype.FLOAT,
        )
        identity_ops.append(identity_op)
    # Check with and without masks
    for output_mask in [None, NPNNCFTensor(np.ones((10,)))]:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.attributes["output_mask"] = output_mask
        MaskPropagationAlgorithm(
            graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
        ).mask_propagation()
        for identity_op in identity_ops:
            identity_op = graph.get_node_by_id(identity_op.node_id)
            assert np.all(identity_op.attributes["output_mask"] == output_mask)


@pytest.mark.parametrize("valid_masks", [None, True, False])
def test_elementwise_prune_ops(valid_masks):
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node("conv_op_0", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node("conv_op_1", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    elementwise_op = graph.add_nncf_node(
        "elementwise", dummy_types.DummyElementwiseMetatype.name, dummy_types.DummyElementwiseMetatype
    )
    add_node = partial(
        graph.add_edge_between_nncf_nodes, tensor_shape=[10] * 4, input_port_id=0, output_port_id=0, dtype=Dtype.FLOAT
    )
    # conv_op_0 -> elementwise
    add_node(from_node_id=conv_op_0.node_id, to_node_id=elementwise_op.node_id)

    # conv_op_1 -> elementwise
    add_node(from_node_id=conv_op_1.node_id, to_node_id=elementwise_op.node_id)

    masks = [NPNNCFTensor(np.ones((10,))), NPNNCFTensor(np.ones((10,)))] if valid_masks is not None else [None, None]

    def set_masks(masks, ops):
        for conv_op, mask in zip(ops, masks):
            conv_op = graph.get_node_by_id(conv_op.node_id)
            conv_op.attributes["output_mask"] = mask

    if valid_masks is None or valid_masks:
        if valid_masks:
            set_masks(masks, [conv_op_0, conv_op_1])
        MaskPropagationAlgorithm(
            graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
        ).mask_propagation()
        elementwise_op = graph.get_node_by_id(elementwise_op.node_id)
        assert np.all(elementwise_op.attributes["output_mask"] == masks[0])
    else:

        def check_wrong_masks(masks):
            with pytest.raises(AssertionError):
                set_masks(masks, [conv_op_0, conv_op_1])
                MaskPropagationAlgorithm(
                    graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
                ).mask_propagation()

        masks[0].tensor[0] = 0
        check_wrong_masks(masks)
        masks[0] = NPNNCFTensorProcessor.concatenate([masks[1], NPNNCFTensor(np.array([1]))], axis=0)
        check_wrong_masks(masks)


@pytest.mark.parametrize(
    "num_channels,num_groups,accept_pruned_input_ref", [(10, 10, True), (10, 5, False), (10, 1, False)]
)
def test_group_norm_pruning_ops(num_channels, num_groups, accept_pruned_input_ref):
    graph = NNCFGraph()
    conv_op = graph.add_nncf_node("conv_op", "conv", dummy_types.DummyConvMetatype)
    group_norm_layer_attributes = GroupNormLayerAttributes(True, num_channels=num_channels, num_groups=num_groups)
    group_norm_op = graph.add_nncf_node(
        "identity",
        dummy_types.DummyGroupNormMetatype.name,
        dummy_types.DummyGroupNormMetatype,
        layer_attributes=group_norm_layer_attributes,
    )
    assert dummy_types.DummyGroupNormPruningOp.accept_pruned_input(group_norm_op) == accept_pruned_input_ref
    graph.add_edge_between_nncf_nodes(
        from_node_id=conv_op.node_id,
        to_node_id=group_norm_op.node_id,
        tensor_shape=[10] * 4,
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    # Check with and without masks
    for output_mask in [None, NPNNCFTensor(np.ones((10,)))]:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.attributes["output_mask"] = output_mask
        MaskPropagationAlgorithm(
            graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
        ).mask_propagation()
        identity_op = graph.get_node_by_id(group_norm_op.node_id)
        if not accept_pruned_input_ref:
            output_mask = None

        assert np.all(identity_op.attributes["output_mask"] == output_mask)


class DummyMaskProducerMetatype(dummy_types.DummyDefaultMetatype):
    name = "mask_producer"


@dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyMaskProducerMetatype.name)
class MockOpMaskProducer(BasePruningOp):
    additional_types = [DummyMaskProducerMetatype.name]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode):
        pass

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph, tensor_processor: NNCFPruningBaseTensorProcessor):
        pass


@pytest.mark.parametrize("transpose", [True, False], ids=["transpose", "not_transpose"])
@pytest.mark.parametrize(
    "layer_attributes,ref_accept_pruned_input,conv_type",
    [
        ({"in_channels": 5, "out_channels": 10, "groups": 1}, True, "usual_conv"),
        ({"in_channels": 10, "out_channels": 20, "groups": 5}, False, "grouped_conv_no_depthwise"),
        ({"in_channels": 10, "out_channels": 20, "groups": 10}, False, "multiply_grouped_conv"),
        ({"in_channels": 20, "out_channels": 20, "groups": 20}, True, "depthwise_conv"),
    ],
    ids=["usual_conv", "grouped_conv_no_depthwise", "multiply_grouped_conv", "depthwise_conv"],
)
def test_conv_pruning_ops(transpose, layer_attributes, ref_accept_pruned_input, conv_type):
    default_conv_params = {
        "weight_requires_grad": True,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "dilations": (1, 1),
        "padding_values": [0, 0],
    }
    graph = NNCFGraph()
    dummy_op_before = graph.add_nncf_node("dummy_op_before", DummyMaskProducerMetatype.name, DummyMaskProducerMetatype)
    target_conv_attributes = ConvolutionLayerAttributes(transpose=transpose, **layer_attributes, **default_conv_params)
    conv_op_target = graph.add_nncf_node(
        "conv_op_target",
        dummy_types.DummyConvMetatype.name,
        dummy_types.DummyConvMetatype,
        layer_attributes=target_conv_attributes,
    )
    graph.add_edge_between_nncf_nodes(
        from_node_id=dummy_op_before.node_id,
        to_node_id=conv_op_target.node_id,
        tensor_shape=[layer_attributes["in_channels"]] * 4,
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    pruning_op_class = dummy_types.DummyTransposeConvPruningOp if transpose else dummy_types.DummyConvPruningOp
    assert pruning_op_class.accept_pruned_input(conv_op_target) == ref_accept_pruned_input
    ones_input_mask = NPNNCFTensor(np.ones((layer_attributes["in_channels"],)))
    ones_output_mask = NPNNCFTensor(np.ones((layer_attributes["out_channels"],)))
    # Check all combinations of masks
    for input_mask in [None, ones_input_mask]:
        for output_mask in [None, ones_output_mask]:
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            conv_op_target = graph.get_node_by_id(conv_op_target.node_id)
            dummy_op_before.attributes["output_mask"] = input_mask
            conv_op_target.attributes["output_mask"] = output_mask
            MaskPropagationAlgorithm(
                graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
            ).mask_propagation()
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            conv_op_target = graph.get_node_by_id(conv_op_target.node_id)
            if conv_type == "usual_conv":
                assert np.all(conv_op_target.attributes["output_mask"] == output_mask)
            elif conv_type in ["grouped_conv_no_depthwise", "multiply_grouped_conv"]:
                assert conv_op_target.attributes["output_mask"] is None
            else:
                assert np.all(conv_op_target.attributes["output_mask"] == input_mask)


def test_linear_pruning_ops():
    graph = NNCFGraph()
    in_features = out_features = 10
    dummy_op_before = graph.add_nncf_node("dummy_op_before", DummyMaskProducerMetatype.name, DummyMaskProducerMetatype)
    target_linear_attributes = LinearLayerAttributes(
        weight_requires_grad=True, in_features=in_features, out_features=out_features
    )
    linear_op_target = graph.add_nncf_node(
        "linear_op_target",
        dummy_types.DummyLinearMetatype.name,
        dummy_types.DummyLinearMetatype,
        layer_attributes=target_linear_attributes,
    )
    graph.add_edge_between_nncf_nodes(
        from_node_id=dummy_op_before.node_id,
        to_node_id=linear_op_target.node_id,
        tensor_shape=[in_features] * 2,
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    # Check linear layer always accept pruned input
    assert dummy_types.LinearPruningOp.accept_pruned_input(linear_op_target)
    ones_input_mask = NPNNCFTensor(np.ones((in_features)))
    ones_output_mask = NPNNCFTensor(np.ones((out_features)))
    # Check all combinations of masks
    for input_mask in [None, ones_input_mask]:
        for output_mask in [None, ones_output_mask]:
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            linear_op_target = graph.get_node_by_id(linear_op_target.node_id)
            dummy_op_before.attributes["output_mask"] = input_mask
            linear_op_target.attributes["output_mask"] = output_mask
            MaskPropagationAlgorithm(
                graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
            ).mask_propagation()
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            linear_op_target = graph.get_node_by_id(linear_op_target.node_id)
            assert np.all(linear_op_target.attributes["output_mask"] == output_mask)


@pytest.mark.parametrize("empty_mask_left_branch", [False, True])
@pytest.mark.parametrize("empty_mask_right_branch", [False, True])
@pytest.mark.parametrize("right_branch_output_channels", [5, 10])
def test_convs_elementwise_source_before_concat(
    empty_mask_right_branch, empty_mask_left_branch, right_branch_output_channels
):
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node("conv_op_0", "conv", dummy_types.DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node("conv_op_1", "conv", dummy_types.DummyConvMetatype)
    conv_op_2 = graph.add_nncf_node("conv_op_2", "conv", dummy_types.DummyConvMetatype)
    elementwise_node = graph.add_nncf_node("elementwise_node", "elementwise", dummy_types.DummyElementwiseMetatype)
    concat_layer_attributes = MultipleInputLayerAttributes(2)
    concat_node = graph.add_nncf_node(
        "concat_node", "concat", dummy_types.DummyConcatMetatype, layer_attributes=concat_layer_attributes
    )
    add_node = partial(graph.add_edge_between_nncf_nodes, input_port_id=0, output_port_id=0, dtype=Dtype.FLOAT)

    # conv_op_0 -> elementwise_node
    add_node(from_node_id=conv_op_0.node_id, to_node_id=elementwise_node.node_id, tensor_shape=[10] * 4)

    # conv_op_1 -> elementwise_node
    add_node(from_node_id=conv_op_1.node_id, to_node_id=elementwise_node.node_id, tensor_shape=[10] * 4)

    # elementwise_node -> concat_node
    add_node(from_node_id=elementwise_node.node_id, to_node_id=concat_node.node_id, tensor_shape=[10] * 4)

    # conv_op_2 -> concat_node
    add_node(
        from_node_id=conv_op_2.node_id,
        to_node_id=concat_node.node_id,
        tensor_shape=[10, 10, right_branch_output_channels, 10],
    )

    # Set masks
    if not empty_mask_left_branch:
        for conv_op in [conv_op_0, conv_op_1]:
            conv_op = graph.get_node_by_id(conv_op.node_id)
            conv_op.attributes["output_mask"] = NPNNCFTensor(np.ones(10))

    if not empty_mask_right_branch:
        conv_op = graph.get_node_by_id(conv_op_2.node_id)
        conv_op.attributes["output_mask"] = NPNNCFTensor(np.ones(right_branch_output_channels))

    # Propagate masks
    MaskPropagationAlgorithm(
        graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
    ).mask_propagation()
    # Check with masks
    concat_node = graph.get_node_by_id(concat_node.node_id)
    if empty_mask_left_branch and empty_mask_right_branch:
        assert concat_node.attributes["output_mask"] is None
    else:
        reference_mask = np.ones((10 + right_branch_output_channels,))
        np.testing.assert_equal(concat_node.attributes["output_mask"].tensor, reference_mask)


def test_concat_output_tensor_device():
    graph = NNCFGraph()
    dummy_ops = [
        graph.add_nncf_node(f"dummy_op_{i}", DummyMaskProducerMetatype.name, DummyMaskProducerMetatype)
        for i in range(3)
    ]
    concat_layer_attributes = MultipleInputLayerAttributes(2)
    concat_node = graph.add_nncf_node(
        "concat_node", "concat", dummy_types.DummyConcatMetatype, layer_attributes=concat_layer_attributes
    )
    for op in dummy_ops:
        graph.add_edge_between_nncf_nodes(
            from_node_id=op.node_id,
            to_node_id=concat_node.node_id,
            tensor_shape=[10] * 4,
            input_port_id=0,
            output_port_id=0,
            dtype=Dtype.FLOAT,
        )

    # Set mask to last dummy node
    ref_device = "some_test_device"
    for op in dummy_ops[:-1]:
        op = graph.get_node_by_id(op.node_id)
        op.attributes["output_mask"] = None

    last_op = graph.get_node_by_id(dummy_ops[-1].node_id)
    last_op.attributes["output_mask"] = NPNNCFTensor(np.ones(10), dummy_device=ref_device)
    # Propagate masks
    MaskPropagationAlgorithm(
        graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
    ).mask_propagation()
    # Check concat op has appropriate device
    concat_node = graph.get_node_by_id(concat_node.node_id)
    assert concat_node.attributes["output_mask"].device == ref_device


RESHAPE_TEST_CASES = [
    ["flatten", (1, 1, 64), (1, 64)],
    ["flatten", (1, 32, 64), (1, 2048)],
    ["flatten", (1, 32, 64, 1), (1, 2048)],
    ["reshape", (1, 32, 64), (1, 2048)],  # Flatten
    ["reshape", (1, 64, 32), (1, 2048)],  # Flatten
    ["reshape", (1, 32, 64, 1), (1, 2048)],  # Flatten
    ["reshape", (1, 1, 64), (1, 1, 1, 64)],  # Expand
    ["reshape", (1, 1, 1, 64), (1, 64)],  # Squeeze
    ["reshape", (1, 1, 1, 64), (1, 1, 1, 64)],
    ["reshape", (1, 1, 32, 64), (1, 64, 32)],
    ["reshape", (1, 1, 32, 64), (1, 64, 16, 16)],
    ["reshape", (1, 1, 64, 1), (1, 1, 1, 64)],
]


METATYPES_MAP = {
    "flatten": {"metatype": dummy_types.DummyFlattenMetatype, "ops": dummy_types.DummyFlattenPruningOp},
    "reshape": {"metatype": dummy_types.DummyReshapeMetatye, "ops": dummy_types.DummyReshapePruningOp},
}


@pytest.mark.parametrize(("node_type", "input_shape", "output_shape"), RESHAPE_TEST_CASES)
def test_reshape_accept_pruned_input(node_type, input_shape, output_shape):
    node_name = "dummy_reshape"
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)
    graph = NNCFGraph()
    node = graph.add_nncf_node(
        node_name, node_type, METATYPES_MAP[node_type]["metatype"], layer_attributes=layer_attributes
    )

    actual_accept_pruned_input = METATYPES_MAP[node_type]["ops"].accept_pruned_input(node)
    assert actual_accept_pruned_input


REF_OUTPUT_MASK_RESHAPE = (
    [
        [np.ones(10), "error"],
        [np.array([0, 1] * 32), np.reshape(np.array([[0] * 32 + [1] * 32] * 32), -1)],
        [np.array([0, 1] * 32), np.reshape(np.array([[0] * 32 + [1] * 32] * 32), -1)],
    ]
    * 2
    + [[np.array([0, 1] * 32)] * 2] * 5
    + [[np.ones(10), None]] * 1
)


@pytest.mark.parametrize(
    ("node_type", "input_shape", "output_shape", "output_mask", "output_mask_ref"),
    [x + y for x, y in zip(RESHAPE_TEST_CASES, REF_OUTPUT_MASK_RESHAPE)],
)
def test_reshape_metatype_mask_prop(node_type, input_shape, output_shape, output_mask, output_mask_ref):
    node_name = "dummy_reshape"
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)

    graph = NNCFGraph()
    prev_node = graph.add_nncf_node("prev_node", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    reshape_node = graph.add_nncf_node(
        node_name, node_type, METATYPES_MAP[node_type]["metatype"], layer_attributes=layer_attributes
    )

    graph.add_edge_between_nncf_nodes(
        from_node_id=prev_node.node_id,
        to_node_id=reshape_node.node_id,
        tensor_shape=output_shape,
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )
    # Check both None mask and not None mask
    for output_mask_cur, output_mask_ref_cur in [(None, None), (output_mask, output_mask_ref)]:
        # Get reference to graph node
        prev_node = graph.get_node_by_id(prev_node.node_id)
        reshape_node = graph.get_node_by_id(reshape_node.node_id)
        prev_node.attributes["output_mask"] = NPNNCFTensor(output_mask_cur) if output_mask_cur is not None else None
        if isinstance(output_mask_ref_cur, str):
            with pytest.raises(AssertionError):
                METATYPES_MAP[node_type]["ops"].mask_propagation(reshape_node, graph, NPNNCFTensorProcessor)
        else:
            METATYPES_MAP[node_type]["ops"].mask_propagation(reshape_node, graph, NPNNCFTensorProcessor)
            if output_mask_ref_cur is None:
                assert reshape_node.attributes["output_mask"] is None
            else:
                assert np.all(reshape_node.attributes["output_mask"].tensor == output_mask_ref_cur)


@pytest.mark.parametrize("node_type", ["reshape", "flatten"])
def test_reshape_is_last_op(node_type):
    node_name = "dummy_reshape"
    layer_attributes = None

    graph = NNCFGraph()
    prev_node = graph.add_nncf_node("prev_node", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    reshape_node = graph.add_nncf_node(
        node_name, node_type, METATYPES_MAP[node_type]["metatype"], layer_attributes=layer_attributes
    )

    assert not METATYPES_MAP[node_type]["ops"].accept_pruned_input(reshape_node)

    graph.add_edge_between_nncf_nodes(
        from_node_id=prev_node.node_id,
        to_node_id=reshape_node.node_id,
        tensor_shape=[1, 32],
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )

    for output_mask in (None, NPNNCFTensor(np.ones((10,)))):
        prev_node = graph.get_node_by_id(prev_node.node_id)
        reshape_node = graph.get_node_by_id(reshape_node.node_id)
        prev_node.attributes["output_mask"] = output_mask
        METATYPES_MAP[node_type]["ops"].mask_propagation(reshape_node, graph, NPNNCFTensorProcessor)
        assert reshape_node.attributes["output_mask"] is None


SPLIT_TEST_CASES = [
    ["chunk", 2, 0],
    ["chunk", 2, 1],
]


@pytest.mark.parametrize(("node_type", "chunks", "axis"), SPLIT_TEST_CASES)
def test_split_accept_pruned_input(node_type, chunks, axis):
    node_name = "dummy_split"
    layer_attributes = MultipleOutputLayerAttributes(chunks, axis)
    graph = NNCFGraph()
    node = graph.add_nncf_node(node_name, node_type, dummy_types.DummySplitMetatype, layer_attributes=layer_attributes)

    actual_accept_pruned_input = dummy_types.DummySplitPruningOp.accept_pruned_input(node)
    assert actual_accept_pruned_input


@pytest.mark.parametrize("empty_mask_left_branch", [False])
@pytest.mark.parametrize("empty_mask_right_branch", [False])
@pytest.mark.parametrize("layer_attributes", [{"in_channels": 5, "out_channels": 10, "groups": 1}])
def test_split_metatype_mask_prop(empty_mask_left_branch, empty_mask_right_branch, layer_attributes):
    default_conv_params = {
        "weight_requires_grad": True,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "dilations": (1, 1),
        "padding_values": [0, 0],
    }
    split_attributes = MultipleOutputLayerAttributes(chunks=2, axis=1)
    conv_attributes = ConvolutionLayerAttributes(transpose=False, **layer_attributes, **default_conv_params)

    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node(
        "conv_op_0",
        dummy_types.DummyConvMetatype.name,
        dummy_types.DummyConvMetatype,
        layer_attributes=ConvolutionLayerAttributes,
    )
    split_node = graph.add_nncf_node(
        "split_0",
        dummy_types.DummySplitMetatype.name,
        dummy_types.DummySplitMetatype,
        layer_attributes=split_attributes,
    )
    conv_op_1 = graph.add_nncf_node(
        "conv_op_1", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype, layer_attributes=conv_attributes
    )
    conv_op_2 = graph.add_nncf_node(
        "conv_op_2", dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype, layer_attributes=conv_attributes
    )

    add_node = partial(graph.add_edge_between_nncf_nodes, input_port_id=0, output_port_id=0, dtype=Dtype.FLOAT)

    # conv_op_0 -> split_node
    add_node(from_node_id=conv_op_0.node_id, to_node_id=split_node.node_id, tensor_shape=[10] * 4)

    # split_node -> conv_op_1
    add_node(from_node_id=split_node.node_id, to_node_id=conv_op_1.node_id, tensor_shape=[10, 5, 10, 10])

    # split_node -> conv_op_2
    add_node(from_node_id=split_node.node_id, to_node_id=conv_op_2.node_id, tensor_shape=[10, 5, 10, 10])

    # Set masks
    conv_op_0_node = graph.get_node_by_id(conv_op_0.node_id)
    conv_op_0_node.attributes["output_mask"] = NPNNCFTensor(np.ones(10))

    # Set in_channles
    for node in (conv_op_1, conv_op_2):
        conv_node = graph.get_node_by_id(node.node_id)
        conv_node.layer_attributes.in_channels = 5

    # Propagate masks
    MaskPropagationAlgorithm(
        graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES, NPNNCFTensorProcessor
    ).mask_propagation()

    # Check with masks
    split_node = graph.get_node_by_id(split_node.node_id)
    split_output_masks = split_node.attributes["output_mask"]
    reference_mask = np.ones((5,))
    for node in (conv_op_1, conv_op_2):
        conv_node = graph.get_node_by_id(conv_op_1.node_id)
        output_mask = split_output_masks[conv_node.node_name]
        np.testing.assert_equal(output_mask.tensor, reference_mask)
