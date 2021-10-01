import numpy as np
import pytest

from typing import List

from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.graph import NNCFGraph
from nncf.common.pruning.export_helpers import(
OpElementwise,
OpConvolution,
 OpConcat,
OpStopMaskForwardOps,

)


TEST_CASES = [
    ['flatten', (1, 1, 64), (1,)],
    ['flatten', (1, 32, 64), (1,)],
    ['reshape', (1, 32, 64), (1,)], # Flatten
    ['reshape', (1, 1, 64), (1, 1, 1, 64)], # Expand
    ['reshape', (1, 1, 1, 64), (1, 64)], # Squeeze
    ['reshape', (1, 1, 1, 64), (1, 1, 64, 1)],# Transpose
    ['reshape', (1, 1, 32, 64), (1, 64, 32)],# Transpose
    ['reshape', (1, 1, 32, 64), (1, 64, 16, 16)],
]

REF_ACCEPT_PRUNED = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False
]


class DummyInputMetatype(OperatorMetatype):
    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return ['input']


class DummyElementwise(OperatorMetatype):
    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return ['elementwise']


class DummyStopPropOp(OperatorMetatype):
    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return ['stop_prop_op']


class DummyConvMetatype(OperatorMetatype):
    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return ['conv']


class DummyConcatMetatype(OperatorMetatype):
    @classmethod
    def get_all_aliases(cls) -> List[str]:
        return ['concat']


class DummyOpInput(OpConcat):
    additional_types = ['input']


class DummyOpStopMaskForward(OpStopMaskForwardOps):
    additional_types = ['stop_prop_op']


class DummyOpConv(OpConvolution):
    additional_types = ['conv']


class DummyOpElementwise(OpElementwise):
    additional_types = ['elementwise']


class DummyOpConcat(OpConcat):
    ConvolutionOp = DummyOpConv
    StopMaskForwardOp = DummyOpStopMaskForward
    InputOp = DummyOpInput
    additional_types = ['concat']


def test_stop_ops_elementwise_source_before_concat():
    graph = NNCFGraph()
    stop_op_0 = graph.add_nncf_node('stop_op_0', 'stop_prop_op', DummyStopPropOp)
    stop_op_1 = graph.add_nncf_node('stop_op_1', 'stop_prop_op', DummyStopPropOp)
    elementwise_node = graph.add_nncf_node('elementwise_node', 'elementwise', DummyElementwise)
    concat_node = graph.add_nncf_node('concat_node', 'concat', DummyConcatMetatype)

    # stop_op_0 -> elementwise_node
    graph.add_edge_between_nncf_nodes(from_node_id=stop_op_0.node_id,
                                      to_node_id=elementwise_node.node_id,
                                      tensor_shape=[10, 10],
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # stop_op_1 -> elementwise_node
    graph.add_edge_between_nncf_nodes(from_node_id=stop_op_1.node_id,
                                      to_node_id=elementwise_node.node_id,
                                      tensor_shape=[10, 10],
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # elementwise_node -> concat_node
    graph.add_edge_between_nncf_nodes(from_node_id=elementwise_node.node_id,
                                      to_node_id=concat_node.node_id,
                                      tensor_shape=[10, 10],
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)

    assert not DummyOpConcat.check_concat(concat_node, graph)
    DummyOpConcat.mask_propagation(concat_node, graph)
    assert concat_node.data['output_mask'] is None


def test_convs_elementwise_source_before_concat():
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node('conv_op_0', 'conv', DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node('conv_op_1', 'conv', DummyConvMetatype)
    elementwise_node = graph.add_nncf_node('elementwise_node', 'elementwise', DummyElementwise)
    concat_node = graph.add_nncf_node('concat_node', 'concat', DummyConcatMetatype)

    # conv_op_0 -> elementwise_node
    graph.add_edge_between_nncf_nodes(from_node_id=conv_op_0.node_id,
                                      to_node_id=elementwise_node.node_id,
                                      tensor_shape=[10] * 4,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # conv_op_1 -> elementwise_node
    graph.add_edge_between_nncf_nodes(from_node_id=conv_op_1.node_id,
                                      to_node_id=elementwise_node.node_id,
                                      tensor_shape=[10] * 4,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # elementwise_node -> concat_node
    graph.add_edge_between_nncf_nodes(from_node_id=elementwise_node.node_id,
                                      to_node_id=concat_node.node_id,
                                      tensor_shape=[10] * 4,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)

    # Check without masks
    assert DummyOpConcat.check_concat(concat_node, graph)
    # Set masks
    conv_op_0 = graph.get_node_by_id(conv_op_0.node_id)
    conv_op_1 = graph.get_node_by_id(conv_op_1.node_id)
    elementwise_node = graph.get_node_by_id(elementwise_node.node_id)
    conv_op_0.data['output_mask'] = np.ones(10)
    conv_op_1.data['output_mask'] = np.ones(10)
    # Propagate masks
    DummyOpElementwise.mask_propagation(elementwise_node, graph)
    # Check with masks
    assert DummyOpConcat.check_concat(concat_node, graph)
    DummyOpConcat.mask_propagation(concat_node, graph)
    reference_mask = []
    assert concat_node.data['output_mask'] is None
#@pytest.mark.parametrize(('node_type', 'input_shape', 'output_shape', 'output_mask', 'output_mask_ref'),
#                         [input + ref for input, ref in zip(TEST_CASES, REF_OUTPUT_MASK)])
#def test_reshape_metatype_mask_prop(node_type, input_shape, output_shape, output_mask, output_mask_ref):
#    node_name = 'dummy_reshape'
#    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)
#
#    graph = NNCFGraph()
#    prev_node = graph.add_nncf_node('prev_node', 'linear', DummyLinearMetatype)
#    reshape_node = graph.add_nncf_node(node_name, node_type, ReshapeMetatype, layer_attributes=layer_attributes)
#
#    graph.add_edge_between_nncf_nodes(from_node_id=prev_node.node_id,
#                                      to_node_id=reshape_node.node_id,
#                                      tensor_shape=output_shape,
#                                      input_port_id=0,
#                                      output_port_id=0,
#                                      dtype=Dtype.FLOAT)
#    # Get reference to graph node
#    prev_node = graph.get_node_by_id(prev_node.node_id)
#    reshape_node = graph.get_node_by_id(reshape_node.node_id)
#    prev_node.data['output_mask'] = output_mask
#    if output_mask_ref == 'error':
#        with pytest.raises(AssertionError):
#            PTReshape.mask_propagation(reshape_node, graph)
#    else:
#        PTReshape.mask_propagation(reshape_node, graph)
#        assert torch.all(reshape_node.data['output_mask'] == output_mask_ref)
#