import pytest
import torch

from nncf.torch.graph.operator_metatypes import LinearMetatype
from nncf.torch.graph.operator_metatypes import ReshapeMetatype
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.torch.pruning.export_helpers import PTReshape


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


@pytest.mark.parametrize(('node_type', 'input_shape', 'output_shape', 'ref_accept_pruned_input'),
                         [input + [ref] for input, ref in zip(TEST_CASES, REF_ACCEPT_PRUNED)])
def test_reshape_accept_pruned_input(node_type, input_shape, output_shape, ref_accept_pruned_input):
    node_name = 'dummy_reshape'
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)
    graph = PTNNCFGraph()
    node = graph.add_nncf_node(node_name, node_type, ReshapeMetatype, layer_attributes=layer_attributes)
    actual_accept_pruned_input = PTReshape.accept_pruned_input(node)
    assert ref_accept_pruned_input == actual_accept_pruned_input


REF_OUTPUT_MASK = [
    [torch.ones(10), 'error'],
    [torch.tensor([0, 1]).repeat(32), torch.tensor([[0] * 32 + [1] * 32] * 32).view(-1)],
    [torch.tensor([0, 1]).repeat(32), torch.tensor([[0] * 32 + [1] * 32] * 32).view(-1)],
    [torch.tensor([0, 1]).repeat(32)] * 2,
]


@pytest.mark.parametrize(('node_type', 'input_shape', 'output_shape', 'output_mask', 'output_mask_ref'),
                         [input + ref for input, ref in zip(TEST_CASES, REF_OUTPUT_MASK)])
def test_reshape_metatype_mask_prop(node_type, input_shape, output_shape, output_mask, output_mask_ref):
    node_name = 'dummy_reshape'
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)

    graph = PTNNCFGraph()
    prev_node = graph.add_nncf_node('prev_node', 'linear', LinearMetatype)
    reshape_node = graph.add_nncf_node(node_name, node_type, ReshapeMetatype, layer_attributes=layer_attributes)

    graph.add_edge_between_nncf_nodes(from_node_id=prev_node.node_id,
                                      to_node_id=reshape_node.node_id,
                                      tensor_shape=output_shape,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # Get reference to graph node
    prev_node = graph.get_node_by_id(prev_node.node_id)
    reshape_node = graph.get_node_by_id(reshape_node.node_id)
    prev_node.data['output_mask'] = output_mask
    if output_mask_ref == 'error':
        with pytest.raises(AssertionError):
            PTReshape.mask_propagation(reshape_node, graph)
    else:
        PTReshape.mask_propagation(reshape_node, graph)
        assert torch.all(reshape_node.data['output_mask'] == output_mask_ref)
