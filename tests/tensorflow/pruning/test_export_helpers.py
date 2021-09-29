import pytest
import tensorflow as tf

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.tensorflow.pruning.export_helpers import TFReshapeOps
from nncf.tensorflow.pruning.export_helpers import TFFlattenOps
from nncf.tensorflow.graph.metatypes.keras_layers import TFDenseLayerMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import TFReshapeLayerMetatype
from nncf.tensorflow.graph.metatypes.keras_layers import TFFlattenLayerMetatype


TEST_CASES = [
    ['flatten', (1, 1, 64), (1,), True],
    ['flatten', (1, 32, 64), (1,), True],
    ['flatten', (1, 32, 64, 1), (1,), True],
    ['reshape', (1, 32, 64), (1,), True], # Flatten
    ['reshape', (1, 64, 32), (1,), True], # Flatten
    ['reshape', (1, 32, 64, 1), (1,), True], # Flatten
    ['reshape', (1, 1, 64), (1, 1, 1, 64), True], # Expand
    ['reshape', (1, 1, 1, 64), (1, 64), True], # Squeeze
    ['reshape', (1, 1, 1, 64), (1, 1, 64, 1), True],# Transpose
    ['reshape', (1, 1, 32, 64), (1, 64, 32), True],# Transpose
    ['reshape', (1, 1, 32, 64), (1, 64, 16, 16), False],
]


METATYPES_MAP = {
    'flatten': {'metatype': TFFlattenLayerMetatype,
                'ops': TFFlattenOps},
    'reshape': {'metatype': TFReshapeLayerMetatype,
                'ops': TFReshapeOps}
}


@pytest.mark.parametrize(('node_type', 'input_shape', 'output_shape', 'ref_accept_pruned_input'),
                         TEST_CASES)
def test_reshape_accept_pruned_input(node_type, input_shape, output_shape, ref_accept_pruned_input):
    node_name = 'dummy_reshape'
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)
    graph = NNCFGraph()
    node = graph.add_nncf_node(node_name, node_type, METATYPES_MAP[node_type]['metatype'],
                               layer_attributes=layer_attributes)

    actual_accept_pruned_input = METATYPES_MAP[node_type]['ops'].accept_pruned_input(node)
    assert ref_accept_pruned_input == actual_accept_pruned_input


REF_OUTPUT_MASK_RESHAPE = [
    [tf.ones(10), 'error'],
    [tf.Variable([0, 1] * 32), tf.reshape(tf.Variable([[0] * 32 + [1] * 32] * 32), -1)],
    [tf.Variable([0, 1] * 32), tf.reshape(tf.Variable([[0] * 32 + [1] * 32] * 32), -1)]
] * 2 + [[tf.Variable([0, 1] * 32)] * 2] * 4


@pytest.mark.parametrize(('node_type', 'input_shape', 'output_shape', 'output_mask', 'output_mask_ref'),
                         [case[:-1] + mask for case, mask in zip(TEST_CASES, REF_OUTPUT_MASK_RESHAPE)])
def test_reshape_metatype_mask_prop(node_type, input_shape, output_shape, output_mask, output_mask_ref):
    node_name = 'dummy_reshape'
    layer_attributes = ReshapeLayerAttributes(input_shape, output_shape)

    graph = NNCFGraph()
    prev_node = graph.add_nncf_node('prev_node', 'linear', TFDenseLayerMetatype)
    reshape_node = graph.add_nncf_node(node_name, node_type, METATYPES_MAP[node_type]['metatype'],
                                       layer_attributes=layer_attributes)

    graph.add_edge_between_nncf_nodes(from_node_id=prev_node.node_id,
                                      to_node_id=reshape_node.node_id,
                                      tensor_shape=output_shape,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    # Check both None mask and not None mask
    for output_mask_cur, output_mask_ref_cur in ([(None, None), (output_mask, output_mask_ref)]):
        # Get reference to graph node
        prev_node = graph.get_node_by_id(prev_node.node_id)
        reshape_node = graph.get_node_by_id(reshape_node.node_id)
        prev_node.data['output_mask'] = output_mask_cur
        if isinstance(output_mask_ref_cur, str):
            with pytest.raises(AssertionError):
                TFReshapeOps.mask_propagation(reshape_node, graph)
        else:
            TFReshapeOps.mask_propagation(reshape_node, graph)
            assert tf.reduce_all(reshape_node.data['output_mask'] == output_mask_ref_cur)


@pytest.mark.parametrize('node_type', ['reshape', 'flatten'])
def test_reshape_is_last_op(node_type):
    node_name = 'dummy_reshape'
    layer_attributes = None

    graph = NNCFGraph()
    prev_node = graph.add_nncf_node('prev_node', 'linear', TFDenseLayerMetatype)
    reshape_node = graph.add_nncf_node(node_name, node_type, METATYPES_MAP[node_type]['metatype'],
                                       layer_attributes=layer_attributes)

    assert not METATYPES_MAP[node_type]['ops'].accept_pruned_input(reshape_node)

    graph.add_edge_between_nncf_nodes(from_node_id=prev_node.node_id,
                                      to_node_id=reshape_node.node_id,
                                      tensor_shape=[1, 32],
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)

    for output_mask in (None, tf.ones((10,))):
        prev_node = graph.get_node_by_id(prev_node.node_id)
        reshape_node = graph.get_node_by_id(reshape_node.node_id)
        prev_node.data['output_mask'] = output_mask
        METATYPES_MAP[node_type]['ops'].mask_propagation(reshape_node, graph)
        assert reshape_node.data['output_mask'] is None
