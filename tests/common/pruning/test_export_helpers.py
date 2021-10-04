import numpy as np
import pytest

from typing import List
from functools import partial

from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.graph import NNCFGraph
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.export_helpers import (
    OpElementwise,
    OpConvolution,
    OpConcat,
    OpStopMaskForwardOps,
)


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


DUMMY_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


@DUMMY_PRUNING_OPERATOR_METATYPES.register('input')
class DummyOpInput(OpConcat):
    additional_types = ['input']


@DUMMY_PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class DummyOpStopMaskForward(OpStopMaskForwardOps):
    additional_types = ['stop_prop_op']


@DUMMY_PRUNING_OPERATOR_METATYPES.register('conv')
class DummyOpConv(OpConvolution):
    additional_types = ['conv']


@DUMMY_PRUNING_OPERATOR_METATYPES.register('elementwise')
class DummyOpElementwise(OpElementwise):
    additional_types = ['elementwise']


@DUMMY_PRUNING_OPERATOR_METATYPES.register('concat')
class DummyOpConcat(OpConcat):
    ConvolutionOp = DummyOpConv
    StopMaskForwardOp = DummyOpStopMaskForward
    InputOp = DummyOpInput
    additional_types = ['concat']


@pytest.mark.parametrize('with_elementwise', [False, True])
def test_stop_ops_elementwise_source_before_concat(with_elementwise):
    graph = NNCFGraph()
    stop_op_0 = graph.add_nncf_node('stop_op_0', 'stop_prop_op', DummyStopPropOp)
    stop_op_1 = graph.add_nncf_node('stop_op_1', 'stop_prop_op', DummyStopPropOp)
    concat_layer_attributes = MultipleInputLayerAttributes(-1)
    concat_node = graph.add_nncf_node('concat_node', 'concat', DummyConcatMetatype,
                                      layer_attributes=concat_layer_attributes)
    add_node = partial(graph.add_edge_between_nncf_nodes,
                       tensor_shape=[10, 10],
                       input_port_id=0,
                       output_port_id=0,
                       dtype=Dtype.FLOAT)

    if not with_elementwise:
        # stop_op_0 -> concat_node
        add_node(from_node_id=stop_op_0.node_id,
                 to_node_id=concat_node.node_id)

        # stop_op_1 -> concat_node
        add_node(from_node_id=stop_op_1.node_id,
                 to_node_id=concat_node.node_id)
    else:
        elementwise_op = graph.add_nncf_node('elementwise', 'elementwise', DummyElementwise)

        # stop_op_0 -> elementwise
        add_node(from_node_id=stop_op_0.node_id,
                 to_node_id=elementwise_op.node_id)

        # stop_op_1 -> elementwise
        add_node(from_node_id=stop_op_1.node_id,
                 to_node_id=elementwise_op.node_id)

        # elementwise -> concat
        add_node(from_node_id=elementwise_op.node_id,
                 to_node_id=concat_node.node_id)

    assert not DummyOpConcat.check_concat(concat_node, graph)
    MaskPropagationAlgorithm(graph, DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
    concat_node = graph.get_node_by_id(concat_node.node_id)
    assert concat_node.data['output_mask'] is None


@pytest.mark.parametrize('empty_mask_branch', [False, True])
def test_convs_elementwise_source_before_concat(empty_mask_branch):
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node('conv_op_0', 'conv', DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node('conv_op_1', 'conv', DummyConvMetatype)
    conv_op_2 = graph.add_nncf_node('conv_op_2', 'conv', DummyConvMetatype)
    elementwise_node = graph.add_nncf_node('elementwise_node', 'elementwise', DummyElementwise)
    concat_layer_attributes = MultipleInputLayerAttributes(2)
    concat_node = graph.add_nncf_node('concat_node', 'concat', DummyConcatMetatype,
                                      layer_attributes=concat_layer_attributes)
    add_node = partial(graph.add_edge_between_nncf_nodes,
                       tensor_shape=[10] * 4,
                       input_port_id=0,
                       output_port_id=0,
                       dtype=Dtype.FLOAT)

    # conv_op_0 -> elementwise_node
    add_node(from_node_id=conv_op_0.node_id,
             to_node_id=elementwise_node.node_id)

    # conv_op_1 -> elementwise_node
    add_node(from_node_id=conv_op_1.node_id,
             to_node_id=elementwise_node.node_id)

    # elementwise_node -> concat_node
    add_node(from_node_id=elementwise_node.node_id,
             to_node_id=concat_node.node_id)

    # conv_op_2 -> concat_node
    add_node(from_node_id=conv_op_2.node_id,
             to_node_id=concat_node.node_id)

    # Check without masks
    assert DummyOpConcat.check_concat(concat_node, graph)
    # Set masks
    masked_convs = [conv_op_0, conv_op_1]
    if not empty_mask_branch:
        masked_convs.append(conv_op_2)

    for conv_op in masked_convs:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.data['output_mask'] = np.ones(10)

    # Propagate masks
    MaskPropagationAlgorithm(graph, DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
    # Check with masks
    concat_node = graph.get_node_by_id(concat_node.node_id)
    assert DummyOpConcat.check_concat(concat_node, graph)
    reference_mask = np.ones((20,))
    np.testing.assert_equal(concat_node.data['output_mask'], reference_mask)
