from collections import Counter

import pytest
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import BatchNormMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ReluMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GlobalAveragePoolMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConcatLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import AddLayerMetatype

from tests.onnx.models import LinearModel
from tests.onnx.models import MultiInputOutputModel
from tests.onnx.models import ModelWithIntEdges

TEST_MODELS = [LinearModel, MultiInputOutputModel, ModelWithIntEdges]
REF_METATYPES_COUNTERS = [
    [InputNoopMetatype, ConvolutionMetatype, BatchNormMetatype, ReluMetatype, GlobalAveragePoolMetatype,
     ConvolutionMetatype, OutputNoopMetatype],
    [InputNoopMetatype, InputNoopMetatype, InputNoopMetatype,
     ConcatLayerMetatype, AddLayerMetatype, OutputNoopMetatype, OutputNoopMetatype],
    [InputNoopMetatype, UnknownMetatype, UnknownMetatype, OutputNoopMetatype]]


@pytest.mark.parametrize(("model_creator_func, ref_metatypes"),
                         zip(TEST_MODELS, REF_METATYPES_COUNTERS))
def test_mapping_onnx_metatypes(model_creator_func, ref_metatypes):
    model = model_creator_func()
    nncf_graph = GraphConverter.create_nncf_graph(model.onnx_model)
    actual_metatypes = [node.metatype for node in nncf_graph.get_all_nodes()]
    assert Counter(ref_metatypes) == Counter(actual_metatypes)
