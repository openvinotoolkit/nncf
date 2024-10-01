from typing import Tuple

import pytest
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes, LinearLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTConstNoopMetatype, PTConv2dMetatype, PTDepthwiseConv2dSubtype, PTLinearMetatype
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.test_min_max import TemplateTestGetChannelAxes, TemplateTestGetTargetPointShape, TemplateTestMinMaxAlgorithm


class TestTorchFXMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @property
    def backend(self) -> MinMaxAlgoBackend:
        return FXMinMaxAlgoBackend

    @property
    def conv_metatype(self):
        return PTConv2dMetatype

    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> PTTargetPoint:
        if target_point_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        return PTTargetPoint(target_point_type, name, input_port_id=port_id)
    
class TestTorchFXGetTargetPointShape(TemplateTestGetTargetPointShape, TestTorchFXMinMaxAlgorithm):
    def get_nncf_graph(self, weight_port_id: int, weight_shape: Tuple[int]) -> NNCFGraph:
        return NNCFGraphToTest(
            conv_metatype=PTConv2dMetatype,
            nncf_graph_cls=PTNNCFGraph,
            const_metatype=PTConstNoopMetatype
        ).nncf_graph
        
class TestTorchFXGetChannelAxes(TemplateTestGetChannelAxes, TestTorchFXMinMaxAlgorithm):
    @property
    def depthwiseconv_metatype(self):
        return PTDepthwiseConv2dSubtype

    @property
    def matmul_metatype(self):
        return PTLinearMetatype
    
    @staticmethod
    def get_conv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ConvolutionLayerAttributes:
        pass

    @staticmethod
    def get_depthwiseconv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ConvolutionLayerAttributes:
        pass

    
    @staticmethod
    def get_matmul_node_attrs(weight_port_id: int, transpose_weight: Tuple[int], weight_shape: Tuple[int]):
        pass
    
    def test_get_channel_axes_matmul_node_ov_onnx(self):
        pytest.skip("Test is not applied for Torch FX backend.")

    def test_get_channel_axes_deptwiseconv_node_ov(self):
        pytest.skip("Test is not applied for Torch FX backend.")