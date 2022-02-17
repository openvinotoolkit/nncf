from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import LinearMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import SigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import GlobalAveragePoolMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import AddLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import MulLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConcatLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import BatchNormMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ResizeMetatype

from nncf.common.graph.operator_metatypes import UnknownMetatype

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        ConvolutionMetatype,
        LinearMetatype,
        GlobalAveragePoolMetatype,
        AddLayerMetatype,
        MulLayerMetatype,
        BatchNormMetatype,
        ResizeMetatype,  # ????
    ],
    QuantizationTrait.NON_QUANTIZABLE: [SigmoidMetatype,
                                        UnknownMetatype],
    QuantizationTrait.CONCAT: [ConcatLayerMetatype],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}
