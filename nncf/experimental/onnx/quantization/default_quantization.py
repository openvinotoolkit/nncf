from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_ops import ReluMetatype

DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        ConvolutionMetatype,
        ReluMetatype
    ],
    QuantizationTrait.NON_QUANTIZABLE: [],
    QuantizationTrait.CONCAT: [],
    QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS: []
}