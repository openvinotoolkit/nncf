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

from typing import List

from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait

METATYPES_FOR_TEST = OperatorMetatypeRegistry("TEST_METATYPES")


class TestMetatype(OperatorMetatype):
    name = None

    @classmethod
    def get_all_aliases(cls) -> List[str]:
        assert cls.name is not None
        return [cls.name]


@METATYPES_FOR_TEST.register()
class BatchNormTestMetatype(TestMetatype):
    name = "batch_norm"


@METATYPES_FOR_TEST.register()
class Conv2dTestMetatype(TestMetatype):
    name = "conv2d"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class MatMulTestMetatype(TestMetatype):
    name = "matmul"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class MaxPool2dTestMetatype(TestMetatype):
    name = "max_pool2d"


@METATYPES_FOR_TEST.register()
class GeluTestMetatype(TestMetatype):
    name = "gelu"


@METATYPES_FOR_TEST.register()
class DropoutTestMetatype(TestMetatype):
    name = "dropout"


@METATYPES_FOR_TEST.register()
class MinTestMetatype(TestMetatype):
    name = "min"


@METATYPES_FOR_TEST.register()
class SoftmaxTestMetatype(TestMetatype):
    name = "softmax"


@METATYPES_FOR_TEST.register()
class CatTestMetatype(TestMetatype):
    name = "cat"


@METATYPES_FOR_TEST.register()
class LinearTestMetatype(TestMetatype):
    name = "linear"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class TopKTestMetatype(TestMetatype):
    name = "topk"


@METATYPES_FOR_TEST.register()
class NMSTestMetatype(TestMetatype):
    name = "nms"


@METATYPES_FOR_TEST.register()
class IdentityTestMetatype(TestMetatype):
    name = "identity"


@METATYPES_FOR_TEST.register()
class ReshapeTestMetatype(TestMetatype):
    name = "reshape"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class QuantizerTestMetatype(TestMetatype):
    name = "quantizer"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class ConstantTestMetatype(TestMetatype):
    name = "constant"


@METATYPES_FOR_TEST.register()
class ReluTestMetatype(TestMetatype):
    name = "relu"


@METATYPES_FOR_TEST.register()
class AddTestMetatype(TestMetatype):
    name = "add"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class ShapeOfTestMetatype(TestMetatype):
    name = "shapeof"


@METATYPES_FOR_TEST.register()
class PowerTestMetatype(TestMetatype):
    name = "power"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class MultiplyTestMetatype(TestMetatype):
    name = "multiply"
    num_expected_input_edges = 2


@METATYPES_FOR_TEST.register()
class InterpolateTestMetatype(TestMetatype):
    name = "interpolate"
    num_expected_input_edges = 3


@METATYPES_FOR_TEST.register()
class StridedSliceTestMetatype(TestMetatype):
    name = "strided_slice"


@METATYPES_FOR_TEST.register()
class DivideTestMetatype(TestMetatype):
    name = "divide"


@METATYPES_FOR_TEST.register()
@INPUT_NOOP_METATYPES.register()
class ParameterTestMetatype(TestMetatype):
    name = "parameter"


@METATYPES_FOR_TEST.register()
class FakeQuantizeTestMetatype(TestMetatype):
    name = "fake_quantize"


@METATYPES_FOR_TEST.register()
class QuantizeTestMetatype(TestMetatype):
    name = "quantize"


@METATYPES_FOR_TEST.register()
class DequantizeTestMetatype(TestMetatype):
    name = "dequantize"


@METATYPES_FOR_TEST.register()
class ScaledDotProductAttentionMetatype(TestMetatype):
    name = "scaled_dot_product_attention"
    target_input_ports = [0, 1]


WEIGHT_LAYER_METATYPES = [LinearTestMetatype, Conv2dTestMetatype, MatMulTestMetatype]


DEFAULT_TEST_QUANT_TRAIT_MAP = {
    QuantizationTrait.INPUTS_QUANTIZABLE: [
        BatchNormTestMetatype,
        Conv2dTestMetatype,
        MatMulTestMetatype,
        GeluTestMetatype,
        LinearTestMetatype,
        AddTestMetatype,
        ScaledDotProductAttentionMetatype,
    ],
    QuantizationTrait.CONCAT: [CatTestMetatype],
}


QUANTIZER_METATYPES = [
    FakeQuantizeTestMetatype,
    QuantizeTestMetatype,
    DequantizeTestMetatype,
]


CONSTANT_METATYPES = [
    ConstantTestMetatype,
]


QUANTIZABLE_METATYPES = [
    Conv2dTestMetatype,
    AddTestMetatype,
    MultiplyTestMetatype,
    PowerTestMetatype,
    InterpolateTestMetatype,
    DivideTestMetatype,
]


QUANTIZE_AGNOSTIC_METATYPES = [
    MaxPool2dTestMetatype,
    ReluTestMetatype,
    StridedSliceTestMetatype,
]


SHAPEOF_METATYPES = [
    ShapeOfTestMetatype,
]
