"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.common.graph.patterns import merge_two_types_of_operations
from nncf.common.graph.graph import NNCFGraph
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXLinearMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXReluMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXLeakyReluMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXDivLayerMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype

LINEAR_OPERATIONS = {'type': [ONNXConvolutionMetatype,
                              ONNXConvolutionTransposeMetatype,
                              ONNXLinearMetatype
                              ],
                     'label': 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {'type': [ONNXBatchNormMetatype],
                                  'label': 'BATCH_NORMALIZATION'}

RELU_OPERATIONS = {'type': [ONNXReluMetatype,
                            ONNXLeakyReluMetatype,
                            ],
                   'label': 'RELU'}

NON_RELU_ACTIVATIONS_OPERATIONS = {'type': [
    ONNXSigmoidMetatype,
    ONNXHardSigmoidMetatype,
],
    'label': 'NON_RELU_ACTIVATIONS'}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                              NON_RELU_ACTIVATIONS_OPERATIONS,
                                                              'ATOMIC_ACTIVATIONS')

ARITHMETIC_OPERATIONS = {'type': [ONNXAddLayerMetatype,
                                  ONNXSubMetatype,
                                  ONNXMulLayerMetatype,
                                  ONNXDivLayerMetatype,
                                  ],
                         'label': 'ARITHMETIC'}

MATMUL_OPERATIONS = {'type': [ONNXMatMulMetatype
                              ],
                     'label': 'MATMUL'}
