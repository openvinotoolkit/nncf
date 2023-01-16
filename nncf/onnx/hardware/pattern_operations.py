"""
 Copyright (c) 2023 Intel Corporation
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
from nncf.common.graph.graph_matching import GraphPattern
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXBatchNormMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXReluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXLeakyReluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXThresholdedReluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXEluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXPReluMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXHardSigmoidMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXHardSwishMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMulLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDivLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXSubMetatype

LINEAR_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ONNXConvolutionMetatype,
                                                  ONNXDepthwiseConvolutionMetatype,
                                                  ONNXConvolutionTransposeMetatype,
                                                  ONNXLinearMetatype,
                                                  ONNXMatMulMetatype
                                                  ],
                     GraphPattern.LABEL_ATTR: 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ONNXBatchNormMetatype],
                                  GraphPattern.LABEL_ATTR: 'BATCH_NORMALIZATION'}

RELU_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ONNXReluMetatype,
                                                ONNXLeakyReluMetatype,
                                                ONNXThresholdedReluMetatype
                                                ],
                   GraphPattern.LABEL_ATTR: 'RELU'}

NON_RELU_ACTIVATIONS_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ONNXEluMetatype,
                                                                ONNXPReluMetatype,
                                                                ONNXSigmoidMetatype,
                                                                ONNXHardSigmoidMetatype,
                                                                ONNXHardSwishMetatype
                                                                ],
                                   GraphPattern.LABEL_ATTR: 'NON_RELU_ACTIVATIONS'}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(RELU_OPERATIONS,
                                                              NON_RELU_ACTIVATIONS_OPERATIONS,
                                                              'ATOMIC_ACTIVATIONS')

ARITHMETIC_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ONNXAddLayerMetatype,
                                                      ONNXSubMetatype,
                                                      ONNXMulLayerMetatype,
                                                      ONNXDivLayerMetatype,
                                                      ],
                         GraphPattern.LABEL_ATTR: 'ARITHMETIC'}
