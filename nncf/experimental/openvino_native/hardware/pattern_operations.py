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

from nncf.common.graph.patterns import GraphPattern
from nncf.experimental.openvino_native.graph.metatypes import openvino_metatypes as ov_metatypes

LINEAR_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVConvolutionMetatype,
                                                  ov_metatypes.OVGroupConvolutionMetatype,
                                                  ov_metatypes.OVDepthwiseConvolutionMetatype,
                                                  ov_metatypes.OVConvolutionBackpropDataMetatype,
                                                  ov_metatypes.OVGroupConvolutionBackpropDataMetatype,
                                                  ov_metatypes.OVMatMulMetatype
                                                  ],
                     GraphPattern.LABEL_ATTR: 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVBatchNormMetatype],
                                  GraphPattern.LABEL_ATTR: 'BATCH_NORMALIZATION'}

ATOMIC_ACTIVATIONS_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVReluMetatype,
                                                              ov_metatypes.OVClampMetatype,
                                                              ov_metatypes.OVEluMetatype,
                                                              ov_metatypes.OVPReluMetatype,
                                                              ov_metatypes.OVSigmoidMetatype,
                                                              ov_metatypes.OVHardSigmoidMetatype,
                                                              ov_metatypes.OVSwishMetatype,
                                                              ],
                                   GraphPattern.LABEL_ATTR: 'ATOMIC_ACTIVATIONS'}

ARITHMETIC_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVAddMetatype,
                                                      ov_metatypes.OVSubtractMetatype,
                                                      ov_metatypes.OVMultiplyMetatype,
                                                      ov_metatypes.OVDivideMetatype,
                                                      ],
                         GraphPattern.LABEL_ATTR: 'ARITHMETIC'}

ELEMENTWISE_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVAddMetatype,
                                                       ov_metatypes.OVMultiplyMetatype,
                                                       ov_metatypes.OVSubtractMetatype,
                                                       ov_metatypes.OVDivideMetatype,
                                                       ov_metatypes.OVLessMetatype,
                                                       ov_metatypes.OVLessEqualMetatype,
                                                       ov_metatypes.OVGreaterMetatype,
                                                       ov_metatypes.OVGreaterEqualMetatype,
                                                       ov_metatypes.OVEqualMetatype,
                                                       ov_metatypes.OVNotEqualMetatype,
                                                       ov_metatypes.OVFloorModMetatype,
                                                       ov_metatypes.OVLogicalOrMetatype,
                                                       ov_metatypes.OVLogicalXorMetatype,
                                                       ov_metatypes.OVLogicalAndMetatype,
                                                       ov_metatypes.OVMaximumMetatype,
                                                       ov_metatypes.OVMinimumMetatype,
                                                       ],
                          GraphPattern.LABEL_ATTR: 'ELEMENTWISE'}

TRANSPOSED_OPERATIONS = {GraphPattern.METATYPE_ATTR: [ov_metatypes.OVConvolutionBackpropDataMetatype],
                         GraphPattern.LABEL_ATTR: 'CONVOLUTION_BACKPROP_DATA'}
