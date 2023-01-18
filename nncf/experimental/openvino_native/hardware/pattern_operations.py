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

from nncf.common.graph.graph_matching import GraphPattern
from nncf.experimental.openvino_native.graph.metatypes import openvino_metatypes as om

LINEAR_OPERATIONS = {GraphPattern.METATYPE_ATTR: [om.OVConvolutionMetatype,
                                                  om.OVConvolutionBackpropDataMetatype,
                                                  om.OVMatMulMetatype
                                                  ],
                     GraphPattern.LABEL_ATTR: 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {GraphPattern.METATYPE_ATTR: [om.OVBatchNormMetatype],
                                  GraphPattern.LABEL_ATTR: 'BATCH_NORMALIZATION'}

ATOMIC_ACTIVATIONS_OPERATIONS = {GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                              om.OVEluMetatype,
                                                              om.OVPReluMetatype,
                                                              om.OVSigmoidMetatype,
                                                              om.OVHardSigmoidMetatype,
                                                              ],
                                   GraphPattern.LABEL_ATTR: 'ATOMIC_ACTIVATIONS'}

ARITHMETIC_OPERATIONS = {GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                      om.OVSubMetatype,
                                                      om.OVMulMetatype,
                                                      om.OVDivMetatype,
                                                      ],
                         GraphPattern.LABEL_ATTR: 'ARITHMETIC'}

ELEMENTWISE_OPERATIONS = {GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                       om.OVMulMetatype,
                                                       om.OVSubMetatype,
                                                       om.OVDivMetatype,
                                                       om.OVLessMetatype,
                                                       om.OVLessEqualMetatype,
                                                       om.OVGreaterMetatype,
                                                       om.OVGreaterEqualMetatype,
                                                       om.OVEqualMetatype,
                                                       om.OVNotEqualMetatype,
                                                       om.OVFloorModMetatype,
                                                       om.OVOrMetatype,
                                                       om.OVXorMetatype,
                                                       om.OVAndMetatype,
                                                       om.OVMaximumMetatype,
                                                       om.OVMinimumMetatype,
                                                       ],
                          GraphPattern.LABEL_ATTR: 'ELEMENTWISE'}
