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

from nncf.common.graph.graph_matching import GraphPattern
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVBatchNormMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVReluMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVEluMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVPReluMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSigmoidMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVHardSigmoidMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVDivMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVSubMetatype

LINEAR_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVConvolutionMetatype,
                                                  OVConvolutionBackpropDataMetatype,
                                                  OVMatMulMetatype
                                                  ],
                     GraphPattern.LABEL_ATTR: 'LINEAR'}

BATCH_NORMALIZATION_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVBatchNormMetatype],
                                  GraphPattern.LABEL_ATTR: 'BATCH_NORMALIZATION'}

ATOMIC_ACTIVATIONS_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVReluMetatype,
                                                              OVEluMetatype,
                                                              OVPReluMetatype,
                                                              OVSigmoidMetatype,
                                                              OVHardSigmoidMetatype,
                                                              ],
                                   GraphPattern.LABEL_ATTR: 'ATOMIC_ACTIVATIONS'}

ARITHMETIC_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVAddMetatype,
                                                      OVSubMetatype,
                                                      OVMulMetatype,
                                                      OVDivMetatype,
                                                      ],
                         GraphPattern.LABEL_ATTR: 'ARITHMETIC'}

TRANSPOSED_OPERATIONS = {GraphPattern.METATYPE_ATTR: [OVConvolutionBackpropDataMetatype],
                         GraphPattern.LABEL_ATTR: 'CONVOLUTION_BACKPROP_DATA'}
