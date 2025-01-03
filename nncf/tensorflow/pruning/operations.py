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

from typing import Dict, List

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.pruning.operations import BatchNormPruningOp
from nncf.common.pruning.operations import ConcatPruningOp
from nncf.common.pruning.operations import ConvolutionPruningOp
from nncf.common.pruning.operations import ElementwisePruningOp
from nncf.common.pruning.operations import FlattenPruningOp
from nncf.common.pruning.operations import IdentityMaskForwardPruningOp
from nncf.common.pruning.operations import InputPruningOp
from nncf.common.pruning.operations import LinearPruningOp
from nncf.common.pruning.operations import OutputPruningOp
from nncf.common.pruning.operations import ReshapePruningOp
from nncf.common.pruning.operations import SplitPruningOp
from nncf.common.pruning.operations import StopMaskForwardPruningOp
from nncf.common.pruning.operations import TransposeConvolutionPruningOp
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.graph.metatypes import tf_ops as op_metatypes
from nncf.tensorflow.graph.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import KERAS_ACTIVATIONS_OPERATIONS
from nncf.tensorflow.graph.pattern_operations import TF_ACTIVATIONS_OPERATIONS

TF_PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


def _get_types(operations_dict: Dict) -> List[str]:
    return operations_dict["type"]


@TF_PRUNING_OPERATOR_METATYPES.register("model_input")
class TFInputPruningOp(InputPruningOp):
    additional_types = ["InputLayer", NNCFGraphNodeType.INPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register("model_output")
class TFOutputPruningOp(OutputPruningOp):
    additional_types = [NNCFGraphNodeType.OUTPUT_NODE]


@TF_PRUNING_OPERATOR_METATYPES.register("identity_mask_propagation")
class TFIdentityMaskForwardPruningOp(IdentityMaskForwardPruningOp):
    additional_types = (
        _get_types(KERAS_ACTIVATIONS_OPERATIONS)
        + _get_types(TF_ACTIVATIONS_OPERATIONS)
        + layer_metatypes.TFAveragePooling2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFGlobalAveragePooling2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFMaxPooling2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFGlobalMaxPooling2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFZeroPadding2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFUpSampling2DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFAveragePooling3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFGlobalAveragePooling3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFMaxPooling3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFGlobalMaxPooling3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFZeroPadding3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFUpSampling3DLayerMetatype.get_all_aliases()
        + layer_metatypes.TFDropoutLayerMetatype.get_all_aliases()
        + op_metatypes.TFIdentityOpMetatype.get_all_aliases()
        + op_metatypes.TFPadOpMetatype.get_all_aliases()
    )


@TF_PRUNING_OPERATOR_METATYPES.register("convolution")
class TFConvolutionPruningOp(ConvolutionPruningOp):
    additional_types = ["Conv1D", "Conv2D", "Conv3D", "DepthwiseConv2D"]


@TF_PRUNING_OPERATOR_METATYPES.register("transpose_convolution")
class TFTransposeConvolutionPruningOp(TransposeConvolutionPruningOp):
    additional_types = ["Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose"]


@TF_PRUNING_OPERATOR_METATYPES.register("linear")
class TFLinearPruningOp(LinearPruningOp):
    additional_types = (
        layer_metatypes.TFDenseLayerMetatype.get_all_aliases() + op_metatypes.TFMatMulOpMetatype.get_all_aliases()
    )


@TF_PRUNING_OPERATOR_METATYPES.register("batch_norm")
class TFBatchNormPruningOp(BatchNormPruningOp):
    additional_types = ["BatchNormalization", "SyncBatchNormalization"]


@TF_PRUNING_OPERATOR_METATYPES.register("elementwise")
class TFElementwisePruningOp(ElementwisePruningOp):
    additional_types = _get_types(ELEMENTWISE_OPERATIONS)


@TF_PRUNING_OPERATOR_METATYPES.register("reshape")
class TFReshapeOps(ReshapePruningOp):
    additional_types = op_metatypes.TFReshapeOpMetatype.get_all_aliases()


@TF_PRUNING_OPERATOR_METATYPES.register("flatten")
class TFFlattenOps(FlattenPruningOp):
    additional_types = layer_metatypes.TFFlattenLayerMetatype.get_all_aliases()


@TF_PRUNING_OPERATOR_METATYPES.register("stop_propagation_ops")
class TFStopMaskForwardPruningOp(StopMaskForwardPruningOp):
    additional_types = []


@TF_PRUNING_OPERATOR_METATYPES.register("concat")
class TFConcatPruningOp(ConcatPruningOp):
    additional_types = (
        layer_metatypes.TFConcatenateLayerMetatype.get_all_aliases() + op_metatypes.TFConcatOpMetatype.get_all_aliases()
    )


@TF_PRUNING_OPERATOR_METATYPES.register("split")
class TFSplitPruningOp(SplitPruningOp):
    additional_types = op_metatypes.TFSplitOpMetatype.get_all_aliases()
