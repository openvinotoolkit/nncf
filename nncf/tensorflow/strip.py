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

from typing import Dict

import tensorflow as tf

from nncf.common.utils.backend import copy_model
from nncf.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from nncf.tensorflow.graph.transformations.commands import TFRemovalCommand
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.quantization.quantizers import AsymmetricQuantizer
from nncf.tensorflow.quantization.quantizers import SymmetricQuantizer
from nncf.tensorflow.quantization.utils import apply_overflow_fix_to_layer
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from nncf.tensorflow.sparsity.utils import apply_mask


def strip(model: tf.keras.Model, do_copy: bool = True) -> tf.keras.Model:
    """
    Implementation of the nncf.strip() function for the TF backend

    :param model: The compressed model.
    :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
      will return the currently associated model object "stripped" in-place.
    :return: The stripped model.
    """
    # Check to understand if the model is after NNCF or not.
    wrapped_layers = collect_wrapped_layers(model)
    if not wrapped_layers:
        return model

    if do_copy:
        model = copy_model(model)
        wrapped_layers = collect_wrapped_layers(model)

    op_to_priority: Dict[NNCFOperation, int] = {
        SymmetricQuantizer: 1,
        AsymmetricQuantizer: 1,
        BinaryMask: 2,
        RBSparsifyingWeight: 2,
    }

    key_fn = lambda op: op_to_priority.get(op, 0)

    transformation_layout = TFTransformationLayout()
    for wrapped_layer in wrapped_layers:
        for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
            for op in sorted(ops.values(), key=key_fn, reverse=True):
                # quantization
                if isinstance(op, (SymmetricQuantizer, AsymmetricQuantizer)) and op.half_range:
                    apply_overflow_fix_to_layer(wrapped_layer, weight_attr, op)
                # sparsity, pruning
                if isinstance(op, (BinaryMask, RBSparsifyingWeight)):
                    apply_mask(wrapped_layer, weight_attr, op)
                    transformation_layout.register(
                        TFRemovalCommand(
                            target_point=TFOperationWithWeights(
                                wrapped_layer.name, weights_attr_name=weight_attr, operation_name=op.name
                            )
                        )
                    )
    if transformation_layout.transformations:
        model = TFModelTransformer(model).transform(transformation_layout)

    return model
