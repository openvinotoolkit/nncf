"""
 Copyright (c) 2020 Intel Corporation
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

from texttable import Texttable

import tensorflow as tf

from beta.nncf.tensorflow.graph.model_transformer import ModelTransformer
from beta.nncf.tensorflow.graph.transformations.commands import LayerWeightOperation
from beta.nncf.tensorflow.graph.transformations.commands import RemovalCommand
from beta.nncf.tensorflow.graph.transformations.layout import TransformationLayout
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMask


def convert_raw_to_printable(raw_sparsity_statistics):
    sparsity_statistics = {}
    sparsity_statistics.update(raw_sparsity_statistics)

    table = Texttable()
    header = ['Name', 'Weight\'s Shape', 'SR', '% weights']
    data = [header]

    for sparsity_info in raw_sparsity_statistics['sparsity_statistic_by_module']:
        row = [sparsity_info[h] for h in header]
        data.append(row)
    table.add_rows(data)
    sparsity_statistics['sparsity_statistic_by_module'] = table
    return sparsity_statistics


def prepare_for_tensorboard(raw_sparsity_statistics):
    sparsity_statistics = {}
    base_prefix = '2.compression/statistics/'
    detailed_prefix = '3.compression_details/statistics/'
    for key, value in raw_sparsity_statistics.items():
        if key == 'sparsity_statistic_by_module':
            for v in value:
                sparsity_statistics[detailed_prefix + v['Name'] + '/sparsity'] = v['SR']
        else:
            sparsity_statistics[base_prefix + key] = value

    return sparsity_statistics


def strip_model_from_masks(model: tf.keras.Model) -> tf.keras.Model:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            'Expected model to be a `tf.keras.Model` instance but got: ', model)

    transformations = TransformationLayout()

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            for weight_attr, ops in layer.weights_attr_ops.items():
                # BinaryMask operation must be the first operation
                op_name, op = next(iter(ops.items()))
                if isinstance(op, BinaryMask):
                    apply_mask(layer, weight_attr, op_name)

                    transformations.register(
                        RemovalCommand(
                            target_point=LayerWeightOperation(
                                layer.name,
                                weights_attr_name=weight_attr,
                                operation_name=op_name)
                        ))

    return ModelTransformer(model, transformations).transform()


def apply_mask(wrapped_layer: NNCFWrapper, weight_attr: str, op_name: str):
    layer_weight = wrapped_layer.layer_weights[weight_attr]
    op = wrapped_layer.weights_attr_ops[weight_attr][op_name]
    layer_weight.assign(
        op(layer_weight,
           wrapped_layer.ops_weights[op_name],
           False)
    )
    wrapped_layer.set_layer_weight(weight_attr, layer_weight)
