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

import os
import tensorflow as tf

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

SAVEDMODEL_FORMAT = 'tf'
KERAS_H5_FORMAT = 'h5'
FROZEN_GRAPH_FORMAT = 'frozen_graph'


def keras_model_to_frozen_graph(model):
    input_signature = []
    for item in model.inputs:
        input_signature.append(tf.TensorSpec(item.shape, item.dtype, item.name))
    concrete_function = tf.function(model).get_concrete_function(input_signature)
    frozen_func = convert_variables_to_constants_v2(concrete_function, lower_control_flow=False)
    return frozen_func.graph.as_graph_def(add_shapes=True)


def save_model_as_frozen_graph(model, save_path, as_text=False):
    frozen_graph = keras_model_to_frozen_graph(model)
    save_dir, name = os.path.split(save_path)
    tf.io.write_graph(frozen_graph, save_dir, name, as_text=as_text)


def save_model(model, save_path, save_format=FROZEN_GRAPH_FORMAT):
    if save_format == FROZEN_GRAPH_FORMAT:
        save_model_as_frozen_graph(model, save_path)
    else:
        model.save(save_path, save_format=save_format)
        if save_format == SAVEDMODEL_FORMAT:
            model = tf.saved_model.load(save_path)
            tf.saved_model.save(model, save_path)
