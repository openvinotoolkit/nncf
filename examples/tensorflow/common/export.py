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
import os.path as osp

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from examples.tensorflow.common.utils import FROZEN_GRAPH_FORMAT


def export_model(model: tf.keras.Model, save_path: str, save_format: str) -> None:
    """
    Export compressed model. Supported types 'tf', 'h5', 'frozen_graph'.

    :param model: Target model.
    :param save_path: Path to save.
    :param save_format: Model format used to save model.
    """

    if save_format == FROZEN_GRAPH_FORMAT:
        input_signature = []
        for item in model.inputs:
            input_signature.append(tf.TensorSpec(item.shape, item.dtype, item.name))
        concrete_function = tf.function(model).get_concrete_function(input_signature)
        frozen_func = convert_variables_to_constants_v2(concrete_function, lower_control_flow=False)
        frozen_graph = frozen_func.graph.as_graph_def(add_shapes=True)

        save_dir, name = osp.split(save_path)
        tf.io.write_graph(frozen_graph, save_dir, name, as_text=False)
    else:
        model.save(save_path, save_format=save_format)
