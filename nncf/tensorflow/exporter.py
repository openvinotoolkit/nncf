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

from typing import Optional
import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from nncf.common.exporter import Exporter


# TODO(andrey-churkin): Add support for `input_names` and `output_names`
class TFExporter(Exporter):
    """
    This class provides export of the compressed model to the Frozen Graph,
    TensorFlow SavedModel, or Keras H5 formats.
    """

    _SAVED_MODEL_FORMAT = 'tf'
    _KERAS_H5_FORMAT = 'h5'
    _FROZEN_GRAPH_FORMAT = 'frozen_graph'

    def export_model(self, save_path: str, save_format: Optional[str] = None) -> None:
        """
        Exports the compressed model to the specified format.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format.
            One of the following:
                - `tf` for export to the Tensorflow SavedModel format.
                - `h5` for export to the Keras H5 format.
                - `frozen_graph` for export to the Frozen Graph format.
            The Frozen Graph format will be used if `save_format` is not specified.
        """
        if save_format is None:
            save_format = TFExporter._FROZEN_GRAPH_FORMAT

        format_to_export_fn = {
            TFExporter._SAVED_MODEL_FORMAT: self._export_to_saved_model,
            TFExporter._KERAS_H5_FORMAT: self._export_to_h5,
            TFExporter._FROZEN_GRAPH_FORMAT: self._export_to_frozen_graph,
        }

        export_fn = format_to_export_fn.get(save_format)

        if export_fn is None:
            available_formats = list(format_to_export_fn.keys())
            raise ValueError(f'Unsupported saving format: \'{save_format}\'. '
                             f'Available formats: {available_formats}')

        export_fn(save_path)

    def _export_to_saved_model(self, save_path: str) -> None:
        """
        Exports the compressed model to the TensorFlow SavedModel format.

        :param save_path: The path where the model will be saved.
        """
        self._model.save(save_path, save_format=TFExporter._SAVED_MODEL_FORMAT)
        model = tf.saved_model.load(save_path)
        tf.saved_model.save(model, save_path)

    def _export_to_h5(self, save_path: str) -> None:
        """
        Exports the compressed model to the Keras H5 format.

        :param save_path: The path where the model will be saved.
        """
        self._model.save(save_path, save_format=TFExporter._KERAS_H5_FORMAT)

    def _export_to_frozen_graph(self, save_path: str) -> None:
        """
        Exports the compressed model to the Frozen Graph format.

        :param save_path: The path where the model will be saved.
        """
        # Convert Keras model to the frozen graph.
        input_signature = []
        for item in self._model.inputs:
            input_signature.append(tf.TensorSpec(item.shape, item.dtype, item.name))
        concrete_function = tf.function(self._model).get_concrete_function(input_signature)
        frozen_func = convert_variables_to_constants_v2(concrete_function, lower_control_flow=False)
        frozen_graph = frozen_func.graph.as_graph_def(add_shapes=True)

        save_dir, name = os.path.split(save_path)
        tf.io.write_graph(frozen_graph, save_dir, name, as_text=False)
