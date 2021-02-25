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

import tensorflow as tf

from beta.examples.tensorflow.common.object_detection import base_model
from beta.examples.tensorflow.common.object_detection.architecture import factory
from beta.examples.tensorflow.common.logger import logger


class YOLOv4Model(base_model.Model):
    """YOLOv4 model function."""
    def __init__(self, params):
        super().__init__(params)

        self._params = params
        self._input_layer = tf.keras.layers.Input(shape=(None, None, 3), name='image_input')

        # Architecture generators.
        self._backbone_fn = factory.backbone_generator(params)
        self._yolo4_predictions_fn = factory.yolo_v4_head_generator()


    def build_outputs(self, inputs, is_training):
        """Create YOLO_V4 model CNN body in Keras."""
        darknet = tf.keras.models.Model(inputs, self._backbone_fn(inputs))

        # f1: 13 x 13 x 1024
        f1 = darknet.output
        # f2: 26 x 26 x 512
        f2 = darknet.layers[204].output
        # f3: 52 x 52 x 256
        f3 = darknet.layers[131].output

        f1_channel_num = 1024
        f2_channel_num = 512
        f3_channel_num = 256

        y1, y2, y3 = self._yolo4_predictions_fn((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num),
                                       self._params['num_feature_layers'], self._params['num_classes'])

        model_outputs = {
            'y1': y1,
            'y2': y2,
            'y3': y3
        }

        return model_outputs


    def build_model(self, weights=None, is_training=None):
        outputs = self.model_outputs(self._input_layer, is_training)
        keras_model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name='yolo_v4')

        if weights:
            logger.info('Loaded pretrained weights from {}'.format(weights))
            keras_model.load_weights(weights, by_name=True)

        return keras_model