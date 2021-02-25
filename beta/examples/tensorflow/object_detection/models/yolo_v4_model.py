"""
 Copyright (c) 2021 Intel Corporation
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
from beta.examples.tensorflow.common.object_detection.architecture import keras_utils
from beta.examples.tensorflow.common.object_detection.ops import postprocess_ops
from beta.examples.tensorflow.common.object_detection.evaluation import coco_evaluator
from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.object_detection import losses


class YOLOv4Model(base_model.Model):
    """YOLOv4 model function."""
    def __init__(self, params):
        super().__init__(params)

        self._params = params

        # Architecture generators.
        self._backbone_fn = factory.backbone_generator(params)
        self._yolo4_predictions_fn = factory.yolo_v4_head_generator(params)

        self._input_layer = tf.keras.layers.Input(shape=(None, None, 3), name='image_input')

        self._loss_fn = losses.YOLOv4Loss()


    def build_outputs(self, inputs, is_training):
        """Create YOLO_V4 model CNN body in Keras."""
        # darknet = tf.keras.models.Model(inputs, csp_darknet53_body(inputs))
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

        # y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num),
        #                                self._params['num_feature_layers'], self._params['num_classes'])
        y1, y2, y3 = self._yolo4_predictions_fn((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num),
                                       self._params['num_feature_layers'], self._params['num_classes'])

        model_outputs = {
            'y1': y1,
            'y2': y2,
            'y3': y3
        }

        return model_outputs

        # backbone_features = self._backbone_fn(inputs)
        # fpn_features = self._fpn_fn(backbone_features)
        # cls_outputs, box_outputs = self._head_fn(fpn_features)
        #
        # model_outputs = {
        #     'cls_outputs': cls_outputs,
        #     'box_outputs': box_outputs,
        # }
        #
        # return model_outputs

    def build_loss_fn(self, compress_model):

        def _total_loss_fn(labels, outputs):
            anchors_path = self._params['anchors_path']
            num_classes = self._params['num_classes']

            loss, total_location_loss, total_confidence_loss, total_class_loss = self._loss_fn(labels, outputs,
                                                                                                   anchors_path, num_classes,
                                                                                                   ignore_thresh=.5,
                                                                                                   label_smoothing=0,
                                                                                                   elim_grid_sense=True,
                                                                                                   use_focal_loss=False,
                                                                                                   use_focal_obj_loss=False,
                                                                                                   use_softmax_loss=False,
                                                                                                   use_giou_loss=False,
                                                                                                   use_diou_loss=True)
            return {
                'total_loss': loss,
                'total_location_loss': total_location_loss,
                'total_confidence_loss': total_confidence_loss,
                'total_class_loss': total_class_loss
            }

        return _total_loss_fn


    def build_model(self, weights=None, is_training=None):
        # with keras_utils.maybe_enter_backend_graph():
        outputs = self.model_outputs(self._input_layer, is_training)
        keras_model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name='yolo_v4')

        if self._checkpoint_path:
            logger.info('Init backbone')
            init_checkpoint_fn = self.make_restore_checkpoint_fn()
            init_checkpoint_fn(keras_model)

        if weights:
            logger.info('Loaded pretrained weights from {}'.format(weights))
            keras_model.load_weights(weights, by_name=True)

        return keras_model