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


from beta.examples.tensorflow.object_detection.yolo_v4_architecture.darknet_backbone import csp_darknet53_body
from beta.examples.tensorflow.object_detection.yolo_v4_architecture.yolo_v4_head import yolo4_predictions


class YOLOv4Model(base_model.Model):
    """YOLOv4 model function."""

    def __init__(self, params):
        super().__init__(params)

        self._params = params

        # self._checkpoint_prefix = 'resnet50/'

        # # Architecture generators.
        # self._backbone_fn = factory.backbone_generator(params)
        # self._fpn_fn = factory.multilevel_features_generator(params)
        # self._head_fn = factory.retinanet_head_generator(params)

        # # Loss function.
        # self._cls_loss_fn = losses.RetinanetClassLoss(params.model_params.loss_params,
        #                                               params.model_params.architecture.num_classes)
        # self._box_loss_fn = losses.RetinanetBoxLoss(params.model_params.loss_params)
        # self._box_loss_weight = params.model_params.loss_params.box_loss_weight

        # # Predict function.
        # self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        #     params.model_params.architecture.min_level,
        #     params.model_params.architecture.max_level,
        #     params.model_params.postprocessing)

        # Input layer.
        # self._input_layer = tf.keras.layers.Input(
        #     shape=(None, None, params.input_info.sample_size[-1]),
        #     name='',
        #     dtype=tf.float32)
        self._input_layer = tf.keras.layers.Input(shape=(None, None, 3), name='image_input')


    def build_outputs(self, inputs, is_training):
        """Create YOLO_V4 model CNN body in Keras."""
        darknet = tf.keras.models.Model(inputs, csp_darknet53_body(inputs))
        print('Backbone layers number: {}'.format(len(darknet.layers)))
        # weights_path = 'weights/cspdarknet53.h5'
        # if weights_path is not None:
        #     darknet.load_weights(weights_path, by_name=True)
        #     print('Load backbone weights {}.'.format(weights_path))

        # f1: 13 x 13 x 1024
        f1 = darknet.output
        # f2: 26 x 26 x 512
        f2 = darknet.layers[204].output
        # f3: 52 x 52 x 256
        f3 = darknet.layers[131].output

        f1_channel_num = 1024
        f2_channel_num = 512
        f3_channel_num = 256

        y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num),
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


    def build_loss_fn(self, keras_model):
        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                         labels['cls_targets'],
                                         labels['num_positives'])
            box_loss = self._box_loss_fn(outputs['box_outputs'],
                                         labels['box_targets'],
                                         labels['num_positives'])

            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss + l2_regularization_loss

            return {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'box_loss': box_loss,
                'model_loss': model_loss,
                'l2_regularization_loss': l2_regularization_loss,
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

    def post_processing(self, labels, outputs):
        required_output_fields = ['cls_outputs', 'box_outputs']

        for field in required_output_fields:
            if field not in outputs:
                raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
                                 field, required_output_fields, outputs.keys()))

        boxes, scores, classes, valid_detections = self._generate_detections_fn(
            outputs['box_outputs'], outputs['cls_outputs'], labels['anchor_boxes'],
            labels['image_info'][:, 1:2, :])
        # Discards the old output tensors to save memory. The `cls_outputs` and
        # `box_outputs` are pretty big and could potentiall lead to memory issue.
        outputs = {
            'source_id': labels['source_id'],
            'image_info': labels['image_info'],
            'num_detections': valid_detections,
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
        }

        return labels, outputs

    def eval_metrics(self):
        annotation_file = self._params.get('val_json_file', None)
        evaluator = coco_evaluator.COCOEvaluator(annotation_file=annotation_file,
                                                 include_mask=False)
        return coco_evaluator.MetricWrapper(evaluator)
