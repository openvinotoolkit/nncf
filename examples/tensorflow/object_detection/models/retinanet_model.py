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

import tensorflow as tf

from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.object_detection import base_model
from examples.tensorflow.common.object_detection import losses
from examples.tensorflow.common.object_detection.architecture import factory
from examples.tensorflow.common.object_detection.evaluation import coco_evaluator
from examples.tensorflow.common.object_detection.ops import postprocess_ops


class RetinanetModel(base_model.Model):
    """RetinaNet model function."""

    def __init__(self, params):
        super().__init__(params)

        # For eval metrics.
        self._params = params

        self._checkpoint_prefix = "resnet50/"

        # Architecture generators.
        self._backbone_fn = factory.backbone_generator(params)
        self._fpn_fn = factory.multilevel_features_generator(params)
        self._head_fn = factory.retinanet_head_generator(params)

        # Loss function.
        self._cls_loss_fn = losses.RetinanetClassLoss(
            params.model_params.loss_params, params.model_params.architecture.num_classes
        )
        self._box_loss_fn = losses.RetinanetBoxLoss(params.model_params.loss_params)
        self._box_loss_weight = params.model_params.loss_params.box_loss_weight

        # Predict function.
        self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
            params.model_params.architecture.min_level,
            params.model_params.architecture.max_level,
            params.model_params.postprocessing,
        )

        # Input layer.
        self._input_layer = tf.keras.layers.Input(shape=(params.input_info.sample_size[1:]), name="", dtype=tf.float32)

    def build_outputs(self, inputs, is_training):
        backbone_features = self._backbone_fn(inputs)
        fpn_features = self._fpn_fn(backbone_features)
        cls_outputs, box_outputs = self._head_fn(fpn_features)

        model_outputs = {
            "cls_outputs": cls_outputs,
            "box_outputs": box_outputs,
        }

        return model_outputs

    def build_loss_fn(self, keras_model, compression_loss_fn):
        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            cls_loss = self._cls_loss_fn(outputs["cls_outputs"], labels["cls_targets"], labels["num_positives"])
            box_loss = self._box_loss_fn(outputs["box_outputs"], labels["box_targets"], labels["num_positives"])

            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            compression_loss = compression_loss_fn()
            total_loss = model_loss + l2_regularization_loss + compression_loss

            return {
                "total_loss": total_loss,
                "cls_loss": cls_loss,
                "box_loss": box_loss,
                "model_loss": model_loss,
                "l2_regularization_loss": l2_regularization_loss,
                "compression_loss": compression_loss,
            }

        return _total_loss_fn

    def build_model(self, weights=None, is_training=None):
        outputs = self.model_outputs(self._input_layer, is_training)
        keras_model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name="retinanet")

        if self._checkpoint_path:
            logger.info("Init backbone")
            init_checkpoint_fn = self.make_restore_checkpoint_fn()
            init_checkpoint_fn(keras_model)

        if weights:
            logger.info("Loaded pretrained weights from {}".format(weights))
            keras_model.load_weights(weights)

        return keras_model

    def post_processing(self, labels, outputs):
        required_output_fields = ["cls_outputs", "box_outputs"]

        for field in required_output_fields:
            if field not in outputs:
                raise ValueError(
                    '"{}" is missing in outputs, requried {} found {}'.format(
                        field, required_output_fields, outputs.keys()
                    )
                )

        boxes, scores, classes, valid_detections = self._generate_detections_fn(
            outputs["box_outputs"], outputs["cls_outputs"], labels["anchor_boxes"], labels["image_info"][:, 1:2, :]
        )
        # Discards the old output tensors to save memory. The `cls_outputs` and
        # `box_outputs` are pretty big and could potentially lead to memory issue.
        outputs = {
            "source_id": labels["source_id"],
            "image_info": labels["image_info"],
            "num_detections": valid_detections,
            "detection_boxes": boxes,
            "detection_classes": classes,
            "detection_scores": scores,
        }

        return labels, outputs

    def eval_metrics(self):
        annotation_file = self._params.get("val_json_file", None)
        evaluator = coco_evaluator.COCOEvaluator(annotation_file=annotation_file, include_mask=False)
        return coco_evaluator.MetricWrapper(evaluator)
