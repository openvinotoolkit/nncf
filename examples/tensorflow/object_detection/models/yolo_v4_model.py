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
from examples.tensorflow.object_detection.postprocessing.yolo_v4_postprocessing import postprocess_yolo_v4_np


class YOLOv4Model(base_model.Model):
    """YOLOv4 model function."""

    def __init__(self, params):
        super().__init__(params)

        self._params = params
        self._input_layer = tf.keras.layers.Input(shape=params.input_info.sample_size[1:], name="image_input")
        self._loss_fn = losses.YOLOv4Loss()

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

        y1, y2, y3 = self._yolo4_predictions_fn(
            (f1, f2, f3),
            (f1_channel_num, f2_channel_num, f3_channel_num),
            self._params.model_params.num_feature_layers,
            self._params.model_params.num_classes,
        )

        model_outputs = {"y1": y1, "y2": y2, "y3": y3}

        return model_outputs

    def build_model(self, weights=None, is_training=None):
        outputs = self.model_outputs(self._input_layer, is_training)
        keras_model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name="yolo_v4")
        if weights:
            logger.info("Loaded pretrained weights from {}".format(weights))
            keras_model.load_weights(weights, by_name=True)
        return keras_model

    def build_loss_fn(self, keras_model, compression_loss_fn):
        def _total_loss_fn(labels, outputs):
            loss_fn_out = self._loss_fn(
                labels,
                outputs,
                self._params.anchors,
                self._params.model_params.num_classes,
                ignore_thresh=self._params.model_params.loss_params.ignore_thresh,
                label_smoothing=self._params.model_params.loss_params.label_smoothing,
                elim_grid_sense=self._params.elim_grid_sense,
                use_focal_loss=self._params.model_params.loss_params.use_focal_loss,
                use_focal_obj_loss=self._params.model_params.loss_params.use_focal_obj_loss,
                use_softmax_loss=self._params.model_params.loss_params.use_softmax_loss,
                use_giou_loss=self._params.model_params.loss_params.use_giou_loss,
                use_diou_loss=self._params.model_params.loss_params.use_diou_loss,
            )
            loss, total_location_loss, total_confidence_loss, total_class_loss = loss_fn_out
            compression_loss = compression_loss_fn()
            loss += compression_loss
            return {
                "total_loss": loss,
                "total_location_loss": total_location_loss,
                "total_confidence_loss": total_confidence_loss,
                "total_class_loss": total_class_loss,
                "compression_loss": compression_loss,
            }

        return _total_loss_fn

    def post_processing(self, labels, outputs):
        out1 = outputs["y1"]
        out2 = outputs["y2"]
        out3 = outputs["y3"]
        boxes, classes, scores, valid_detections = tf.py_function(
            postprocess_yolo_v4_np,
            [
                labels["image_info"],
                out1,
                out2,
                out3,
                self._params.anchors,
                self._params.model_params.num_classes,
                self._params.input_shape,
                self._params.postprocessing.conf_threshold,
                self._params.elim_grid_sense,
            ],
            [tf.float32, tf.int32, tf.float32, tf.int32],
        )
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
        evaluator = coco_evaluator.COCOEvaluator(
            annotation_file=annotation_file, include_mask=False, need_rescale_bboxes=False
        )
        return coco_evaluator.MetricWrapper(evaluator)
