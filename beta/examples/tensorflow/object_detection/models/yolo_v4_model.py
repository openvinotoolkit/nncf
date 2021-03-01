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
from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.object_detection import losses
from beta.examples.tensorflow.common.object_detection.evaluation import coco_evaluator


class YOLOv4Model(base_model.Model):
    """YOLOv4 model function."""
    def __init__(self, params):
        super().__init__(params)

        self._params = params
        self._input_layer = tf.keras.layers.Input(shape=(None, None, 3), name='image_input')
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

    def build_model(self, weights=None, is_training=None):
        outputs = self.model_outputs(self._input_layer, is_training)
        keras_model = tf.keras.models.Model(inputs=self._input_layer, outputs=outputs, name='yolo_v4')

        if weights:
            logger.info('Loaded pretrained weights from {}'.format(weights))
            keras_model.load_weights(weights, by_name=True)

        return keras_model

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



    def post_processing(self, labels, outputs):

        import numpy as np
        import copy
        from scipy.special import expit, softmax

        def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
            '''Decode final layer features to bounding box parameters.'''
            batch_size = np.shape(prediction)[0]
            num_anchors = len(anchors)

            grid_size = np.shape(prediction)[1:3]
            # check if stride on height & width are same
            assert input_dims[0] // grid_size[0] == input_dims[1] // grid_size[1], 'model stride mismatch.'
            stride = input_dims[0] // grid_size[0]

            prediction = np.reshape(prediction,
                                    (batch_size, grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

            ################################
            # generate x_y_offset grid map
            grid_y = np.arange(grid_size[0])
            grid_x = np.arange(grid_size[1])
            x_offset, y_offset = np.meshgrid(grid_x, grid_y)

            x_offset = np.reshape(x_offset, (-1, 1))
            y_offset = np.reshape(y_offset, (-1, 1))

            x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
            x_y_offset = np.tile(x_y_offset, (1, num_anchors))
            x_y_offset = np.reshape(x_y_offset, (-1, 2))
            x_y_offset = np.expand_dims(x_y_offset, 0)

            ################################

            # Log space transform of the height and width
            anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
            anchors = np.expand_dims(anchors, 0)

            if scale_x_y:
                # Eliminate grid sensitivity trick involved in YOLOv4
                #
                # Reference Paper & code:
                #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
                #     https://arxiv.org/abs/2004.10934
                #     https://github.com/opencv/opencv/issues/17148
                #
                box_xy_tmp = expit(prediction[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
                box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_size)[::-1]
            else:
                box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
            box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

            # Sigmoid objectness scores
            objectness = expit(prediction[..., 4])  # p_o (objectness score)
            objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

            if use_softmax:
                # Softmax class scores
                class_scores = softmax(prediction[..., 5:], axis=-1)
            else:
                # Sigmoid class scores
                class_scores = expit(prediction[..., 5:])

            return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)
        def yolo3_decode(predictions, anchors, num_classes, input_dims, elim_grid_sense=False):
            """
            YOLOv3 Head to process predictions from YOLOv3 models

            :param num_classes: Total number of classes
            :param anchors: YOLO style anchor list for bounding box assignment
            :param input_dims: Input dimensions of the image
            :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
            :return: A tensor with the shape (N, num_boxes, 85)
            """
            assert len(predictions) == len(anchors) // 3, 'anchor numbers does not match prediction.'

            if len(predictions) == 3:  # assume 3 set of predictions is YOLOv3
                anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
            elif len(predictions) == 2:  # 2 set of predictions is YOLOv3-tiny
                anchor_mask = [[3, 4, 5], [0, 1, 2]]
                scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]
            else:
                raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

            results = []
            for i, prediction in enumerate(predictions):
                results.append(
                    yolo_decode(prediction, anchors[anchor_mask[i]], num_classes, input_dims, scale_x_y=scale_x_y[i],
                                use_softmax=False))

            return np.concatenate(results, axis=1)
        def yolo_correct_boxes(predictions, img_shape, model_image_size):
            '''rescale predicition boxes back to original image shape'''
            box_xy = predictions[..., :2]
            box_wh = predictions[..., 2:4]
            objectness = np.expand_dims(predictions[..., 4], -1)
            class_scores = predictions[..., 5:]

            # model_image_size & image_shape should be (height, width) format
            model_image_size = np.array(model_image_size, dtype='float32')
            image_shape = np.array(img_shape, dtype='float32')
            height, width = image_shape

            new_shape = np.round(image_shape * np.min(model_image_size / image_shape))
            offset = (model_image_size - new_shape) / 2. / model_image_size
            scale = model_image_size / new_shape
            # reverse offset/scale to match (w,h) order
            offset = offset[..., ::-1]
            scale = scale[..., ::-1]

            box_xy = (box_xy - offset) * scale
            box_wh *= scale

            # Convert centoids to top left coordinates
            box_xy -= box_wh / 2

            # Scale boxes back to original image shape.
            image_wh = image_shape[..., ::-1]
            box_xy *= image_wh
            box_wh *= image_wh

            return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)
        def box_diou(boxes):
            """
            Calculate DIoU value of 1st box with other boxes of a box array
            Reference Paper:
                "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
                https://arxiv.org/abs/1911.08287

            Parameters
            ----------
            boxes: bbox numpy array, shape=(N, 4), xywh
                   x,y are top left coordinates

            Returns
            -------
            diou: numpy array, shape=(N-1,)
                 IoU value of boxes[1:] with boxes[0]
            """
            # get box coordinate and area
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            areas = w * h

            # check IoU
            inter_xmin = np.maximum(x[1:], x[0])
            inter_ymin = np.maximum(y[1:], y[0])
            inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
            inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

            inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
            inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

            inter = inter_w * inter_h
            iou = inter / (areas[1:] + areas[0] - inter)

            # box center distance
            x_center = x + w / 2
            y_center = y + h / 2
            center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(y_center[1:] - y_center[0], 2)

            # get enclosed area
            enclose_xmin = np.minimum(x[1:], x[0])
            enclose_ymin = np.minimum(y[1:], y[0])
            enclose_xmax = np.maximum(x[1:] + w[1:], x[0] + w[0])
            enclose_ymax = np.maximum(x[1:] + w[1:], x[0] + w[0])
            enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
            enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
            # get enclosed diagonal distance
            enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
            # calculate DIoU, add epsilon in denominator to avoid dividing by 0
            diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

            return diou
        def nms_boxes(boxes, classes, scores, iou_threshold, confidence=0.1, use_diou=True, is_soft=False,
                      use_exp=False, sigma=0.5):
            nboxes, nclasses, nscores = [], [], []
            for c in set(classes):
                # handle data for one class
                inds = np.where(classes == c)
                b = boxes[inds]
                c = classes[inds]
                s = scores[inds]

                # make a data copy to avoid breaking
                # during nms operation
                b_nms = copy.deepcopy(b)
                c_nms = copy.deepcopy(c)
                s_nms = copy.deepcopy(s)

                while len(s_nms) > 0:
                    # pick the max box and store, here
                    # we also use copy to persist result
                    i = np.argmax(s_nms, axis=-1)
                    nboxes.append(copy.deepcopy(b_nms[i]))
                    nclasses.append(copy.deepcopy(c_nms[i]))
                    nscores.append(copy.deepcopy(s_nms[i]))

                    # swap the max line and first line
                    b_nms[[i, 0], :] = b_nms[[0, i], :]
                    c_nms[[i, 0]] = c_nms[[0, i]]
                    s_nms[[i, 0]] = s_nms[[0, i]]

                    if use_diou:
                        iou = box_diou(b_nms)
                        # iou = box_diou_matrix(b_nms, b_nms)[0][1:]
                    else:
                        iou = box_iou(b_nms)
                        # iou = box_iou_matrix(b_nms, b_nms)[0][1:]

                    # drop the last line since it has been record
                    b_nms = b_nms[1:]
                    c_nms = c_nms[1:]
                    s_nms = s_nms[1:]

                    if is_soft:
                        # Soft-NMS
                        if use_exp:
                            # score refresh formula:
                            # score = score * exp(-(iou^2)/sigma)
                            s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                        else:
                            # score refresh formula:
                            # score = score * (1 - iou) if iou > threshold
                            depress_mask = np.where(iou > iou_threshold)[0]
                            s_nms[depress_mask] = s_nms[depress_mask] * (1 - iou[depress_mask])
                        keep_mask = np.where(s_nms >= confidence)[0]
                    else:
                        # normal Hard-NMS
                        keep_mask = np.where(iou <= iou_threshold)[0]

                    # keep needed box for next loop
                    b_nms = b_nms[keep_mask]
                    c_nms = c_nms[keep_mask]
                    s_nms = s_nms[keep_mask]

            # reformat result for output
            nboxes = [np.array(nboxes)]
            nclasses = [np.array(nclasses)]
            nscores = [np.array(nscores)]
            return nboxes, nclasses, nscores
        def filter_boxes(boxes, classes, scores, max_boxes):
            '''
            Sort the prediction boxes according to score
            and only pick top "max_boxes" ones
            '''
            # sort result according to scores
            sorted_indices = np.argsort(scores)
            sorted_indices = sorted_indices[::-1]
            nboxes = boxes[sorted_indices]
            nclasses = classes[sorted_indices]
            nscores = scores[sorted_indices]

            # only pick max_boxes
            nboxes = nboxes[:max_boxes]
            nclasses = nclasses[:max_boxes]
            nscores = nscores[:max_boxes]

            return nboxes, nclasses, nscores
        def yolo_handle_predictions(predictions, image_shape, max_boxes=100, confidence=0.1, iou_threshold=0.4,
                                    use_cluster_nms=False, use_wbf=False):
            boxes = predictions[:, :, :4]
            box_confidences = np.expand_dims(predictions[:, :, 4], -1)
            box_class_probs = predictions[:, :, 5:]

            # filter boxes with confidence threshold
            box_scores = box_confidences * box_class_probs
            box_classes = np.argmax(box_scores, axis=-1)
            box_class_scores = np.max(box_scores, axis=-1)
            pos = np.where(box_class_scores >= confidence)

            boxes = boxes[pos]
            classes = box_classes[pos]
            scores = box_class_scores[pos]

            if use_cluster_nms:
                # use Fast/Cluster NMS for boxes postprocess
                n_boxes, n_classes, n_scores = fast_cluster_nms_boxes(boxes, classes, scores, iou_threshold,
                                                                      confidence=confidence)
            elif use_wbf:
                # use Weighted-Boxes-Fusion for boxes postprocess
                n_boxes, n_classes, n_scores = weighted_boxes_fusion([boxes], [classes], [scores], image_shape,
                                                                     weights=None, iou_thr=iou_threshold)
            else:
                # Boxes, Classes and Scores returned from NMS
                n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence)

            if n_boxes:
                boxes = np.concatenate(n_boxes)
                classes = np.concatenate(n_classes).astype('int32')
                scores = np.concatenate(n_scores)
                boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

                return boxes, classes, scores

            else:
                return [], [], []
        def yolo_adjust_boxes(boxes, img_shape):
            '''
            change box format from (x,y,w,h) top left coordinate to
            (xmin,ymin,xmax,ymax) format
            '''
            if boxes is None or len(boxes) == 0:
                return []

            image_shape = np.array(img_shape, dtype='float32')
            height, width = image_shape

            adjusted_boxes = []
            for box in boxes:
                x, y, w, h = box

                xmin = x
                ymin = y
                xmax = x + w
                ymax = y + h

                ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
                xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
                ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
                xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
                adjusted_boxes.append([xmin, ymin, xmax, ymax])

            return np.array(adjusted_boxes, dtype=np.int32)
        def get_anchors(anchors_path):
            '''loads the anchors from a file'''
            with open(anchors_path) as f:
                anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
        def yolo3_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100,
                                 confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
            predictions = yolo3_decode(yolo_outputs, anchors, num_classes, input_dims=model_image_size,
                                       elim_grid_sense=elim_grid_sense)
            predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

            boxes, classes, scores = yolo_handle_predictions(predictions,
                                                             image_shape,
                                                             max_boxes=max_boxes,
                                                             confidence=confidence,
                                                             iou_threshold=iou_threshold)

            boxes = yolo_adjust_boxes(boxes, image_shape)

            return boxes, classes, scores

        def postprocess_main(image_info, out1, out2, out3):
            image_info = image_info.numpy()
            out1 = out1.numpy()
            out2 = out2.numpy()
            out3 = out3.numpy()

            batch_size = image_info.shape[0]

            boxes = []
            classes = []
            scores = []
            for i in range(batch_size):
                prediction = [out1[i][None, ...], out2[i][None, ...], out3[i][None, ...]]
                image_shape = image_info[i][0]
                anchors = get_anchors(self._params['anchors_path'])
                pred_boxes, pred_classes, pred_scores = yolo3_postprocess_np(prediction, image_shape, anchors, self._params['num_classes'],
                                                                             self._params['input_shape'], max_boxes=100,
                                                                             confidence=self._params['conf_threshold'],
                                                                             elim_grid_sense=self._params['elim_grid_sense'])
                print('pred_boxes', pred_boxes.shape, type(pred_boxes))
                boxes.append(pred_boxes)
                classes.append(pred_classes)
                scores.append(pred_scores)

            boxes = np.concatenate(boxes, axis=0)
            classes = np.concatenate(classes, axis=0)
            scores = np.concatenate(scores, axis=0)

            print('boxes', boxes.shape)
            print('classes', classes.shape)
            print('scores', scores.shape)

            boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            classes = tf.convert_to_tensor(classes, dtype=tf.float32)
            scores = tf.convert_to_tensor(scores, dtype=tf.float32)

            return boxes, classes, scores


        # print('labels', type(labels))
        # for key, val in labels.items():
        #     print(key, val)
        # print([item.shape for item in list(outputs.values())])

        image_info = labels['image_info']
        out1 = outputs['y1']
        out2 = outputs['y2']
        out3 = outputs['y3']

        boxes, classes, scores = tf.py_function(postprocess_main, [image_info, out1, out2, out3], [tf.float32, tf.float32, tf.float32]) # , tf.float32

        # batch_size = labels.shape[0]
        # print('batch_size', batch_size)
        # for i in batch_size:






        # required_output_fields = ['cls_outputs', 'box_outputs']
        #
        # for field in required_output_fields:
        #     if field not in outputs:
        #         raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
        #                          field, required_output_fields, outputs.keys()))
        #
        # boxes, scores, classes, valid_detections = self._generate_detections_fn(
        #     outputs['box_outputs'], outputs['cls_outputs'], labels['anchor_boxes'],
        #     labels['image_info'][:, 1:2, :])
        # # Discards the old output tensors to save memory. The `cls_outputs` and
        # # `box_outputs` are pretty big and could potentially lead to memory issue.
        # outputs = {
        #     'source_id': labels['source_id'],
        #     'image_info': labels['image_info'],
        #     'num_detections': valid_detections, # a numpy array of int of shape [batch_size].
        #     'detection_boxes': boxes,
        #     'detection_classes': classes,
        #     'detection_scores': scores,
        # }




        return labels, outputs







    def eval_metrics(self):
        annotation_file = self._params.get('val_json_file', None)
        evaluator = coco_evaluator.COCOEvaluator(annotation_file=annotation_file,
                                                 include_mask=False)
        return coco_evaluator.MetricWrapper(evaluator)