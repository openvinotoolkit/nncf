#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv3 models with different backbone & head
"""
import warnings
from functools import partial

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# from yolo3.models.yolo3_darknet import yolo3_body, custom_tiny_yolo3_body, yolo3lite_body, tiny_yolo3lite_body, custom_yolo3_spp_body
# from yolo3.models.yolo3_mobilenet import yolo3_mobilenet_body, tiny_yolo3_mobilenet_body, yolo3lite_mobilenet_body, yolo3lite_spp_mobilenet_body, tiny_yolo3lite_mobilenet_body
# from yolo3.models.yolo3_mobilenetv2 import yolo3_mobilenetv2_body, tiny_yolo3_mobilenetv2_body, yolo3lite_mobilenetv2_body, yolo3lite_spp_mobilenetv2_body, tiny_yolo3lite_mobilenetv2_body, yolo3_ultralite_mobilenetv2_body, tiny_yolo3_ultralite_mobilenetv2_body
# from yolo3.models.yolo3_shufflenetv2 import yolo3_shufflenetv2_body, tiny_yolo3_shufflenetv2_body, yolo3lite_shufflenetv2_body, yolo3lite_spp_shufflenetv2_body, tiny_yolo3lite_shufflenetv2_body
# from yolo3.models.yolo3_vgg16 import yolo3_vgg16_body, tiny_yolo3_vgg16_body
# from yolo3.models.yolo3_xception import yolo3_xception_body, yolo3lite_xception_body, tiny_yolo3_xception_body, tiny_yolo3lite_xception_body, yolo3_spp_xception_body
# from yolo3.models.yolo3_nano import yolo3_nano_body
# from yolo3.models.yolo3_efficientnet import yolo3_efficientnet_body, tiny_yolo3_efficientnet_body, yolo3lite_efficientnet_body, yolo3lite_spp_efficientnet_body, tiny_yolo3lite_efficientnet_body
# from yolo3.models.yolo3_mobilenetv3_large import yolo3_mobilenetv3large_body, yolo3lite_mobilenetv3large_body, tiny_yolo3_mobilenetv3large_body, tiny_yolo3lite_mobilenetv3large_body
# from yolo3.models.yolo3_mobilenetv3_small import yolo3_mobilenetv3small_body, yolo3lite_mobilenetv3small_body, tiny_yolo3_mobilenetv3small_body, tiny_yolo3lite_mobilenetv3small_body, yolo3_ultralite_mobilenetv3small_body, tiny_yolo3_ultralite_mobilenetv3small_body
# #from yolo3.models.yolo3_resnet50v2 import yolo3_resnet50v2_body, yolo3lite_resnet50v2_body, yolo3lite_spp_resnet50v2_body, tiny_yolo3_resnet50v2_body, tiny_yolo3lite_resnet50v2_body


from yolo4.models.yolo4_darknet import yolo4_body
# from yolo4.models.yolo4_mobilenet import yolo4_mobilenet_body, yolo4lite_mobilenet_body, tiny_yolo4_mobilenet_body, tiny_yolo4lite_mobilenet_body
# from yolo4.models.yolo4_mobilenetv2 import yolo4_mobilenetv2_body, yolo4lite_mobilenetv2_body, tiny_yolo4_mobilenetv2_body, tiny_yolo4lite_mobilenetv2_body

# from yolo4.models.yolo4_mobilenetv3_large import yolo4_mobilenetv3large_body, yolo4lite_mobilenetv3large_body, tiny_yolo4_mobilenetv3large_body, tiny_yolo4lite_mobilenetv3large_body
# from yolo4.models.yolo4_mobilenetv3_small import yolo4_mobilenetv3small_body, yolo4lite_mobilenetv3small_body, tiny_yolo4_mobilenetv3small_body, tiny_yolo4lite_mobilenetv3small_body
# from yolo4.models.yolo4_efficientnet import yolo4_efficientnet_body, yolo4lite_efficientnet_body, tiny_yolo4_efficientnet_body, tiny_yolo4lite_efficientnet_body
#from yolo4.models.yolo4_resnet50v2 import yolo4_resnet50v2_body, yolo4lite_resnet50v2_body, tiny_yolo4_resnet50v2_body, tiny_yolo4lite_resnet50v2_body

from yolo3.loss import yolo3_loss
from yolo3.postprocess import batched_yolo3_postprocess, batched_yolo3_prenms, Yolo3PostProcessLayer

from common.model_utils import add_metrics#, get_pruning_model


# A map of model type to construction info list for YOLOv3
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_path]
#
yolo3_model_map = {
    # 'yolo3_mobilenet': [yolo3_mobilenet_body, 87, None],
    # 'yolo3_mobilenet_lite': [yolo3lite_mobilenet_body, 87, None],
    # 'yolo3_mobilenet_lite_spp': [yolo3lite_spp_mobilenet_body, 87, None],
    # 'yolo3_mobilenetv2': [yolo3_mobilenetv2_body, 155, None],
    # 'yolo3_mobilenetv2_lite': [yolo3lite_mobilenetv2_body, 155, None],
    # 'yolo3_mobilenetv2_lite_spp': [yolo3lite_spp_mobilenetv2_body, 155, None],
    # 'yolo3_mobilenetv2_ultralite': [yolo3_ultralite_mobilenetv2_body, 155, None],
    #
    # 'yolo3_mobilenetv3large': [yolo3_mobilenetv3large_body, 195, None],
    # 'yolo3_mobilenetv3large_lite': [yolo3lite_mobilenetv3large_body, 195, None],
    # 'yolo3_mobilenetv3small': [yolo3_mobilenetv3small_body, 166, None],
    # 'yolo3_mobilenetv3small_lite': [yolo3lite_mobilenetv3small_body, 166, None],
    # 'yolo3_mobilenetv3small_ultralite': [yolo3_ultralite_mobilenetv3small_body, 166, None],

    #'yolo3_resnet50v2': [yolo3_resnet50v2_body, 190, None],
    #'yolo3_resnet50v2_lite': [yolo3lite_resnet50v2_body, 190, None],
    #'yolo3_resnet50v2_lite_spp': [yolo3lite_spp_resnet50v2_body, 190, None],

    # 'yolo3_shufflenetv2': [yolo3_shufflenetv2_body, 205, None],
    # 'yolo3_shufflenetv2_lite': [yolo3lite_shufflenetv2_body, 205, None],
    # 'yolo3_shufflenetv2_lite_spp': [yolo3lite_spp_shufflenetv2_body, 205, None],
    #
    # # NOTE: backbone_length is for EfficientNetB3
    # # if change to other efficientnet level, you need to modify it
    # 'yolo3_efficientnet': [yolo3_efficientnet_body, 382, None],
    # 'yolo3_efficientnet_lite': [yolo3lite_efficientnet_body, 382, None],
    # 'yolo3_efficientnet_lite_spp': [yolo3lite_spp_efficientnet_body, 382, None],
    #
    # 'yolo3_darknet': [yolo3_body, 185, 'weights/darknet53.h5'],
    # 'yolo3_darknet_spp': [custom_yolo3_spp_body, 185, 'weights/yolov3-spp.h5'],
    # #Doesn't have pretrained weights, so no need to return backbone length
    # 'yolo3_darknet_lite': [yolo3lite_body, 0, None],
    # 'yolo3_vgg16': [yolo3_vgg16_body, 19, None],
    # 'yolo3_xception': [yolo3_xception_body, 132, None],
    # 'yolo3_xception_lite': [yolo3lite_xception_body, 132, None],
    # 'yolo3_xception_spp': [yolo3_spp_xception_body, 132, None],
    #
    # 'yolo3_nano': [yolo3_nano_body, 268, None],

    'yolo4_darknet': [yolo4_body, 250, 'weights/cspdarknet53.h5'],
    # 'yolo4_mobilenet': [yolo4_mobilenet_body, 87, None],
    # 'yolo4_mobilenet_lite': [yolo4lite_mobilenet_body, 87, None],
    #
    # 'yolo4_mobilenetv2': [yolo4_mobilenetv2_body, 155, None],
    # 'yolo4_mobilenetv2_lite': [yolo4lite_mobilenetv2_body, 155, None],
    #
    # 'yolo4_mobilenetv3large': [yolo4_mobilenetv3large_body, 195, None],
    # 'yolo4_mobilenetv3large_lite': [yolo4lite_mobilenetv3large_body, 195, None],
    # 'yolo4_mobilenetv3small': [yolo4_mobilenetv3small_body, 166, None],
    # 'yolo4_mobilenetv3small_lite': [yolo4lite_mobilenetv3small_body, 166, None],

    #'yolo4_resnet50v2': [yolo4_resnet50v2_body, 190, None],
    #'yolo4_resnet50v2_lite': [yolo4lite_resnet50v2_body, 190, None],

    # NOTE: backbone_length is for EfficientNetB1
    # if change to other efficientnet level, you need to modify it
    # 'yolo4_efficientnet': [yolo4_efficientnet_body, 337, None],
    # 'yolo4_efficientnet_lite': [yolo4lite_efficientnet_body, 337, None],

}


# A map of model type to construction info list for Tiny YOLOv3
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_file]
#
# yolo3_tiny_model_map = {
#     'tiny_yolo3_mobilenet': [tiny_yolo3_mobilenet_body, 87, None],
#     'tiny_yolo3_mobilenet_lite': [tiny_yolo3lite_mobilenet_body, 87, None],
#     'tiny_yolo3_mobilenetv2': [tiny_yolo3_mobilenetv2_body, 155, None],
#     'tiny_yolo3_mobilenetv2_lite': [tiny_yolo3lite_mobilenetv2_body, 155, None],
#     'tiny_yolo3_mobilenetv2_ultralite': [tiny_yolo3_ultralite_mobilenetv2_body, 155, None],
#
#     'tiny_yolo3_mobilenetv3large': [tiny_yolo3_mobilenetv3large_body, 195, None],
#     'tiny_yolo3_mobilenetv3large_lite': [tiny_yolo3lite_mobilenetv3large_body, 195, None],
#     'tiny_yolo3_mobilenetv3small': [tiny_yolo3_mobilenetv3small_body, 166, None],
#     'tiny_yolo3_mobilenetv3small_lite': [tiny_yolo3lite_mobilenetv3small_body, 166, None],
#     'tiny_yolo3_mobilenetv3small_ultralite': [tiny_yolo3_ultralite_mobilenetv3small_body, 166, None],
#
#     #'tiny_yolo3_resnet50v2': [tiny_yolo3_resnet50v2_body, 190, None],
#     #'tiny_yolo3_resnet50v2_lite': [tiny_yolo3lite_resnet50v2_body, 190, None],
#
#     'tiny_yolo3_shufflenetv2': [tiny_yolo3_shufflenetv2_body, 205, None],
#     'tiny_yolo3_shufflenetv2_lite': [tiny_yolo3lite_shufflenetv2_body, 205, None],
#
#     # NOTE: backbone_length is for EfficientNetB0
#     # if change to other efficientnet level, you need to modify it
#     'tiny_yolo3_efficientnet': [tiny_yolo3_efficientnet_body, 235, None],
#     'tiny_yolo3_efficientnet_lite': [tiny_yolo3lite_efficientnet_body, 235, None],
#
#     'tiny_yolo3_darknet': [custom_tiny_yolo3_body, 20, 'weights/yolov3-tiny.h5'],
#     #Doesn't have pretrained weights, so no need to return backbone length
#     'tiny_yolo3_darknet_lite': [tiny_yolo3lite_body, 0, None],
#     'tiny_yolo3_vgg16': [tiny_yolo3_vgg16_body, 19, None],
#     'tiny_yolo3_xception': [tiny_yolo3_xception_body, 132, None],
#     'tiny_yolo3_xception_lite': [tiny_yolo3lite_xception_body, 132, None],
#
#     'tiny_yolo4_mobilenet': [tiny_yolo4_mobilenet_body, 87, None],
#     'tiny_yolo4_mobilenet_lite': [tiny_yolo4lite_mobilenet_body, 87, None],
#     'tiny_yolo4_mobilenet_lite_nospp': [partial(tiny_yolo4lite_mobilenet_body, use_spp=False), 87, None],
#     'tiny_yolo4_mobilenetv2': [tiny_yolo4_mobilenetv2_body, 155, None],
#     'tiny_yolo4_mobilenetv2_lite': [tiny_yolo4lite_mobilenetv2_body, 155, None],
#     'tiny_yolo4_mobilenetv2_lite_nospp': [partial(tiny_yolo4lite_mobilenetv2_body, use_spp=False), 155, None],
#
#     'tiny_yolo4_mobilenetv3large': [tiny_yolo4_mobilenetv3large_body, 195, None],
#     'tiny_yolo4_mobilenetv3large_lite': [tiny_yolo4lite_mobilenetv3large_body, 195, None],
#     'tiny_yolo4_mobilenetv3large_lite_nospp': [partial(tiny_yolo4lite_mobilenetv3large_body, use_spp=False), 195, None],
#
#     'tiny_yolo4_mobilenetv3small': [tiny_yolo4_mobilenetv3small_body, 166, None],
#     'tiny_yolo4_mobilenetv3small_lite': [tiny_yolo4lite_mobilenetv3small_body, 166, None],
#     'tiny_yolo4_mobilenetv3small_lite_nospp': [partial(tiny_yolo4lite_mobilenetv3small_body, use_spp=False), 166, None],
#
#     #'tiny_yolo4_resnet50v2': [tiny_yolo4_resnet50v2_body, 190, None],
#     #'tiny_yolo4_resnet50v2_lite': [tiny_yolo4lite_resnet50v2_body, 190, None],
#
#     # NOTE: backbone_length is for EfficientNetB0
#     # if change to other efficientnet level, you need to modify it
#     'tiny_yolo4_efficientnet': [tiny_yolo4_efficientnet_body, 235, None],
#     'tiny_yolo4_efficientnet_lite': [tiny_yolo4lite_efficientnet_body, 235, None],
#     'tiny_yolo4_efficientnet_lite_nospp': [partial(tiny_yolo4lite_efficientnet_body, use_spp=False), 235, None],
#
# }


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None, model_pruning=False, pruning_end_step=10000):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    #Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            weights_path = yolo3_tiny_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//2, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    #YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//3, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    # if model_pruning:
    #     model_body = get_pruning_model(model_body, begin_step=0, end_step=pruning_end_step)

    return model_body, backbone_len



def get_yolo3_train_model(model_type, anchors, num_classes, weights_path=None, optimizer=Adam(lr=1e-3, decay=0), label_smoothing=0, elim_grid_sense=False, model_pruning=False, pruning_end_step=10000):
    '''create the training model, for YOLOv3'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    #feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [Input(shape=(None, None, 3, num_classes+5), name='y_true_{}'.format(l)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, model_pruning=model_pruning, pruning_end_step=pruning_end_step)
    print('Create {} {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))
    print('model layer number:', len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    freeze_level = 0
    # if freeze_level in [1, 2]:
    #     # Freeze the backbone part or freeze all but final feature map & input layers.
    #     num = (backbone_len, len(model_body.layers)-3)[freeze_level-1]
    #     for i in range(num): model_body.layers[i].trainable = False
    #     print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    if freeze_level == 0:
        # Unfreeze all layers.
        # TODO: check if it is really needed
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    return model


def get_yolo3_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    '''create the inference model, for YOLOv3'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=input_shape)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo3_postprocess, name='yolo3_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_threshold, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

