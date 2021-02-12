#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from __future__ import division
from functools import wraps

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow as tf

L2_FACTOR = 1e-5

@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return Conv2D(*args, **yolo_conv_kwargs)


@wraps(DepthwiseConv2D)
def YoloDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for DepthwiseConv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return DepthwiseConv2D(*args, **yolo_conv_kwargs)


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)

