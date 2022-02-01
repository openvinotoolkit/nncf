from nncf.common.utils.ordered_enum import OrderedEnum


class Backend(OrderedEnum):
    ONNX = 1
    PYTORCH = 2
    TENSORFLOW = 3
    OPENVINO = 4


def define_the_backend(model):
    from torch.nn import Module
    from tensorflow.keras.models import Model
    from onnx import ModelProto
    if isinstance(model, ModelProto):
        return Backend.ONNX
    if isinstance(model, str):
        return Backend.ONNX
    elif isinstance(model, Module):
        return Backend.PYTORCH
    elif isinstance(model, Model):
        return Backend.TENSORFLOW
    elif isinstance(model, ):  # TODO: add OpenVINO
        return Backend.OPENVINO
    raise RuntimeError('This backend is not supported')
