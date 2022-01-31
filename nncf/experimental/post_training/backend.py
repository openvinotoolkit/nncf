from nncf.common.utils.ordered_enum import OrderedEnum


class Backend(OrderedEnum):
    ONNX = 1
    PYTORCH = 2
    TENSORFLOW = 3
    OPENVINO = 4


def define_the_backend(model):
    from torch.nn import Module
    from tensorflow.keras.models import Model

    if isinstance(model, str):
        return InitializationAlgorithmPriority.ONNX
    elif isinstance(model, Module):
        return InitializationAlgorithmPriority.PYTORCH
    elif isinstance(model, Model):
        return InitializationAlgorithmPriority.TENSORFLOW
    elif isinstance(model, ):  # TODO: add OpenVINO
        return InitializationAlgorithmPriority.OPENVINO
    raise RuntimeError('This backend is not supported')
