from typing import TypeVar

from nncf.common.utils.ordered_enum import OrderedEnum

ModelType = TypeVar('ModelType')


class Backend(OrderedEnum):
    ONNX = 1
    PYTORCH = 2
    TENSORFLOW = 3
    OPENVINO = 4


def determine_model_backend(model: ModelType) -> Backend:
    from onnx import ModelProto
    if isinstance(model, ModelProto):
        return Backend.ONNX
    raise RuntimeError('This backend is not supported')
