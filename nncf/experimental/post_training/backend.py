from nncf.common.utils.ordered_enum import OrderedEnum


class BACKEND(OrderedEnum):
    ONNX = 1
    PYTORCH = 2
    TENSORFLOW = 3
    OPENVINO = 4
