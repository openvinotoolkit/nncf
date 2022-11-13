import onnx
from onnx.version_converter import convert_version, ConvertError  # pylint: disable=no-name-in-module
from nncf.common.utils.logger import logger as nncf_logger

TARGET_OPSET_VERSION = 13
TARGET_IR_VERSION = 7


def convert_opset_version(model: onnx.ModelProto, opset_version: int = TARGET_OPSET_VERSION) -> onnx.ModelProto:
    """
    Tries to convert 'model' Opset Version to 'opset_version'.
    If the 'model' can not be converted returns the original 'model'.
    :param model: ONNX model to convert.
    :param opset_version: target Opset Version.
    :return: Converted ONNX model or Original ONNX model.
    """
    # pylint: disable=no-member
    try:
        modified_model = convert_version(model, opset_version)
        onnx.checker.check_model(modified_model)
        nncf_logger.info(
            'The model was successfully converted  to the Opset Version = {}'.format(
                modified_model.opset_import[0].version))
        return modified_model
    except (RuntimeError, ConvertError):
        nncf_logger.error(
            f"Couldn't convert target model to the Opset Version {opset_version}. "
            f"Using the copy of the original model")
        return model


def convert_ir_version(model: onnx.ModelProto, ir_version: int = TARGET_IR_VERSION) -> onnx.ModelProto:
    """
    Creates a new model from the 'model' graph with the target IR Version.
    :param model: ONNX model to convert.
    :param ir_version: Target IR Version.
    :return: Converted ONNX model.
    """
    op = onnx.OperatorSetIdProto()
    op.version = model.opset_import[0].version
    modified_model = onnx.helper.make_model(model.graph, ir_version=ir_version, opset_imports=[op])
    onnx.checker.check_model(modified_model)
    nncf_logger.info(
        'The model was successfully converted  to the IR Version = {}'.format(
            modified_model.ir_version))
    return modified_model
