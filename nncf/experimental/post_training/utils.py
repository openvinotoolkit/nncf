from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.backend import BACKEND


def merge_two_dicts(x, y):
    # TODO: check intersections keys
    z = x.copy()
    z.update(y)
    return z


def export(compressed_model: CompressedModel, path: str):
    if compressed_model.model_backend == BACKEND.ONNX:
        import onnx
        onnx.save_model(compressed_model.compressed_model, path)
