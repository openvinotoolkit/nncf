import onnx

from examples.onnx.dataloader import create_train_dataloader
from nncf.experimental.onnx.api.compressed_model import CompressedModel
from nncf.experimental.onnx.api.engine import OnnxEngine
from nncf.experimental.post_training_api.compression_builder import CompressionBuilder
from nncf.experimental.onnx.quantization.api import PostTrainingQuantizationBuilder

fp32_model_onnx_path = ''
dataset_dir = ''
input_shape = []
model = onnx.load(fp32_model_onnx_path)

compressed_model = CompressedModel(model)
dataloader = create_train_dataloader(dataset_dir=dataset_dir, input_shape=input_shape)

engine = OnnxEngine(dataloader, num_iters=100, providers=['OpenVINOExecutionProvider'])

builder = CompressionBuilder()

quantization_config = {

}

quantization_builder = PostTrainingQuantizationBuilder(quantization_config, engine)

builder.add_algorithm(quantization_builder)
builder.init(compressed_model)
