from examples.onnx.dataloader import create_train_dataloader
from nncf.experimental.onnx.engine import OnnxEngine
from nncf.experimental.post_training_api.compression_builder import CompressionBuilder
from nncf.experimental.onnx.quantization.algorithm import ONNXPostTrainingQuantization

fp32_model_onnx_path = ''
dataset_dir = ''
input_shape = []
dataloader = create_train_dataloader(dataset_dir=dataset_dir, input_shape=input_shape)
engine = OnnxEngine(providers=['OpenVINOExecutionProvider'])

builder = CompressionBuilder()

quantization_config = {

}

quantization = ONNXPostTrainingQuantization(quantization_config, engine, dataloader)

builder.add_algorithm(quantization)
compressed_model = builder.init(fp32_model_onnx_path)  # Transformed model
