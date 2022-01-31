from examples.experimental.onnx.dataloader import create_train_dataloader
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.post_training_api.compression_builder import CompressionBuilder
from nncf.experimental.onnx.quantization.algorithm import ONNXPostTrainingQuantization
from nncf.experimental.post_training_api.quantization.config import DEFAULT

fp32_model_onnx_path = ''
int8_model_onnx_path = ''
dataset_dir = ''
input_shape = []

# Step 1: Initialize the data loader.
dataloader = create_train_dataloader(dataset_dir=dataset_dir, input_shape=input_shape)
# Step 2: Initialize the engine for metric calculation and statistics collection.
engine = ONNXEngine()

# Step 3: Create a pipeline of compression algorithms.
builder = CompressionBuilder()
quantization = ONNXPostTrainingQuantization(DEFAULT, engine, dataloader)
builder.add_algorithm(quantization)
# Step 4: Execute the pipeline.
compressed_model = builder.compress_model(fp32_model_onnx_path)
# Step 5: Export the compressed model.
compressed_model.export(int8_model_onnx_path)
