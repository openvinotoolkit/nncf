import argparse

import onnx
from examples.experimental.onnx.dataloader import create_train_dataloader

from nncf.experimental.onnx.quantization.algorithm import apply_post_training_quantization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", "-m", help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o", help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--data", help="Path to train data", type=str)
    parser.add_argument("--input_shape", help="Model's input shape", type=list, default=[1, 3, 224, 224])
    parser.add_argument("--init_samples", help="Number of initialization samples", type=int, default=10)

    args = parser.parse_args()
    onnx_model_path = args.onnx_model_path
    output_model_path = args.output_model_path
    data_loader_path = args.data
    num_init_samples = args.init_samples
    input_shape = args.input_shape

    onnx.checker.check_model(onnx_model_path)
    onnx_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")
    train_loader = create_train_dataloader(data_loader_path, input_shape)
    print(f"The dataloader is built with the data located on  {data_loader_path}")
    print("Post-Training Quantization is just started!")
    quantized_onnx_model = apply_post_training_quantization(onnx_model, train_loader, num_init_samples)
    onnx.save(quantized_onnx_model, output_model_path)
    print(f"The quantized model is saved on {output_model_path}")
    onnx.checker.check_model(output_model_path)
