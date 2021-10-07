import argparse

import onnx
from google.protobuf.json_format import MessageToDict
from examples.experimental.onnx.dataloader import create_train_dataloader

from nncf.experimental.onnx.quantization.algorithm import apply_post_training_quantization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model_path", help="Path to ONNX model", type=str)
    parser.add_argument("output_model_path", help="Path to quantized ONNX model", type=str)
    parser.add_argument("data", help="Path to train dataloader", type=str)
    parser.add_argument("--init_samples", help="Number of initialization samples", type=int, default=10)
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_model_path)
    num_init_samples = args.init_samples
    data_loader_path = args.data
    output_model_path = args.output_model_path

    for _input in onnx_model.graph.input:
        m_dict = MessageToDict(_input)
        dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
        input_shape = [int(d.get("dimValue")) for d in dim_info]
    train_loader = create_train_dataloader(data_loader_path, input_shape)

    quantized_onnx_model = apply_post_training_quantization(onnx_model, train_loader, num_init_samples)
    onnx.save(quantized_onnx_model, output_model_path)
