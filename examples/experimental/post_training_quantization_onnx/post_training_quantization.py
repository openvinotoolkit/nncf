import argparse
import os
from typing import List
from typing import Union
import onnx

from examples.experimental.post_training_quantization_onnx.dataloader import create_train_dataloader

from nncf.experimental.onnx.quantization.algorithm import apply_post_training_quantization


def run(onnx_model_path: Union[str, bytes, os.PathLike], output_model_path: Union[str, bytes, os.PathLike],
        data_loader_path: Union[str, bytes, os.PathLike], dataset_name: str, num_init_samples: int,
        input_shape: List[int], per_channel: bool, ignored_scopes: List[str] = None):
    print("Quantization options:")
    print("  num_init_samples: ", num_init_samples)
    print("  per_channel quantization: ", per_channel)
    print("  ignored_scopes: ", ignored_scopes)
    onnx.checker.check_model(onnx_model_path)
    onnx_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")
    train_loader = create_train_dataloader(dataset_name, data_loader_path, input_shape)
    print(f"The dataloader is built with the data located on  {data_loader_path}")
    print("Post-Training Quantization is just started!")

    quantized_onnx_model = apply_post_training_quantization(onnx_model, train_loader, num_init_samples, ignored_scopes,
                                                            per_channel)
    onnx.save(quantized_onnx_model, output_model_path)
    print(f"The quantized model is saved on {output_model_path}")
    onnx.checker.check_model(output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", "-m", help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o", help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--data", help="Path to train data", type=str)
    parser.add_argument("--dataset_name", help="Dataset name", type=str)
    parser.add_argument("--input_shape", help="Model's input shape", nargs="+", type=int, default=[1, 3, 224, 224])
    parser.add_argument("--init_samples", help="Number of initialization samples", type=int, default=10)
    parser.add_argument("--ignored_scopes", help="Ignored operations ot quantize", nargs="+", default=None)
    parser.add_argument("--per_channel", help="per_channel Weights quantization", dest='per_channel',
                        action='store_true')
    parser.add_argument("--per_tensor", help="per_tensor Weights quantization", dest='per_channel',
                        action='store_false')
    parser.set_defaults(per_channel=True)

    args = parser.parse_args()
    run(args.onnx_model_path,
        args.output_model_path,
        args.data,
        args.dataset_name,
        args.init_samples,
        args.input_shape,
        args.per_channel,
        args.ignored_scopes
        )
