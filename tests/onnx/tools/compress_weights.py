from argparse import ArgumentParser

import onnx

from nncf.onnx.graph.model_utils import compress_quantize_weights_transformation

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("input_model")
    arg_parser.add_argument("output_model")
    args = arg_parser.parse_args()

    model = onnx.load(args.input_model)
    q_model = compress_quantize_weights_transformation(model)
    onnx.save_model(q_model, args.output_model)
    onnx.checker.check_model(q_model)
