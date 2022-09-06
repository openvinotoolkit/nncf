"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from examples.experimental.onnx.common.sample_config import CustomArgumentParser


def get_common_argument_parser():
    """Defines command-line arguments, and parses them.

    """
    parser = CustomArgumentParser()

    parser.add_argument("--onnx_model_path", "-m",
                        help="Path to ONNX model", type=str, required=True)

    parser.add_argument("--output_model_path", "-o",
                        help="Path to output quantized ONNX model", type=str, required=True)

    parser.add_argument("--data",
                        help="Path to dataset", type=str, required=True)

    parser.add_argument("--input_shape",
                        help="Model's input shape set in ONNX model. e.g. [1, 3, 768, 960]",
                        nargs="+", type=int, default=None)

    parser.add_argument("--input_name",
                        help="Model's input name set in ONNX model. e.g. data_0",
                        type=str, default=None)

    parser.add_argument("--init_samples",
                        help="Number of initialization samples", type=int, default=300)

    parser.add_argument("--ignored_scopes",
                        help="Ignored operations ot quantize", nargs="+", default=None)

    return parser
