# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to save only model topology without weights.
Usage example: to save model topology in tests/onnx/data/models for future usage in graph tests.
"""
from argparse import ArgumentParser

import onnx

from tests.onnx.weightless_model import save_model_without_tensors

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_model")
    args = parser.parse_args()
    model = onnx.load(args.model)
    save_model_without_tensors(model, args.output_model)
