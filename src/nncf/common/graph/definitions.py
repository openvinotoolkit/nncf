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

MODEL_INPUT_OP_NAME = "nncf_model_input"
MODEL_OUTPUT_OP_NAME = "nncf_model_output"
MODEL_CONST_OP_NAME = "nncf_model_const"


class NNCFGraphNodeType:
    INPUT_NODE = MODEL_INPUT_OP_NAME
    OUTPUT_NODE = MODEL_OUTPUT_OP_NAME
    CONST_NODE = MODEL_CONST_OP_NAME
