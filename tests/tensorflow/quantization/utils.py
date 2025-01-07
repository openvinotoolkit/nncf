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


from nncf import NNCFConfig


def get_basic_quantization_config(model_size=4):
    config = NNCFConfig()
    config.update(
        {
            "model": "basic_quant_conv",
            "input_info": {
                "sample_size": [1, model_size, model_size, 1],
            },
            "compression": {
                "algorithm": "quantization",
            },
        }
    )
    return config


def get_basic_asym_quantization_config(model_size=4):
    config = get_basic_quantization_config(model_size)
    config["compression"]["activations"] = {"mode": "asymmetric"}
    return config
