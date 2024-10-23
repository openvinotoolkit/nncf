# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

from time import time

import torch
import torch.fx
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
from torchvision import models

import nncf
import nncf.torch
from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e
from tests.torch.fx.helpers import visualize_fx_model


def measure_time(model, example_inputs, num_iters=3000):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(*example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def main(model_cls):
    model = model_cls()
    example_inputs = torch.ones((1, 3, 224, 224))
    exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    nncf_quantizer_model = quantize_pt2e(exported_model, quantizer, calibration_dataset=nncf.Dataset([example_inputs]))

    visualize_fx_model(nncf_quantizer_model, "nncf_quantizer_before_fold_resnet.svg")
    return nncf_quantizer_model

    # exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))
    # nncf_int8 = nncf.quantize(exported_model, nncf.Dataset([example_inputs]))
    # visualize_fx_model(nncf_int8, "nncf_resnet.svg")


def main_native(model_cls):
    model = model_cls()
    example_inputs = torch.ones((1, 3, 224, 224))
    exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(example_inputs)
    converted_model = convert_pt2e(prepared_model)
    visualize_fx_model(converted_model, "x86int8_resnet.svg")
    return converted_model


def constant_fold(m):
    pass


if __name__ == "__main__":
    with nncf.torch.disable_patching():
        for model_cls in (models.resnet18, models.mobilenet_v3_small, models.vit_b_16, models.swin_v2_s):
            # for model_cls in (models.mobilenet_v3_small,):
            print(f"{model_cls} check!")
            nncf_q_model = main(model_cls)

            constant_fold(nncf_q_model)
            visualize_fx_model(nncf_q_model, "nncf_quantizer_after_constant_fold_resnet.svg")

            pt_q_model = main_native(model_cls)
            print("benchmarking...")
            pt_compiled = torch.compile(model_cls())
            pt_int8_compiled = torch.compile(pt_q_model)
            nncf_comipled = torch.compile(nncf_q_model)

            example_inputs = (torch.ones((1, 3, 224, 224)),)

            pt_time = measure_time(pt_compiled, example_inputs)
            print(f"PT fp32 performance measured: {pt_time}")

            pt_int8_time = measure_time(pt_int8_compiled, example_inputs)
            print(f"PT int8 performance measured: {pt_int8_time}")

            nncf_int8_time = measure_time(nncf_comipled, example_inputs)
            print(f"NNCF int8 performance measured: {nncf_int8_time}")
