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

# Enable torch inductor freezing feature first
import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"


import argparse
import copy
import time
from collections import defaultdict

import openvino.torch  # noqa
import torch

# Optional: using the C++ wrapper instead of default Python wrapper
import torch._inductor.config as config
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx.passes.graph_drawer import FxGraphDrawer

from nncf.experimental.torch_fx.model_transformer import QPARAMPerChannel
from nncf.experimental.torch_fx.model_transformer import QPARAMSPerTensor
from nncf.experimental.torch_fx.model_transformer import insert_qdq_to_model
from nncf.experimental.torch_fx.nncf_graph_builder import GraphConverter  # noqa


def get_exported_model_from_nn_module(module, example_inputs):
    with torch.no_grad():
        return capture_pre_autograd_graph(module, example_inputs)


NNCF_IMPL = True


def get_qsetup(exported_model, example_inputs):
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(*example_inputs)
    converted_model = convert_pt2e(prepared_model)
    g = FxGraphDrawer(converted_model, "resnet18_int8")
    g.get_dot_graph().write_svg("resnet18_int8_compiled.svg")
    qsetup = defaultdict(lambda: dict())

    for node in converted_model.graph.nodes:
        if "dequantize" in node.name:
            quantize = node.all_input_nodes[0]
            # place = "activations"
            # if len(quantize.all_input_nodes) > 1:
            #    place = "weights"
            if "per_tensor" in node.name:
                params = QPARAMSPerTensor(*node.args[1:])
            else:
                params = []
                for i in range(1, 3):
                    name = node.args[i].target
                    params.append(getattr(converted_model, name))
                params = QPARAMPerChannel(*(params + list(node.args[3:])))

            target_node_name = quantize.all_input_nodes[0].name
            qsetup[target_node_name] = params
    return qsetup


def quantize(model, example_inputs):
    if NNCF_IMPL:
        # Use NNCF here on exported model
        # to create a quantized model which is compatible with
        # convert_pt2e function
        pass
        # 1. Convert torch.graph to NNCFGraph.
        # # 2. Analize nncf grpah for SQ/CA
        # # 3. Collect statistics
        # # 4. Update params
        # 5. Analize nncf graph for quantization
        # 6. Insert observers
        # 7. prepared_model(*example_inputs)
        # 8. convert_pt2e(prepared_model)
        import nncf

        calibration_dataset = nncf.Dataset(example_inputs)
        exported_model = get_exported_model_from_nn_module(model, example_inputs)
        quantized_model = nncf.quantize(exported_model, calibration_dataset)
        g = FxGraphDrawer(quantized_model, "resnet18_quantized_native_nncf")
        g.get_dot_graph().write_svg("resnet18_quantized_native_nncf.svg")
        return quantized_model

    else:
        # g = FxGraphDrawer(exported_model, "resnet18")
        # g.get_dot_graph().write_svg("resnet18_compiled.svg")

        # MOCK NNCF QUANTIZATION
        exported_model = get_exported_model_from_nn_module(model, example_inputs)
        qsetup = get_qsetup(exported_model, example_inputs)
        exported_model = get_exported_model_from_nn_module(model, example_inputs)
        exported_model = insert_qdq_to_model(exported_model, qsetup)
        g = FxGraphDrawer(exported_model, "resnet18_int8")
        g.get_dot_graph().write_svg("resnet18_int8_compiled_manually.svg")
        return exported_model

    return None  # converted_model


config.cpp_wrapper = True


def measure_time(model, example_inputs, num_iters):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for i in range(0, num_iters):
            start_time = time.time()
            model(*example_inputs)
            total_time += time.time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def get_dummy_dataset():
    traced_bs = 1
    x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    example_inputs = (x,)
    return example_inputs


def main_nncf(model_name, num_iters):
    model = models.__dict__[model_name](pretrained=True)
    model = model.eval()

    example_inputs = get_dummy_dataset()
    import nncf

    calibration_dataset = nncf.Dataset(example_inputs)
    quantized_model = nncf.quantize(model, calibration_dataset)

    import openvino as ov

    ov_model = ov.convert_model(quantized_model.cpu(), example_input=example_inputs[0])
    ov.serialize(ov_model, "./model_cache_nncf/model.xml")


def main(model_name, num_iters):
    model = models.__dict__[model_name](pretrained=True)
    model = model.eval()

    example_inputs = get_dummy_dataset()

    converted_model = quantize(copy.deepcopy(model), example_inputs)

    print("original model execution time: ", measure_time(model, example_inputs, num_iters))

    native_optimized_model_fp32 = torch.compile(model)
    print(
        "Torch Inductor FP32 model execution time: ",
        measure_time(native_optimized_model_fp32, example_inputs, num_iters),
    )

    native_optimized_model_int8 = torch.compile(converted_model)
    print(
        "Torch Inductor INT8 model execution time: ",
        measure_time(native_optimized_model_int8, example_inputs, num_iters),
    )

    ov_optimized_model_fp32 = torch.compile(model, backend="openvino")
    print(
        "Torch.compile OpenVINO FP32 model execution time: ",
        measure_time(ov_optimized_model_fp32, example_inputs, num_iters),
    )

    ov_optimized_model_int8 = torch.compile(
        converted_model, backend="openvino", options={"model_caching": True, "cache_dir": "./model_cache"}
    )
    print(
        "Torch.compile OpenVINO INT8 model execution time: ",
        measure_time(ov_optimized_model_int8, example_inputs, num_iters),
    )

    import intel_extension_for_pytorch  # noqa

    ipex_optimized_model_fp32 = torch.compile(model, backend="ipex")
    print(
        "Torch.compile IPEX FP32 model execution time: ",
        measure_time(ipex_optimized_model_fp32, example_inputs, num_iters),
    )

    ipex_optimized_model_int8 = torch.compile(converted_model, backend="ipex")
    print(
        "Torch.compile IPEX INT8 model execution time: ",
        measure_time(ipex_optimized_model_int8, example_inputs, num_iters),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", help="number of inference iterations", type=int, default=100)
    parser.add_argument("--model", help="torchvision model name", type=str, default="resnet18")
    args = parser.parse_args()
    model_name = args.model
    num_iters = args.num_iters
    main(model_name, num_iters)
    # main_nncf(model_name, num_iters)
