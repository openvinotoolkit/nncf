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

import argparse
import copy
import re
import subprocess
import time
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.torch  # noqa
import pandas as pd
import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torchvision.models as models
from sklearn.metrics import accuracy_score
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.jit import TracerWarning
from torchao.utils import benchmark_model as ao_benchmark_model
from torchvision import datasets
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification

import nncf
from nncf.common.logging.track_progress import track
from nncf.common.quantization.structs import QuantizationPreset  # noqa
from nncf.parameters import ModelType
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASET_IMAGENET = "/home/dlyakhov/datasets/imagenet/val"

hf_models = ()


def hf_model_builder(model_id: str):
    def build(weights):
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

        class ModelWithProcessing(torch.nn.Module):
            def __init__(self, processor, model):
                super().__init__()
                self.processor = processor
                self.model = model

            def forward(self, x):
                processed_input = processor(x, return_tensors="pt")
                return model(processed_input)

        # return ModelWithProcessing(processor, model)
        return model

    class DummyWeights:
        def transforms(self):
            return models.ResNet18_Weights.DEFAULT.transforms()

        @property
        def meta(self):
            return {}

    return build, DummyWeights()


MODELS_DICT = {
    "vit_h_14": (models.vit_h_14, models.ViT_H_14_Weights.DEFAULT),
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    "swin_v2_t": (models.swin_v2_t, models.Swin_V2_T_Weights.DEFAULT),
    "swin_v2_s": (models.swin_v2_s, models.Swin_V2_S_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    # "densenet161": (models.densenet161, models.DenseNet161_Weights.DEFAULT),
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
    "regnet_x_32gf": (models.regnet_x_32gf, models.RegNet_X_32GF_Weights.DEFAULT),
    # "google/vit-base-patch16-224": hf_model_builder("google/vit-base-patch16-224"),
    # "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
    # "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
}


def measure_time(model, example_inputs, num_iters=1000):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for i in range(0, num_iters):
            start_time = time.time()
            model(*example_inputs)
            total_time += time.time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def measure_time_ov(model, example_inputs, num_iters=1000):
    ie = ov.Core()
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(example_inputs)
    total_time = 0
    for i in range(0, num_iters):
        start_time = time.time()
        infer_request.infer(example_inputs)
        total_time += time.time() - start_time
    average_time = (total_time / num_iters) * 1000
    return average_time


def quantize(model, example_inputs, calibration_dataset, subset_size=300):
    with torch.no_grad():
        exported_model = capture_pre_autograd_graph(model, example_inputs)

    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    from tqdm import tqdm

    for inp, _ in islice(tqdm(calibration_dataset), subset_size):
        prepared_model(inp)
    converted_model = convert_pt2e(prepared_model)
    return converted_model


def validate(model, val_loader, subset_size=None):
    dataset_size = len(val_loader)

    predictions = np.zeros((dataset_size))
    references = -1 * np.ones((dataset_size))

    with track(total=dataset_size, description="Validation") as pbar:

        for i, (images, target) in enumerate(val_loader):
            if subset_size is not None and i >= subset_size:
                break

            output_data = model(images).detach().numpy()
            predicted_label = np.argmax(output_data, axis=1)
            predictions[i] = predicted_label.item()
            references[i] = target
            pbar.progress.update(pbar.task, advance=1)
    acc_top1 = accuracy_score(predictions, references) * 100
    print(acc_top1)
    return acc_top1


def validate_ov(model, val_loader):
    dataset_size = len(val_loader)

    # Initialize result tensors for async inference support.
    predictions = np.zeros((dataset_size))
    references = -1 * np.ones((dataset_size))

    core = ov.Core()
    compiled_model = core.compile_model(model)

    infer_queue = ov.AsyncInferQueue(compiled_model, 4)
    with track(total=dataset_size, description="Validation") as pbar:

        def process_result(request, userdata):
            output_data = request.get_output_tensor().data
            predicted_label = np.argmax(output_data, axis=1)
            predictions[userdata] = predicted_label.item()
            pbar.progress.update(pbar.task, advance=1)

        infer_queue.set_callback(process_result)

        for i, (images, target) in enumerate(val_loader):
            # W/A for memory leaks when using torch DataLoader and OpenVINO
            image_copies = copy.deepcopy(images.numpy())
            infer_queue.start_async(image_copies, userdata=i)
            references[i] = target

        infer_queue.wait_all()

    acc_top1 = accuracy_score(predictions, references) * 100
    print(acc_top1)
    return acc_top1


def run_benchmark(model_path: Path, shape) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    command += f' -shape="[{",".join(str(x) for x in shape)}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def torch_ao_sq_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input):
    import torch
    from torchao.quantization.smoothquant import smooth_fq_linear_to_inference
    from torchao.quantization.smoothquant import swap_linear_with_smooth_fq_linear

    # Fuse the int8*int8 -> int32 matmul and subsequent mul op avoiding materialization of the int32 intermediary tensor
    torch._inductor.config.force_fuse_int_mm_with_mul = True

    # plug in your model
    # model = torch.compile(pt_model)
    model = pt_model

    # convert linear modules to smoothquant
    # linear module in calibration mode
    swap_linear_with_smooth_fq_linear(model)

    # Create a data loader for calibration
    calibration_loader = val_loader

    # Calibrate the model
    model.train()
    from tqdm import tqdm

    for batch in tqdm(islice(calibration_loader, 300)):
        inputs = batch[0]
        model(inputs)

    # set it to inference mode
    smooth_fq_linear_to_inference(model)

    # compile the model to improve performance
    model = torch.compile(model, mode="max-autotune")
    acc1_quant_model = validate(model, val_loader)
    print(f"torch ao metric acc@1: {acc1_quant_model}")
    result["torch_ao_quant_model_acc"] = acc1_quant_model

    latency = ao_benchmark_model(model, 20, example_input)
    print(f"torch ao latency: {latency}")
    result["torch_ao_quant_model_latency"] = latency


def nncf_fx_2_ov_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input):
    with disable_patching():
        with torch.no_grad():
            exported_model = capture_pre_autograd_graph(pt_model, (example_input,))

        def transform(x):
            return x[0]

        quant_fx_model = nncf.quantize(
            exported_model, nncf.Dataset(val_loader, transform_func=transform), model_type=ModelType.TRANSFORMER
        )
        quant_compile_model = torch.compile(quant_fx_model, backend="openvino")

        # acc1_quant_model = validate(quant_compile_model, val_loader)
        acc1_quant_model = -1.0
        latency_fx = measure_time(quant_compile_model, (example_input,))
        print(f"latency: {latency_fx}")
        result["acc1_nncf_fx_quant_model"] = acc1_quant_model
        result["torch_compile_ov_latency_nncf_fx_quant_model"] = latency_fx

        g = FxGraphDrawer(quant_compile_model, f"b_nncf_{pt_model.__class__.__name__}_int8")
        g.get_dot_graph().write_svg(f"b_nncf_{pt_model.__class__.__name__}_int8.svg")

        # EXPORT TO OV
        exported_model = torch.export.export(quant_compile_model, (example_input,))
        ov_quant_model = ov.convert_model(exported_model, example_input=example_input)
        quant_file_path = output_dir / "quant.xml"
        ov.save_model(ov_quant_model, quant_file_path)

        fps = run_benchmark(quant_file_path, shape_input)
        print(f"fps: {fps}")
        result["ov_fps_nncf_fx_quant_model"] = fps


def fx_2_ov_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input):
    with disable_patching():
        fp32_pt_model = copy.deepcopy(pt_model)
        fp32_compile_model = torch.compile(fp32_pt_model, backend="openvino")

        quant_pt_model = quantize(fp32_compile_model, (example_input,), val_loader)
        quant_compile_model = torch.compile(quant_pt_model, backend="openvino")

        g = FxGraphDrawer(quant_pt_model, f"b_pt_{pt_model.__class__.__name__}_int8")
        g.get_dot_graph().write_svg(f"b_pt_{pt_model.__class__.__name__}_int8.svg")

        acc1_quant_model = validate(quant_compile_model, val_loader)
        result["acc1_quant_model"] = acc1_quant_model

        latency_fx = measure_time(quant_compile_model, (example_input,))
        print(f"latency: {latency_fx}")
        result["torch_compile_latency_fps_quant_model"] = latency_fx


def nncf_pt_2_ov_quantization(pt_model, val_loader, example_input, output_dir, result, shape_input):
    def transform(x):
        return x[0]

    nncf_model = nncf.quantize(copy.deepcopy(pt_model), nncf.Dataset(val_loader, transform_func=transform))

    ov_nncf_model = ov.convert_model(nncf_model, example_input=example_input)
    nncf_pt_file_path = output_dir / "nncf_pt.xml"
    ov.save_model(ov_nncf_model, nncf_pt_file_path)
    acc1_nncf_pt = validate_ov(ov_nncf_model, val_loader)
    result["acc1_nncf_pt"] = acc1_nncf_pt
    fps = run_benchmark(nncf_pt_file_path, shape_input)
    print(f"fps: {fps}")
    result["ov_fps_nncf_pt"] = fps


def nncf_ov_2_ov_quantization(ov_fp32_model, val_loader, output_dir, result, shape_input):
    def transform(x):
        return np.array(x[0])

    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
    from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters

    advanced_params = AdvancedQuantizationParameters()
    # for sq_param in [-1, 0.15, 0.5, 0.75]:
    for sq_param in [0.95]:
        advanced_params.smooth_quant_alphas = AdvancedSmoothQuantParameters(matmul=sq_param)

        from copy import deepcopy

        fast_bias_correction = True
        nncf_ov_int8_model = nncf.quantize(
            deepcopy(ov_fp32_model),
            nncf.Dataset(val_loader, transform_func=transform),
            fast_bias_correction=fast_bias_correction,
            model_type=ModelType.TRANSFORMER,
            preset=QuantizationPreset.MIXED,
            advanced_parameters=advanced_params,
        )
        acc1_nncf_ov = validate_ov(nncf_ov_int8_model, val_loader)
        result[f"acc1_nncf_ov_{sq_param}"] = acc1_nncf_ov
        for precision, model in (("int8", nncf_ov_int8_model), ("fp32", ov_fp32_model)):
            nncf_ov_file_path = output_dir / f"nncf_ov_{precision}.xml"
            ov.save_model(model, nncf_ov_file_path)
            fps = run_benchmark(nncf_ov_file_path, shape_input)
            print(f"fps_{precision}: {fps} {sq_param}")
            result[f"ov_fps_nncf_ov_{precision}_{sq_param}"] = fps

            latency = measure_time_ov(model, next(iter(val_loader))[0], num_iters=10_000)
            print(f"latency_{precision}: {latency}")
            result[f"ov_latency_nncf_ov_{precision}_{sq_param}"] = latency


def process_model(model_name: str):

    result = {"name": model_name}
    model_cls, model_weights = MODELS_DICT[model_name]
    output_dir = Path("models") / model_name
    output_dir.mkdir(exist_ok=True)
    ##############################################################
    # Prepare dataset
    ##############################################################

    val_dataset = datasets.ImageFolder(root=DATASET_IMAGENET, transform=model_weights.transforms())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    ##############################################################
    # Prepare original model
    ##############################################################

    pt_model = model_cls(weights=model_weights)
    pt_model = pt_model.eval()
    example_input = next(iter(val_loader))[0]
    shape_input = list(example_input.shape)
    ##############################################################
    # Process FP32 Model
    ##############################################################

    fp32_pt_model = copy.deepcopy(pt_model)

    orig_infer_acc1 = model_weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
    print(f"fp32 model metric: {orig_infer_acc1}")
    # orig_infer_acc1 = validate(fp32_pt_model, val_loader)
    result["acc1_fp32_openvino"] = orig_infer_acc1

    fp32_pt_model = torch.export.export(fp32_pt_model, (example_input,))
    ov_fp32_model = ov.convert_model(fp32_pt_model, example_input=example_input)
    ov_fp32_file_path = None
    ov_fp32_file_path = output_dir / "fp32.xml"
    ov.save_model(ov_fp32_model, ov_fp32_file_path)
    # result["fps_fp32_openvino"] = run_benchmark(ov_fp32_file_path, shape_input)
    # print(f"fps_fp32_openvino {result['fps_fp32_openvino']}")

    del fp32_pt_model
    ##############################################################
    # Process Torch AO Quantize with SQ
    ##############################################################
    # torch_ao_sq_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input)

    ##############################################################
    # with torch.no_grad():
    #    exported_model = capture_pre_autograd_graph(pt_model, (example_input,))
    #    latency_fx = measure_time(torch.compile(exported_model), (example_input,))
    # print(f"latency: {latency_fx}")
    #############################################################

    ##############################################################
    # Process PT Quantize
    ##############################################################
    fx_2_ov_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input)

    ##############################################################
    # Process NNCF FX Quantize
    ##############################################################
    # nncf_fx_2_ov_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input)

    ##############################################################
    # Process NNCF Quantize by PT
    ##############################################################
    # nncf_pt_2_ov_quantization(pt_model, val_loader, example_input, output_dir, result, shape_input)

    ##############################################################
    # Process NNCF Quantize by OV
    ##############################################################
    # nncf_ov_2_ov_quantization(ov_fp32_model, val_loader, output_dir, result, shape_input)

    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="torchvision model name", type=str, default="all")
    parser.add_argument("--file_name", help="output csv file_name", type=str, default="result.csv")

    args = parser.parse_args()

    results_list = []
    if args.model == "all":
        for model_name in MODELS_DICT:
            print("---------------------------------------------------")
            print(f"name: {model_name}")
            results_list.append(process_model(model_name))
    else:
        results_list.append(process_model(args.model))

    df = pd.DataFrame(results_list)
    print(df)
    df.to_csv(args.file_name)


if __name__ == "__main__":
    main()
