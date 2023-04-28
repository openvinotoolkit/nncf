"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the 'License');
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an 'AS IS' BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import copy
import logging
import os
import re
import traceback
from multiprocessing import Pipe
from multiprocessing import Process
from pathlib import Path
from pathlib import PosixPath
from typing import Optional

import numpy as np
import onnx
import openvino.runtime as ov
import pytest
import timm
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import nncf
from nncf.experimental.torch.quantization.quantize import quantize_impl as pt_impl_experimental
from nncf.openvino.quantization.quantize_model import quantize_impl as ov_quantize_impl
from nncf.torch.nncf_network import NNCFNetwork
from tests.post_training.conftest import PipelineType
from tests.post_training.conftest import RunInfo
from tests.post_training.model_scope import VALIDATION_SCOPE
from tests.shared.command import Command

NOT_AVAILABLE_MESSAGE = "N/A"
DEFAULT_VAL_THREADS = 4


def create_timm_model(name):
    model = timm.create_model(name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path="")
    return model


def get_model_transform(model):
    config = model.default_cfg
    transformations_list = []
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
    input_size = config["input_size"]

    RESIZE_MODE_MAP = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }

    if "fixed_input_size" in config and not config["fixed_input_size"]:
        resize_size = tuple(int(x / config["crop_pct"]) for x in input_size[-2:])
        resize = transforms.Resize(resize_size, interpolation=RESIZE_MODE_MAP[config["interpolation"]])
        transformations_list.append(resize)
    transformations_list.extend([transforms.CenterCrop(input_size[-2:]), transforms.ToTensor(), normalize])

    transform = transforms.Compose(transformations_list)

    return transform


def get_torch_dataloader(folder, transform, batch_size=1):
    val_dataset = datasets.ImageFolder(root=folder, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    return val_loader


def export_to_onnx(model, save_path, data_sample):
    torch.onnx.export(
        model,
        data_sample,
        save_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=False,
    )


def export_to_ir(model_path, save_path, model_name):
    runner = Command(f"mo -m {model_path} -o {save_path} -n {model_name}")
    runner.run()


def run_benchmark(model_path):
    runner = Command(f"benchmark_app -m {model_path} -d CPU -niter 300")
    runner.run()
    cmd_output = " ".join(runner.output)

    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    if match is not None:
        fps = match.group(1)
        return float(fps), cmd_output

    return None, cmd_output


def benchmark_performance(model_path, model_name):
    """
    Receives the OpenVINO IR model and runs benchmark tool for it
    """
    model_perf = NOT_AVAILABLE_MESSAGE

    try:
        model_perf, bench_output = run_benchmark(model_path)

        if model_perf is None:
            logging.info(f"Cannot measure performance for the model: {model_name}\nDetails: {bench_output}\n")
            model_perf = NOT_AVAILABLE_MESSAGE
    except BaseException as error:
        logging.error(f"Error when becnhmarking the model: {model_name} Details: {error}")

    return model_perf


def validate_accuracy(model_path, val_loader):
    dataset_size = len(val_loader)
    predictions = [0] * dataset_size
    references = [-1] * dataset_size

    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model)

    jobs = int(os.environ.get("NUM_VAL_THREADS", DEFAULT_VAL_THREADS))
    infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

    def process_result(request, userdata):
        output_data = request.get_output_tensor().data
        predicted_label = np.argmax(output_data, axis=1)
        predictions[userdata] = [predicted_label]

    infer_queue.set_callback(process_result)

    for i, (images, target) in enumerate(val_loader):
        # W/A for memory leaks when using torch DataLoader and OpenVINO
        image_copies = copy.deepcopy(images.numpy())
        infer_queue.start_async(image_copies, userdata=i)
        references[i] = target

    infer_queue.wait_all()
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


def benchmark_torch_model(model, dataloader, model_name, output_path):
    data_sample, _ = next(iter(dataloader))
    # Dump model
    onnx_path = Path(output_path) / (model_name + ".onnx")
    export_to_onnx(model, onnx_path, data_sample)
    ov_path = Path(output_path) / (model_name + ".xml")
    export_to_ir(onnx_path, output_path, model_name)

    # Benchmark performance
    performance = benchmark_performance(ov_path, model_name)
    # Validate accuracy
    accuracy = validate_accuracy(ov_path, dataloader)
    return performance, accuracy


def benchmark_onnx_model(model, dataloader, model_name, output_path):
    # Dump model
    onnx_path = Path(output_path) / (model_name + ".onnx")
    onnx.save(model, onnx_path)
    ov_path = Path(output_path) / (model_name + ".xml")
    export_to_ir(onnx_path, output_path, model_name)

    # Benchmark performance
    performance = benchmark_performance(ov_path, model_name)
    # Validate accuracy
    accuracy = validate_accuracy(ov_path, dataloader)
    return performance, accuracy


def benchmark_ov_model(model, dataloader, model_name, output_path):
    # Dump model
    ov_path = Path(output_path) / (model_name + ".xml")
    ov.serialize(model, str(ov_path))

    # Benchmark performance
    performance = benchmark_performance(ov_path, model_name)
    # Validate accuracy
    accuracy = validate_accuracy(ov_path, dataloader)
    return performance, accuracy


@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session")
def output(pytestconfig):
    return pytestconfig.getoption("output")


@pytest.fixture(scope="session")
def result(pytestconfig):
    return pytestconfig.test_results


def quantize_ov_native(
    model: ov.Model,
    calibration_dataset: nncf.Dataset,
    preset: nncf.QuantizationPreset = nncf.QuantizationPreset.PERFORMANCE,
    target_device: nncf.TargetDevice = nncf.TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[nncf.ModelType] = None,
    ignored_scope: Optional[nncf.IgnoredScope] = None,
) -> ov.Model:
    quantized_model = ov_quantize_impl(
        model,
        calibration_dataset,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
    )
    return quantized_model


def quantize_torch_ptq(
    model: torch.nn.Module,
    calibration_dataset: nncf.Dataset,
    preset: nncf.QuantizationPreset = nncf.QuantizationPreset.PERFORMANCE,
    target_device: nncf.TargetDevice = nncf.TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[nncf.ModelType] = None,
    ignored_scope: Optional[nncf.IgnoredScope] = None,
) -> NNCFNetwork:
    quantized_model = pt_impl_experimental(
        model, calibration_dataset, preset, target_device, subset_size, fast_bias_correction, model_type, ignored_scope
    )
    return quantized_model


def torch_runner(
    model, calibration_dataset, model_quantization_params, output_folder, model_name, batch_one_dataloader
) -> RunInfo:
    torch_quantized_model = nncf.quantize(model, calibration_dataset, **model_quantization_params)
    # benchmark quantized torch model
    torch_output_path = output_folder / "torch"
    torch_output_path.mkdir(parents=True, exist_ok=True)
    q_torch_model_name = model_name + "_torch_int8"
    q_torch_perf, q_torch_acc = benchmark_torch_model(
        torch_quantized_model,
        batch_one_dataloader,
        q_torch_model_name,
        torch_output_path,
    )
    return RunInfo(q_torch_acc, q_torch_perf)


def torch_ptq_runner(
    model, calibration_dataset, model_quantization_params, output_folder, model_name, batch_one_dataloader
) -> RunInfo:
    def transform_fn(data_item):
        images, _ = data_item
        return images

    calibration_dataset = nncf.Dataset(batch_one_dataloader, transform_fn)

    torch_quantized_model = quantize_torch_ptq(model, calibration_dataset, **model_quantization_params)
    # benchmark quantized torch model
    torch_output_path = output_folder / "torch_ptq"
    torch_output_path.mkdir(parents=True, exist_ok=True)
    q_torch_model_name = model_name + "_torch_ptq_int8"
    q_torch_ptq_perf, q_torch_ptq_acc = benchmark_torch_model(
        torch_quantized_model,
        batch_one_dataloader,
        q_torch_model_name,
        torch_output_path,
    )

    return RunInfo(q_torch_ptq_acc, q_torch_ptq_perf)


def onnx_runner(model, calibration_dataset, model_quantization_params, output_folder, model_name, batch_one_dataloader):
    onnx_model_path = output_folder / (model_name + ".onnx")
    onnx_model = onnx.load(onnx_model_path)
    onnx_input_name = onnx_model.graph.input[0].name

    def onnx_transform_fn(data_item):
        images, _ = data_item
        return {onnx_input_name: images.numpy()}

    onnx_calibration_dataset = nncf.Dataset(batch_one_dataloader, onnx_transform_fn)

    onnx_quantized_model = nncf.quantize(onnx_model, onnx_calibration_dataset, **model_quantization_params)

    onnx_output_path = output_folder / "onnx"
    onnx_output_path.mkdir(parents=True, exist_ok=True)
    q_onnx_model_name = model_name + "_onnx_int8"
    q_onnx_perf, q_onnx_acc = benchmark_onnx_model(
        onnx_quantized_model,
        batch_one_dataloader,
        q_onnx_model_name,
        onnx_output_path,
    )
    return RunInfo(q_onnx_acc, q_onnx_perf)


def ov_native_runner(
    model, calibration_dataset, model_quantization_params, output_folder, model_name, batch_one_dataloader
):
    ov_native_model_path = output_folder / (model_name + ".xml")
    core = ov.Core()
    ov_native_model = core.read_model(ov_native_model_path)

    input_names = set(inp.get_any_name() for inp in ov_native_model.inputs)
    if len(ov_native_model.inputs) != 1:
        RuntimeError("Number of inputs != 1")

    def ov_native_transform_fn(data_item):
        images, _ = data_item
        return {next(iter(input_names)): images.numpy()}

    ov_native_calibration_dataset = nncf.Dataset(batch_one_dataloader, ov_native_transform_fn)

    ov_native_quantized_model = quantize_ov_native(
        ov_native_model, ov_native_calibration_dataset, **model_quantization_params
    )

    ov_native_output_path = output_folder / "openvino_native"
    ov_native_output_path.mkdir(parents=True, exist_ok=True)
    q_ov_native_model_name = model_name + "_openvino_native_int8"
    q_ov_native_perf, q_ov_native_acc = benchmark_ov_model(
        ov_native_quantized_model,
        batch_one_dataloader,
        q_ov_native_model_name,
        ov_native_output_path,
    )
    return RunInfo(q_ov_native_acc, q_ov_native_perf)


def ov_runner(
    model, calibration_dataset, model_quantization_params, output_folder, model_name, batch_one_dataloader
) -> RunInfo:
    def ov_transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    ov_calibration_dataset = nncf.Dataset(batch_one_dataloader, ov_transform_fn)
    ov_model_path = output_folder / (model_name + ".xml")
    core = ov.Core()
    ov_model = core.read_model(ov_model_path)
    ov_quantized_model = nncf.quantize(ov_model, ov_calibration_dataset, **model_quantization_params)

    ov_output_path = output_folder / "openvino"
    ov_output_path.mkdir(parents=True, exist_ok=True)
    q_ov_model_name = model_name + "_openvino_int8"
    q_ov_perf, q_ov_acc = benchmark_ov_model(
        ov_quantized_model,
        batch_one_dataloader,
        q_ov_model_name,
        ov_output_path,
    )
    return RunInfo(q_ov_acc, q_ov_perf)


RUNNERS = {
    PipelineType.TORCH: torch_runner,
    PipelineType.TORCH_PTQ: torch_ptq_runner,
    PipelineType.ONNX: onnx_runner,
    PipelineType.OV_NATIVE: ov_native_runner,
    PipelineType.OV: ov_runner,
}


def run_ptq_timm(
    data, output, timm_model_name, backends, model_quantization_params, process_connection, report_model_name
):  # pylint: disable=W0703
    torch.multiprocessing.set_sharing_strategy("file_system")  # W/A to avoid RuntimeError

    runinfos = {}
    try:
        output_folder = Path(output)
        output_folder.mkdir(parents=True, exist_ok=True)

        model = create_timm_model(timm_model_name)
        model.eval().cpu()
        transform = get_model_transform(model)

        model_name = report_model_name

        batch_one_dataloader = get_torch_dataloader(data, transform, batch_size=1)
        # benchmark original models (once)
        orig_perf, orig_acc = benchmark_torch_model(model, batch_one_dataloader, model_name, output_folder)
        runinfos[PipelineType.FP32] = RunInfo(orig_acc, orig_perf)

        val_dataloader = get_torch_dataloader(data, transform, batch_size=128)

        def transform_fn(data_item):
            images, _ = data_item
            return images

        calibration_dataset = nncf.Dataset(val_dataloader, transform_fn)

        for backend in backends:
            runner = RUNNERS[backend]
            try:
                runinfo = runner(
                    model,
                    calibration_dataset,
                    model_quantization_params,
                    output_folder,
                    model_name,
                    batch_one_dataloader,
                )
            except Exception as error:
                backend_dir = backend.value.replace(" ", "_")
                traceback_path = Path.joinpath(output_folder, backend_dir, model_name + "_error_log.txt")
                create_error_log(traceback_path)
                status = get_error_msg(traceback_path, backend_dir)
                runinfo = RunInfo(-1, -1, status)
            runinfos[backend] = runinfo

        process_connection.send(runinfos)
    except Exception as error:
        traceback_path = Path.joinpath(output_folder, model_name + "_error_log.txt")
        create_error_log(traceback_path)
        status = f"{model_name} traceback: {traceback_path}"
        runinfos[PipelineType.FP32] = RunInfo(-1, -1, status)
        process_connection.send(runinfos)
        raise error


def create_error_log(traceback_path: PosixPath) -> None:
    traceback_path.parents[0].mkdir(parents=True, exist_ok=True)
    with open(traceback_path, "w") as file:
        traceback.print_exc(file=file)
    file.close()
    logging.error(traceback.format_exc())


def get_error_msg(traceback_path: PosixPath, backend_name: str) -> str:
    return f"{backend_name} traceback: {traceback_path}"


@pytest.mark.parametrize("model_args", VALIDATION_SCOPE, ids=[desk["name"] for desk in VALIDATION_SCOPE])
def test_ptq_timm(data, output, result, model_args, backends_list):  # pylint: disable=W0703
    backends = [PipelineType[backend] for backend in backends_list.split(",")]
    model_name = model_args["name"]
    report_model_name = model_args.get("report_model_name", model_name)
    quantization_params = model_args["quantization_params"]
    main_connection, process_connection = Pipe()
    process = Process(
        target=run_ptq_timm,
        args=(data, output, model_name, backends, quantization_params, process_connection, report_model_name),
    )
    process.start()
    process.join()
    result[report_model_name] = main_connection.recv()
    if process.exitcode:
        assert False
