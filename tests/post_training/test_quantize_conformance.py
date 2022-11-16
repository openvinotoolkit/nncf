from bz2 import compress
import os
import re
import logging
from pathlib import Path
import numpy as np

import torch
import nncf
import onnx
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import pytest

import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import openvino.runtime as ov

from model_scope import get_validation_scope

NOT_AVAILABLE_MESSAGE = 'N/A'
DEFAULT_VAL_THREADS = 4

def create_timm_model(name):
    model = timm.create_model(name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path="")
    return model

def get_model_transform(model):
    config = model.default_cfg
    normalize = transforms.Normalize(mean=config["mean"],
                                    std=config["std"])
    input_size = config["input_size"]
    resize_size = tuple([int(x/config["crop_pct"]) for x in input_size[-2:]])

    RESIZE_MODE_MAP = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }

    transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=RESIZE_MODE_MAP[config["interpolation"]]),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            normalize,
        ])

    return transform

def get_torch_dataloader(folder, transform, batch_size=1):
    val_dataset = datasets.ImageFolder(
        root=folder,
        transform = transform
    )
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    return val_loader


def export_to_onnx(model, save_path, data_sample):
    torch.onnx.export(model,
                      data_sample,
                      save_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=False)

def export_to_ir(model_path, save_path, model_name):
    command_line = f'mo -m {model_path} -o {save_path} -n {model_name}'
    os.popen(command_line).read()

def run_benchmark(model_path):
    command_line = f"benchmark_app -m {model_path} -d CPU -niter 300"
    output = os.popen(command_line).read()

    match = re.search("Throughput\: (.+?) FPS", output)
    if match != None:
        fps = match.group(1)
        return float(fps), output

    return None, output

def benchmark_performance(model_path, model_name):
    """
    Receives the OpenVINO IR model and runs benchmark tool for it
    """

    model_perf = NOT_AVAILABLE_MESSAGE

    try:
        model_perf, bench_output = run_benchmark(model_path)

        if model_perf == None:
            logging.info(f"Cannot measure performance for the model: {model_name}\nDetails: {bench_output}\n")
            model_perf = NOT_AVAILABLE_MESSAGE
    except BaseException as error:
        logging.error(f"Error when becnhmarking the model: model_name Details: {error}")

    return model_perf

def validate_accuracy(model_path, val_loader):
    dataset_size = len(val_loader)
    predictions = [0] * dataset_size
    references = [-1] * dataset_size

    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model)

    jobs = os.environ.get('NUM_VAL_THREADS', DEFAULT_VAL_THREADS)
    infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

    def process_result(request, userdata):
        result = request.get_output_tensor().data
        predicted_label = np.argmax(result, axis=1)
        predictions[userdata] = [predicted_label]

    infer_queue.set_callback(process_result)

    for i, (images, target) in tqdm(enumerate(val_loader)):
        infer_queue.start_async(images.numpy(), userdata=i)
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
    if isinstance(model, ov.Model):
        print(f"It is ov.Model")
    else:
        print(f"It is a different type: {type(model)}")

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
    return pytestconfig._test_result

@pytest.mark.parametrize("model_args", 
            get_validation_scope())
def test_ptq_timm(data, output, result, model_args):
    output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True) 
    torch.multiprocessing.set_sharing_strategy('file_system') # W/A to avoid RuntimeError

    model_name = model_args["name"]
    model_quantization_params = model_args["quantization_params"]
    try:
        ouput_folder = Path(output)

        model = create_timm_model(model_name)
        model.eval().cpu()
        transform = get_model_transform(model)

        batch_one_dataloader = get_torch_dataloader(data, transform, batch_size=1)
        # benchmark original models (once)
        orig_perf, orig_acc = benchmark_torch_model(model, batch_one_dataloader, model_name, ouput_folder)

        val_dataloader = get_torch_dataloader(data, transform, batch_size=128)
        def transform_fn(data_item):
            images, _ = data_item
            return images
        calibration_dataset = nncf.Dataset(val_dataloader, transform_fn)

        # quantize PyTorch model
        try:
            torch_quantized_model = nncf.quantize(model, calibration_dataset, **model_quantization_params)
            # benchmark quantized torch model
            torch_output_path = ouput_folder/ "torch"
            torch_output_path.mkdir(parents=True, exist_ok=True)
            q_torch_model_name = model_name + "_torch_int8"
            q_torch_perf, q_torch_acc = benchmark_torch_model(torch_quantized_model, batch_one_dataloader, q_torch_model_name, torch_output_path)
        except BaseException as error:
            q_torch_perf = str(error)
            q_torch_acc = str(error)

        # quantize ONNX model
        try:
            onnx_model_path = ouput_folder / (model_name + ".onnx")
            onnx_model = onnx.load(onnx_model_path)
            onnx_input_name = onnx_model.graph.input[0].name
            def onnx_transform_fn(data_item):
                images, _ = data_item
                return {onnx_input_name: images.numpy()}
            onnx_calibration_dataset = nncf.Dataset(batch_one_dataloader, onnx_transform_fn)

            onnx_quantized_model = nncf.quantize(onnx_model, onnx_calibration_dataset, **model_quantization_params)

            onnx_output_path = ouput_folder/ "onnx"
            onnx_output_path.mkdir(parents=True, exist_ok=True)
            q_onnx_model_name = model_name + "_onnx_int8"
            q_onnx_perf, q_onnx_acc = benchmark_onnx_model(onnx_quantized_model, batch_one_dataloader, q_onnx_model_name, onnx_output_path)
        except BaseException as error:
            q_onnx_perf = str(error)
            q_onnx_acc = str(error)

        # quantize OpenVINO model
        try:
            def ov_transform_fn(data_item):
                images, _ = data_item
                return images.numpy()
            ov_calibration_dataset = nncf.Dataset(batch_one_dataloader, ov_transform_fn)

            ov_model_path = ouput_folder / (model_name + ".xml")
            core = ov.Core()
            ov_model = core.read_model(ov_model_path)
            ov_quantized_model = nncf.quantize(ov_model, ov_calibration_dataset, **model_quantization_params)

            ov_output_path = ouput_folder/ "openvino"
            ov_output_path.mkdir(parents=True, exist_ok=True)
            q_ov_model_name = model_name + "_ov_int8"
            q_ov_perf, q_ov_acc = benchmark_ov_model(ov_quantized_model, batch_one_dataloader, q_ov_model_name, ov_output_path)
        except BaseException as error:
            q_ov_perf = str(error)
            q_ov_acc = str(error)


        result.append([model_name, orig_acc, q_torch_acc, q_onnx_acc, q_ov_acc,
                orig_perf, q_torch_perf, q_onnx_perf, q_ov_perf])
    except BaseException as error:
        result.append([model_name, error, "-", "-", "-",
                "-", "-", "-", "-"])
        logging.error(f"Error when running test for model: {model_name}. Error: {error}")
        assert False
