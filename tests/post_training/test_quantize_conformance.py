from abc import ABCMeta, abstractmethod
from typing import Callable
from multiprocessing import Pool

import os
import re
import logging
from pathlib import Path
import copy
import numpy as np
import pytest

import nncf
import torch
import onnx

from sklearn.metrics import accuracy_score

import openvino.runtime as ov

from model_scope import get_validation_scope
from utils import create_timm_model
from utils import get_model_transform
from utils import get_torch_dataloader
from utils import export_to_onnx
from utils import export_to_ir
from utils import run_benchmark


NOT_AVAILABLE_MESSAGE = "N/A"
DEFAULT_VAL_THREADS = 4
ILSVRC_VALIDATION_SUBSET_SIZE = 500

class QuantizerBase(metaclass=ABCMeta):
    """ Base class for an executor """
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def quantize(self, ):
        """ Abstract method to quantize the model in a certain framework """
        return False
 
class QuantizerFactory:
    """ The factory class for creating quantizers"""
 
    registry = {}
    """ Internal registry for available quantizers """
 
    @classmethod
    def register(cls, target_type=None) -> Callable:
        def inner_wrapper(wrapped_class: QuantizerBase) -> Callable:
            registered_name = wrapped_class.__name__ if target_type == None else target_type
            if registered_name in cls.registry:
                logging.warning('Executor %s already exists. Will replace it', registered_name)
            cls.registry[registered_name] = wrapped_class
            return wrapped_class
 
        return inner_wrapper
 
    @classmethod
    def quantize(cls, quantizer_name, *args, **kwargs):
        if quantizer_name not in cls.registry:
            raise ValueError("Quantizer type does not exist")
 
        quantizer = cls.registry[quantizer_name](quantizer_name)
        return quantizer.quantize(*args, **kwargs)

    @classmethod
    def validate(cls, quantizer_name, *args, **kwargs):
        if quantizer_name not in cls.registry:
            raise ValueError("Quantizer type does not exist")
 
        quantizer = cls.registry[quantizer_name](quantizer_name)
        return quantizer.validate(*args, **kwargs)

@QuantizerFactory.register("torch")
class TorchQuantizer(QuantizerBase):
    """ Torch quantizer class"""

    def quantize(self, model, dataloader, quantization_params):
        def transform_fn(data_item):
                images, _ = data_item
                return images

        calibration_dataset = nncf.Dataset(dataloader, transform_fn)

        quantized_model = nncf.quantize(
            model, calibration_dataset, **quantization_params
        )
        return quantized_model

    def validate(self, model, dataloader, model_name, output_folder, quantized=True):    
        torch_model_name = model_name
        torch_output_path = output_folder

        if quantized:
            torch_output_path = torch_output_path / "torch"
            torch_output_path.mkdir(parents=True, exist_ok=True)
            torch_model_name = torch_model_name + "_torch"
            torch_model_name = torch_model_name + "_int8"

        data_sample, _ = next(iter(dataloader))
        # Dump model
        onnx_path = Path(torch_output_path) / (torch_model_name + ".onnx")
        export_to_onnx(model, onnx_path, data_sample)
        ov_path = Path(torch_output_path) / (torch_model_name + ".xml")
        export_to_ir(onnx_path, torch_output_path, torch_model_name)
        # Validate accuracy
        accuracy = validate_accuracy(ov_path, dataloader)
        return ov_path, accuracy, torch_model_name


@QuantizerFactory.register("onnx")
class ONNXQuantizer(QuantizerBase):
    """ ONNX quantizer class"""

    def quantize(self, model, dataloader, quantization_params):
        onnx_input_name = model.graph.input[0].name

        def transform_fn(data_item):
                images, _ = data_item
                return {onnx_input_name: images.numpy()}

        calibration_dataset = nncf.Dataset(dataloader, transform_fn)

        quantized_model = nncf.quantize(
            model, calibration_dataset, **quantization_params
        )
        return quantized_model

    def validate(self, model, dataloader, model_name, output_folder):
        output_path = output_folder / "onnx"
        output_path.mkdir(parents=True, exist_ok=True)
        q_model_name = model_name + "_onnx_int8"

        # Dump model
        onnx_path = Path(output_path) / (q_model_name + ".onnx")
        onnx.save(model, onnx_path)
        ov_path = Path(output_path) / (q_model_name + ".xml")
        export_to_ir(onnx_path, output_path, q_model_name)
        # Validate accuracy
        accuracy = validate_accuracy(ov_path, dataloader)
        return ov_path, accuracy, q_model_name


@QuantizerFactory.register("openvino")
class OpenVINOQuantizer(QuantizerBase):
    """ OpenVINO quantizer class"""

    def quantize(self, model, dataloader, quantization_params):
        def transform_fn(data_item):
            images, _ = data_item
            return images.numpy()

        calibration_dataset = nncf.Dataset(dataloader, transform_fn)
        quantized_model = nncf.quantize(
            model, calibration_dataset, **quantization_params
        )
        return quantized_model

    def validate(self, model, dataloader, model_name, output_folder):
        output_path = output_folder / "openvino"
        output_path.mkdir(parents=True, exist_ok=True)
        q_model_name = model_name + "_ov_int8"

        # Dump model
        ov_path = Path(output_path) / (q_model_name + ".xml")
        ov.serialize(model, str(ov_path))
        # Validate accuracy
        accuracy = validate_accuracy(ov_path, dataloader)

        return ov_path, accuracy, q_model_name


def benchmark_performance(model_path, model_name):
    """
    Receives the OpenVINO IR model and runs benchmark tool for it
    """

    model_perf = NOT_AVAILABLE_MESSAGE

    try:
        model_perf, bench_output = run_benchmark(model_path)

        if model_perf is None:
            logging.error(
                f"Cannot measure performance for the model: {model_name}.\nDetails: {bench_output}\n"
            )
            model_perf = NOT_AVAILABLE_MESSAGE
    except BaseException as error:
        logging.error(f"Error when becnhmarking the model: {model_name}. Details: {error}")

    return model_perf


def validate_accuracy(model_path, val_loader):
    dataset_size = min(len(val_loader), ILSVRC_VALIDATION_SUBSET_SIZE)
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
        if i > dataset_size - 1:
            break
        image_copies = copy.deepcopy(images.numpy())
        infer_queue.start_async(image_copies, userdata=i)
        references[i] = target

    infer_queue.wait_all()
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)

    return accuracy_score(predictions, references)


def benchmark_models(paths, names):
    performance_list = []

    for model_path, model_name in zip(paths, names):
        if model_path is not None and model_name is not None:
            performance = benchmark_performance(model_path, model_name)
            performance_list.append(performance)
        else:
            performance_list.append("-")
    
    return performance_list


@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")


@pytest.fixture(scope="session")
def output(pytestconfig):
    return pytestconfig.getoption("output")


@pytest.fixture(scope="session")
def subset(pytestconfig):
    return pytestconfig.getoption("subset")


@pytest.fixture(scope="session")
def result(pytestconfig):
    return pytestconfig.test_results 


@pytest.mark.parametrize("model_args", get_validation_scope())
def test_ptq_timm(data, output, subset, result, model_args): # pylint: disable=W0703
    ILSVRC_VALIDATION_SUBSET_SIZE = subset
    BACKEND_ORDER = {
        "torch": 0,
        "onnx": 1,
        "openvino": 2
    }

    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )  # W/A to avoid RuntimeError
    logging.getLogger().setLevel(logging.INFO)

    model_name = model_args["name"]
    quantization_params = model_args["quantization_params"]

    logging.info(f"Quantizing model: '{model_name}', with parameters: "
                 f"{quantization_params}")

    try:
        output_folder = Path(output)
        output_folder.mkdir(parents=True, exist_ok=True)

        model = create_timm_model(model_name)
        model.eval().cpu()
        transform = get_model_transform(model)

        batch_one_dataloader = get_torch_dataloader(data, transform, batch_size=1)

        orig_model_path, orig_acc, _ = QuantizerFactory.validate(
                                                "torch", model, batch_one_dataloader,
                                                model_name, output_folder, False)

        # keep paths and names for performance benchmarking
        ov_model_paths = ["-"] * (len(BACKEND_ORDER)+1) # +1 for original model characteristics
        model_names = ["-"] * (len(BACKEND_ORDER)+1)
        accuracy_results = ["-"] * (len(BACKEND_ORDER)+1)

        ov_model_paths[0] = orig_model_path
        model_names[0] = model_name
        accuracy_results[0] = orig_acc

        def get_backend_model(backend):
            if backend == "torch":
                return model
            elif backend == "onnx":
                return onnx.load(output_folder / (model_name + ".onnx"))
            elif backend == "openvino":
                return ov.Core().read_model(output_folder / (model_name + ".xml"))
            return None

        def quantize_and_validate(backend):
            ov_model_path, q_acc, q_model_name = "-", "-", "-"
            try:
                dataloader = get_torch_dataloader(data, transform, batch_size=1)
                model = get_backend_model(backend)
                q_model = QuantizerFactory.quantize(backend, model, dataloader, quantization_params)
                ov_model_path, q_acc, q_model_name = QuantizerFactory.validate(
                                                    backend, q_model, dataloader,
                                                    model_name, output_folder)
            except Exception as error:
                logging.error(f"Error when handling the model: {model_name} at backend: {backend}."
                            " Details: {error}")
            return backend, ov_model_path, q_acc, q_model_name

        try:
            with Pool() as pool:
                q_results = pool.map(quantize_and_validate, QuantizerFactory.registry.keys())

            ov_model_paths[BACKEND_ORDER[q_results[0]]] = q_results[1]
            model_names[BACKEND_ORDER[q_results[0]]] = q_results[2]
            accuracy_results[BACKEND_ORDER[q_results[0]]] = q_results[3]

        except Exception as error:
            logging.error(f"Error when running quantization in parallel."
                          f" Details: {error}")
            accuracy_results.append(re.escape(str(error)))
            ov_model_paths.append(None)
            model_names.append(None)

        # benchmark sequentially
        perf_results = benchmark_models(ov_model_paths, model_names)
        result.append([model_name] + 
            accuracy_results + 
            perf_results
        )
        logging.info(f"Quantization results: {result[-1]}")

    except Exception as error:
        result.append([model_name, error, "-", "-", "-", "-", "-", "-", "-"])
        logging.error(
            f"Error when running test for model: {model_name}. Error: {error}"
        )
        assert False
