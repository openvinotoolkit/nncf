from bz2 import compress
import os
import re
import logging
from pathlib import Path
import numpy as np

import torch
import nncf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import pytest

import timm
from texttable import Texttable
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import openvino.runtime as ov

NOT_AVAILABLE_MESSAGE = 'N/A'

def get_model_list():
    full_list = timm.list_models()
    #model_list = [m for m in full_list if "mobile" in m] # Shold be revised in the future
    #model_list += [m for m in full_list if "vit" in m]
    #model_list = [m for m in full_list if "levit" in m]
    #model_list = ["adv_inception_v3", "bat_resnext26ts", "beit_base_patch16_224","botnet26t_256", "cait_m36_384", "coat_lite_mini", "convit_tiny", "convmixer_768_32",  "convnext_base","crossvit_9_240", "cspdarknet53", "darknet53", "deit_base_distilled_patch16_224","densenet121", "dla34",  "dm_nfnet_f0",  "dpn68", "eca_botnext26ts_256","ecaresnet26t", "efficientnet_b0","efficientnet_el_pruned","efficientnet_lite0", "efficientnetv2_l", "ese_vovnet19b_dw","fbnetc_100", "gcresnet33ts", "gernet_l", "gernet_m", "gernet_s", "ghostnet_050","gluon_senet154", "gluon_seresnext50_32x4d", "gluon_xception65", "gmixer_12_224",  "gmlp_b16_224","halo2botnet50ts_256", "hardcorenas_a","hrnet_w18", "ig_resnext101_32x8d", "inception_resnet_v2", "inception_v3", "inception_v4", "jx_nest_base","lambda_resnet26rpt_256","lcnet_035","levit_128", "mixer_b16_224","mnasnet_050", "mobilenetv2_035", "mobilenetv2_050", "mobilenetv2_075", "mobilenetv2_100","mobilenetv3_large_075", "mobilenetv3_large_100","nasnetalarge", "nest_base","nf_ecaresnet26", "nf_ecaresnet50", "nf_regnet_b0","nf_seresnet50", "nfnet_f2s","pit_b_distilled_224", "pit_s_224",  "pnasnet5large", "regnetx_002","regnety_002",  "regnetz_b16","repvgg_a2", "repvgg_b2",  "res2net50_14w_8s","resmlp_12_224","resmlp_36_224","resnest14d","resnet18", "resnetblur18","resnetrs50","resnetv2_50d", "resnetv2_50x1_bitm_in21k","resnext26ts","rexnetr_130","sebotnet33ts_256", "sehalonet33ts", "selecsls42","semnasnet_050",  "senet154", "seresnet18", "skresnet18", "spnasnet_100", "ssl_resnet18","swin_base_patch4_window7_224","swsl_resnet18","tresnet_m", "tv_resnet34", "twins_pcpvt_base","vgg11", "visformer_small","vit_base_patch16_224","wide_resnet101_2", "xception","xcit_large_24_p8_224"]
    #model_list = ["mobilenetv2_050","resnet18"]
    model_list = ["mobilenetv2_050"]
    return model_list

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


def export_to_onnx(model, save_path):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(model,
                      x,
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

def benchmark_model(model_path, model_name):
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

    print(f"Model path: {model_path}")
    core = ov.Core()
    ov_model = core.read_model(model_path)
    compiled_model = core.compile_model(ov_model)

    jobs = 4
    infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

    def process_result(request, userdata):
        result = request.get_output_tensor().data
        predicted_label = np.argmax(result, axis=1)

        predictions[userdata] = [predicted_label]

    infer_queue.set_callback(process_result)

    for i, (images, target) in tqdm(enumerate(val_loader)):
        infer_queue.start_async(images.numpy(), userdata=i)
        
        references[i] = target

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)

    return accuracy_score(predictions, references)

@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")

OUTPUT_PATH="./tmp"
@pytest.mark.parametrize("model_name", 
            get_model_list())
def test_torch_quantization(data, model_name):
    torch.multiprocessing.set_sharing_strategy('file_system') # W/A to avoid RuntimeError

    model = create_timm_model(model_name)
    model.eval().cpu()
    transform = get_model_transform(model)
    val_dataloader = get_torch_dataloader(data, transform, batch_size=128)

    # quantize model with NNCF PTQ API
    def transform_fn(data_item):
        images, _ = data_item
        return images
    calibration_dataset = nncf.Dataset(val_dataloader, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Dump FP32 model
    onnx_path = Path(OUTPUT_PATH) / (model_name + ".onnx")
    export_to_onnx(model, onnx_path)
    ov_path = Path(OUTPUT_PATH) / (model_name + ".xml")
    export_to_ir(onnx_path, OUTPUT_PATH, model_name)

     # Dump quantized model
    q_model_name = model_name + "_int8"
    q_onnx_path = Path(OUTPUT_PATH) / (q_model_name + ".onnx")
    export_to_onnx(quantized_model, q_onnx_path)
    q_ov_path = Path(OUTPUT_PATH) / (q_model_name + ".xml")
    export_to_ir(q_onnx_path, OUTPUT_PATH, q_model_name)

    # Benchmark FP32 performance
    q_perf = benchmark_model(ov_path, model_name)
    print(f"Performance of FP32 model: {q_perf}")

    # Benchmark INT8 performance
    q_perf = benchmark_model(q_ov_path, q_model_name)
    print(f"Performance of INT8 model: {q_perf}")

    ov_dataloader = get_torch_dataloader(data, transform, batch_size=1)
    # Validate FP32 accuracy
    accuracy = validate_accuracy(ov_path, ov_dataloader)
    print(f"Accuracy of FP32 model: {accuracy}")

    # Validate quantized accuracy
    q_accuracy = validate_accuracy(q_ov_path, ov_dataloader)
    print(f"Accuracy of INT8 model: {q_accuracy}")

    assert quantized_model


