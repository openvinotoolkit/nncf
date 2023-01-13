"""
 Copyright (c) 2019-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import json
import os
import shlex
import tempfile
from contextlib import contextmanager
from contextlib import nullcontext

import pytest
import torch
import torchvision
from pkg_resources import parse_version
from pytest_dependency import depends

from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.optimizer import get_default_weight_decay
from examples.torch.common.sample_config import SampleConfig
from examples.torch.common.utils import get_name
from examples.torch.common.utils import is_staged_quantization
from nncf.api.compression import CompressionStage
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.common.compression import BaseControllerStateNames
from nncf.common.hardware.config import HWConfigType
from nncf.config import NNCFConfig
from tests.shared.config_factory import ConfigFactory
from tests.shared.paths import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.sample_test_validator import create_command_line

NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1


SAMPLE_TYPES = ["classification", "semantic_segmentation", "object_detection"]

DATASETS = {
    "classification": ["mock_32x32", "mock_299x299", "mock_32x32", "mock_32x32"],
    "semantic_segmentation": ["camvid", "camvid"],
    "object_detection": ["voc"],
}

CONFIGS = {
    "classification": [TEST_ROOT / "torch" / "data" / "configs" / "squeezenet1_1_cifar10_rb_sparsity_int8.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "inception_v3_mock_dataset.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_cifar100_bin_xnor.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_cifar10_staged_quant.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_pruning_magnitude.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_pruning_learned_ranking.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_pruning_accuracy_aware.json",
                       TEST_ROOT / "torch" / "data" / "configs" / "resnet18_int8_accuracy_aware.json"],
    "semantic_segmentation": [TEST_ROOT / "torch" / "data" / "configs" / "unet_camvid_int8.json",
                              TEST_ROOT / "torch" / "data" / "configs" / "unet_camvid_rb_sparsity.json"],
    "object_detection": [TEST_ROOT / "torch" / "data" / "configs" / "ssd300_vgg_voc_int8.json",
                         TEST_ROOT / "torch" / "data" / "configs" / "ssd300_vgg_voc_int8_accuracy_aware.json"]
}

BATCHSIZE_PER_GPU = {
    "classification": [256, 32, 256, 256, 128],
    "semantic_segmentation": [2, 2],
    "object_detection": [128],
}

DATASET_PATHS = {
    "classification": {
        x: lambda dataset_root: dataset_root if dataset_root else os.path.join(
            tempfile.gettempdir(), x) for x in DATASETS["classification"]
    },
    "semantic_segmentation": {
        DATASETS["semantic_segmentation"][0]: lambda dataset_root: TEST_ROOT / "torch" / "data" / "mock_datasets"
                                                                   / "camvid",
        DATASETS["semantic_segmentation"][0]: lambda dataset_root:  TEST_ROOT / "torch" / "data" / "mock_datasets"
                                                                   / "camvid",
    },
    "object_detection": {
        DATASETS["object_detection"][0]: lambda dataset_root: TEST_ROOT / "torch" / "data" / "mock_datasets" / "voc"
    },
}

CONFIG_PARAMS = []
for sample_type_ in SAMPLE_TYPES:
    for tpl in list(zip(CONFIGS[sample_type_], DATASETS[sample_type_], BATCHSIZE_PER_GPU[sample_type_])):
        CONFIG_PARAMS.append((sample_type_,) + tpl)


def update_compression_algo_dict_with_reduced_bn_adapt_params(algo_dict):
    if algo_dict["algorithm"] == "rb_sparsity":
        return
    if 'initializer' not in algo_dict:
        algo_dict['initializer'] = {'batchnorm_adaptation': {'num_bn_adaptation_samples': 5}}
    else:
        algo_dict['initializer'].update({'batchnorm_adaptation': {'num_bn_adaptation_samples': 5}})


def update_compression_algo_dict_with_legr_save_load_params(nncf_config, tmp_path, save=True):
    if 'compression' not in nncf_config:
        return nncf_config
    if isinstance(nncf_config["compression"], list):
        algos_list = nncf_config["compression"]
    else:
        algos_list = [nncf_config["compression"]]

    for algo_dict in algos_list:
        if algo_dict["algorithm"] != "filter_pruning":
            continue

        if "interlayer_ranking_type" in algo_dict['params'] and algo_dict['params'][
            "interlayer_ranking_type"] == 'learned_ranking':
            if save:
                algo_dict['params']['save_ranking_coeffs_path'] = os.path.join(tmp_path, 'ranking_coeffs.json')
            else:
                algo_dict['params']['load_ranking_coeffs_path'] = os.path.join(tmp_path, 'ranking_coeffs.json')
    return nncf_config


def _get_test_case_id(p) -> str:
    return "-".join([p[0], p[1].name, p[2], str(p[3])])


@pytest.fixture(params=CONFIG_PARAMS, name='config',
                ids=[_get_test_case_id(p) for p in CONFIG_PARAMS])
def fixture_config(request, dataset_dir):
    sample_type, config_path, dataset_name, batch_size = request.param
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)

    with config_path.open() as f:
        jconfig = json.load(f)

    if "checkpoint_save_dir" in jconfig.keys():
        del jconfig["checkpoint_save_dir"]

    # Use a reduced number of BN adaptation samples for speed
    if "compression" in jconfig:
        if isinstance(jconfig["compression"], list):
            algos_list = jconfig["compression"]
            for algo_dict in algos_list:
                update_compression_algo_dict_with_reduced_bn_adapt_params(algo_dict)
        else:
            algo_dict = jconfig["compression"]
            update_compression_algo_dict_with_reduced_bn_adapt_params(algo_dict)
    jconfig["dataset"] = dataset_name

    return {
        "sample_type": sample_type,
        'nncf_config': jconfig,
        "model_name": jconfig["model"],
        "dataset_path": dataset_path,
        "batch_size": batch_size,
        "test_case_id": _get_test_case_id(request.param)
    }


@pytest.fixture(scope="module", name='case_common_dirs')
def fixture_case_common_dirs(tmp_path_factory):
    return {
        "checkpoint_save_dir": str(tmp_path_factory.mktemp("models")),
        "save_coeffs_path": str(tmp_path_factory.mktemp("ranking_coeffs")),
    }


@pytest.mark.parametrize(" multiprocessing_distributed",
                         (True, False),
                         ids=['distributed', 'dataparallel'])
def test_pretrained_model_eval(config, tmp_path, multiprocessing_distributed, case_common_dirs):
    if parse_version(torchvision.__version__) < parse_version("0.13") and 'voc' in str(config["dataset_path"]):
        pytest.skip(f'Test calls sample that uses `datasets.VOCDetection.parse_voc_xml` function from latest '
                    f'torchvision.\nThe signature of the function is not compatible with the corresponding signature '
                    f'from the current torchvision version : {torchvision.__version__}')
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    config_factory.config = update_compression_algo_dict_with_legr_save_load_params(config_factory.config,
                                                                                    case_common_dirs[
                                                                                        'save_coeffs_path'])
    args = {
        "--mode": "test",
        "--data": config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--dist-url": "tcp://127.0.0.1:8987"
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True
    elif multiprocessing_distributed:
        args["--multiprocessing-distributed"] = True

    runner = Command(create_command_line(args, config["sample_type"]))
    runner.run()


@pytest.mark.dependency()
@pytest.mark.parametrize(
    "multiprocessing_distributed", [True, False],
    ids=['distributed', 'dataparallel'])
def test_pretrained_model_train(config, tmp_path, multiprocessing_distributed, case_common_dirs):
    checkpoint_save_dir = os.path.join(case_common_dirs["checkpoint_save_dir"],
                                       "distributed" if multiprocessing_distributed else "data_parallel")
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    config_factory.config = update_compression_algo_dict_with_legr_save_load_params(config_factory.config,
                                                                                    case_common_dirs[
                                                                                        'save_coeffs_path'])

    args = {
        "--mode": "train",
        "--data": config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--epochs": 2,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--dist-url": "tcp://127.0.0.1:8989"
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True
    elif multiprocessing_distributed:
        args["--multiprocessing-distributed"] = True
    elif config['nncf_config']["model"] == "inception_v3":
        pytest.skip("InceptionV3 may not be trained in DataParallel "
                    "because it outputs namedtuple, which DP seems to be unable "
                    "to support even still.")

    runner = Command(create_command_line(args, config["sample_type"]))
    runner.run()
    last_checkpoint_path = os.path.join(checkpoint_save_dir, get_name(config_factory.config) + "_last.pth")
    assert os.path.exists(last_checkpoint_path)
    if 'compression' in config['nncf_config']:
        allowed_compression_stages = (CompressionStage.FULLY_COMPRESSED, CompressionStage.PARTIALLY_COMPRESSED)
    else:
        allowed_compression_stages = (CompressionStage.UNCOMPRESSED,)
    compression_stage = extract_compression_stage_from_checkpoint(last_checkpoint_path)
    assert compression_stage in allowed_compression_stages


def depends_on_pretrained_train(request, test_case_id: str, current_multiprocessing_distributed: bool):
    full_test_case_id = test_case_id + ('-distributed' if current_multiprocessing_distributed else '-dataparallel')
    primary_test_case_name = f'test_pretrained_model_train[{full_test_case_id}]'
    depends(request, [primary_test_case_name])


@pytest.mark.dependency()
@pytest.mark.parametrize(
    "multiprocessing_distributed", [True, False],
    ids=['distributed', 'dataparallel'])
def test_trained_model_eval(request, config, tmp_path, multiprocessing_distributed, case_common_dirs):
    if parse_version(torchvision.__version__) < parse_version("0.13") and 'voc' in str(config["dataset_path"]):
        pytest.skip(f'Test calls sample that uses `datasets.VOCDetection.parse_voc_xml` function from latest '
                    f'torchvision.\nThe signature of the function is not compatible with the corresponding signature '
                    f'from the current torchvision version : {torchvision.__version__}')
    depends_on_pretrained_train(request, config["test_case_id"], multiprocessing_distributed)
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    config_factory.config = update_compression_algo_dict_with_legr_save_load_params(config_factory.config,
                                                                                    case_common_dirs[
                                                                                        'save_coeffs_path'])

    ckpt_path = os.path.join(case_common_dirs["checkpoint_save_dir"],
                             "distributed" if multiprocessing_distributed else "data_parallel",
                             get_name(config_factory.config) + "_last.pth")
    args = {
        "--mode": "test",
        "--data": config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--weights": ckpt_path,
        "--dist-url": "tcp://127.0.0.1:8987"
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True
    elif multiprocessing_distributed:
        args["--multiprocessing-distributed"] = True

    runner = Command(create_command_line(args, config["sample_type"]))
    runner.run()


def get_resuming_checkpoint_path(config_factory, multiprocessing_distributed, checkpoint_save_dir):
    return os.path.join(checkpoint_save_dir,
                        "distributed" if multiprocessing_distributed else "data_parallel",
                        get_name(config_factory.config) + "_last.pth")


@pytest.mark.dependency()
@pytest.mark.parametrize(
    "multiprocessing_distributed", [True, False],
    ids=['distributed', 'dataparallel'])
def test_resume(request, config, tmp_path, multiprocessing_distributed, case_common_dirs):
    depends_on_pretrained_train(request, config["test_case_id"], multiprocessing_distributed)
    checkpoint_save_dir = os.path.join(str(tmp_path), "models")
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    config_factory.config = update_compression_algo_dict_with_legr_save_load_params(config_factory.config,
                                                                                    case_common_dirs[
                                                                                        'save_coeffs_path'], False)

    ckpt_path = get_resuming_checkpoint_path(config_factory, multiprocessing_distributed,
                                             case_common_dirs["checkpoint_save_dir"])
    if "max_iter" in config_factory.config:
        config_factory.config["max_iter"] += 2
    args = {
        "--mode": "train",
        "--data": config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--epochs": 3,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--resume": ckpt_path,
        "--dist-url": "tcp://127.0.0.1:8986"
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True
    elif multiprocessing_distributed:
        args["--multiprocessing-distributed"] = True

    runner = Command(create_command_line(args, config["sample_type"]))
    runner.run()
    last_checkpoint_path = os.path.join(checkpoint_save_dir, get_name(config_factory.config) + "_last.pth")
    assert os.path.exists(last_checkpoint_path)
    if 'compression' in config['nncf_config']:
        allowed_compression_stages = (CompressionStage.FULLY_COMPRESSED, CompressionStage.PARTIALLY_COMPRESSED)
    else:
        allowed_compression_stages = (CompressionStage.UNCOMPRESSED,)
    compression_stage = extract_compression_stage_from_checkpoint(last_checkpoint_path)
    assert compression_stage in allowed_compression_stages


def extract_compression_stage_from_checkpoint(last_checkpoint_path):
    compression_state = torch.load(last_checkpoint_path)[COMPRESSION_STATE_ATTR]
    ctrl_state = compression_state[BaseController.CONTROLLER_STATE]
    compression_stage = next(iter(ctrl_state.values()))[BaseControllerStateNames.COMPRESSION_STAGE]
    return compression_stage


@pytest.mark.dependency()
@pytest.mark.parametrize(
    "multiprocessing_distributed", [True, False],
    ids=['distributed', 'dataparallel'])
def test_export_with_resume(request, config, tmp_path, multiprocessing_distributed, case_common_dirs):
    depends_on_pretrained_train(request, config["test_case_id"], multiprocessing_distributed)
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    config_factory.config = update_compression_algo_dict_with_legr_save_load_params(config_factory.config,
                                                                                    case_common_dirs[
                                                                                        'save_coeffs_path'], False)

    ckpt_path = get_resuming_checkpoint_path(config_factory, multiprocessing_distributed,
                                             case_common_dirs["checkpoint_save_dir"])

    onnx_path = os.path.join(str(tmp_path), "model.onnx")
    args = {
        "--mode": "export",
        "--config": config_factory.serialize(),
        "--resume": ckpt_path,
        "--to-onnx": onnx_path
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True

    runner = Command(create_command_line(args, config["sample_type"]))
    runner.run()
    assert os.path.exists(onnx_path)


def test_export_with_pretrained(tmp_path):
    config = SampleConfig()
    config.update({
        "model": "resnet18",
        "dataset": "imagenet",
        "input_info": {
            "sample_size": [2, 3, 299, 299]
        },
        "num_classes": 1000,
        "compression": {"algorithm": "magnitude_sparsity"}
    })
    config_factory = ConfigFactory(config, tmp_path / 'config.json')

    onnx_path = os.path.join(str(tmp_path), "model.onnx")
    args = {
        "--mode": "export",
        "--config": config_factory.serialize(),
        "--pretrained": '',
        "--to-onnx": onnx_path
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True

    runner = Command(create_command_line(args, "classification"))
    runner.run()
    assert os.path.exists(onnx_path)


@pytest.mark.parametrize(('algo', 'ref_weight_decay'),
                         (('rb_sparsity', 0),
                          ('const_sparsity', 1e-4),
                          ('magnitude_sparsity', 1e-4),
                          ('quantization', 1e-4)))
def test_get_default_weight_decay(algo, ref_weight_decay):
    config = NNCFConfig()
    config.update({"compression": {"algorithm": algo}})
    assert ref_weight_decay == get_default_weight_decay(config)


@contextmanager
def set_num_threads_locally(n=1):
    old_n = torch.get_num_threads()
    try:
        torch.set_num_threads(n)
        yield
    finally:
        torch.set_num_threads(old_n)


def test_cpu_only_mode_produces_cpu_only_model(config, tmp_path, mocker):
    config_factory = ConfigFactory(config['nncf_config'], tmp_path / 'config.json')
    args = {
        "--data": config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--epochs": 1,
        "--cpu-only": True
    }

    # to prevent starting a not closed mlflow session due to memory leak of config and SafeMLFLow happens with a
    # mocked train function
    mocker.patch("examples.torch.common.utils.SafeMLFLow")
    arg_list = [key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items()]
    command_line = " ".join(arg_list)
    if config["sample_type"] == "classification":
        import examples.torch.classification.main as sample
        if is_staged_quantization(config['nncf_config']):
            mocker.patch("examples.torch.classification.staged_quantization_worker.train_epoch_staged")
            mocker.patch("examples.torch.classification.staged_quantization_worker.validate")
            import examples.torch.classification.staged_quantization_worker as staged_worker
            staged_worker.validate.return_value = (0, 0, 0)
        else:
            mocker.patch("examples.torch.classification.main.train_epoch")
            mocker.patch("examples.torch.classification.main.validate")
            sample.validate.return_value = (0, 0, 0)
    elif config["sample_type"] == "semantic_segmentation":
        import examples.torch.semantic_segmentation.main as sample
        import examples.torch.semantic_segmentation.train
        mocker.spy(examples.torch.semantic_segmentation.train.Train, "__init__")
    elif config["sample_type"] == "object_detection":
        import examples.torch.object_detection.main as sample
        mocker.spy(sample, "train")

    # Set number of threads = 1 to avoid hang for UNet (ticket 100106).
    # Potentially it might happen when OpenMP is used before fork.
    # The relevant thread: https://github.com/pytorch/pytorch/issues/91547
    with set_num_threads_locally(1) if config["sample_type"] == "semantic_segmentation" else nullcontext():
        sample.main(shlex.split(command_line))

    # pylint: disable=no-member
    if config["sample_type"] == "classification":
        if is_staged_quantization(config['nncf_config']):
            import examples.torch.classification.staged_quantization_worker as staged_worker
            model_to_be_trained = staged_worker.train_epoch_staged.call_args[0][2]  # model
        else:
            model_to_be_trained = sample.train_epoch.call_args[0][1]  # model
    elif config["sample_type"] == "semantic_segmentation":
        model_to_be_trained = examples.torch.semantic_segmentation.train.Train.__init__.call_args[0][1]  # model
    elif config["sample_type"] == "object_detection":
        model_to_be_trained = sample.train.call_args[0][0]  # net

    for p in model_to_be_trained.parameters():
        assert not p.is_cuda


@pytest.mark.parametrize('target_device', [x.value for x in HWConfigType])
def test_sample_propagates_target_device_cl_param_to_nncf_config(mocker, tmp_path, target_device):
    config_dict = {
        "input_info":
            {
                "sample_size": [1, 1, 32, 32],
            },
        "compression": {
            "algorithm": "quantization"
        },
    }
    config_factory = ConfigFactory(config_dict, tmp_path / 'config.json')
    args = {
        "--data": str(tmp_path),
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": 1,
        "--target-device": target_device,
    }
    if not torch.cuda.is_available():
        args["--cpu-only"] = True

    arg_list = [key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items()]
    command_line = " ".join(arg_list)
    import examples.torch.classification.main as sample
    start_worker_mock = mocker.patch("examples.torch.classification.main.start_worker")
    sample.main(shlex.split(command_line))

    config = start_worker_mock.call_args[0][1].nncf_config
    assert config["target_device"] == target_device


@pytest.fixture(name='accuracy_aware_config',
                params=[TEST_ROOT / "torch" / "data" / "configs" / "resnet18_pruning_accuracy_aware.json",
                        TEST_ROOT / "torch" / "data" / "configs" / "resnet18_int8_accuracy_aware.json"])
def fixture_accuracy_aware_config(request):
    config_path = request.param
    with config_path.open() as f:
        jconfig = json.load(f)

    dataset_name = 'mock_32x32'
    dataset_path = os.path.join('/tmp', 'mock_32x32')
    sample_type = 'classification'

    jconfig["dataset"] = dataset_name

    return {
        "sample_type": sample_type,
        'nncf_config': jconfig,
        "model_name": jconfig["model"],
        "dataset_path": dataset_path,
        "batch_size": 12,
    }


@pytest.mark.dependency()
@pytest.mark.parametrize(
    "multiprocessing_distributed", [True, False],
    ids=['distributed', 'dataparallel'])
def test_accuracy_aware_training_pipeline(accuracy_aware_config, tmp_path, multiprocessing_distributed):
    config_factory = ConfigFactory(accuracy_aware_config['nncf_config'], tmp_path / 'config.json')
    log_dir = tmp_path / 'accuracy_aware'
    log_dir = log_dir / 'distributed' if multiprocessing_distributed else log_dir / 'dataparallel'

    args = {
        "--mode": "train",
        "--data": accuracy_aware_config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": log_dir,
        "--batch-size": accuracy_aware_config["batch_size"] * NUM_DEVICES,
        "--workers": 0,  # Workaround for the PyTorch MultiProcessingDataLoader issue
        "--epochs": 2,
        "--dist-url": "tcp://127.0.0.1:8989"
    }

    if not torch.cuda.is_available():
        args["--cpu-only"] = True
    elif multiprocessing_distributed:
        args["--multiprocessing-distributed"] = True

    runner = Command(create_command_line(args, accuracy_aware_config["sample_type"]))
    runner.run()

    from glob import glob
    time_dir_1 = glob(os.path.join(log_dir, get_name(config_factory.config), '*/'))[0].split('/')[-2]
    time_dir_2 = glob(os.path.join(log_dir, get_name(config_factory.config), time_dir_1,
                                   'accuracy_aware_training', '*/'))[0].split('/')[-2]
    last_checkpoint_path = os.path.join(log_dir, get_name(config_factory.config), time_dir_1,
                                        'accuracy_aware_training',
                                        time_dir_2, 'acc_aware_checkpoint_last.pth')

    assert os.path.exists(last_checkpoint_path)
    if 'compression' in accuracy_aware_config['nncf_config']:
        allowed_compression_stages = (CompressionStage.FULLY_COMPRESSED, CompressionStage.PARTIALLY_COMPRESSED)
    else:
        allowed_compression_stages = (CompressionStage.UNCOMPRESSED,)
    compression_stage = extract_compression_stage_from_checkpoint(last_checkpoint_path)
    assert compression_stage in allowed_compression_stages
