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

import json
import os
import tempfile
from functools import partial
from pathlib import Path

import pytest
import tensorflow as tf
from tensorflow.python.eager import context

import nncf
from examples.common.sample_config import EVAL_ONLY_ERROR_TEXT
from examples.tensorflow.classification import main as cls_main
from examples.tensorflow.common.model_loader import AVAILABLE_MODELS
from examples.tensorflow.common.prepare_checkpoint import main as prepare_checkpoint_main
from examples.tensorflow.object_detection import main as od_main
from examples.tensorflow.segmentation import evaluation as seg_eval
from examples.tensorflow.segmentation import train as seg_train
from tests.cross_fw.shared.config_factory import ConfigFactory
from tests.cross_fw.shared.helpers import remove_line_breaks
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.cross_fw.shared.paths import get_accuracy_aware_checkpoint_dir_path
from tests.tensorflow.helpers import get_cifar10_dataset_builders
from tests.tensorflow.helpers import get_coco_dataset_builders
from tests.tensorflow.test_models import SequentialModel
from tests.tensorflow.test_models import SequentialModelNoInput

AVAILABLE_MODELS.update({"SequentialModel": SequentialModel, "SequentialModelNoInput": SequentialModelNoInput})


@pytest.fixture(autouse=True)
def run_around_tests():
    original_cls_main_get_dataset_builders = cls_main.get_dataset_builders
    original_od_main_get_dataset_builders = od_main.get_dataset_builders
    original_seg_train_get_dataset_builders = seg_train.get_dataset_builders
    original_seg_eval_get_dataset_builders = seg_eval.get_dataset_builders
    cls_main.get_dataset_builders = get_cifar10_dataset_builders
    od_main.get_dataset_builders = partial(get_coco_dataset_builders, train=True, validation=True)
    seg_train.get_dataset_builders = partial(get_coco_dataset_builders, train=True, calibration=True)
    seg_eval.get_dataset_builders = partial(get_coco_dataset_builders, validation=True, calibration=True)
    yield
    cls_main.get_dataset_builders = original_cls_main_get_dataset_builders
    od_main.get_dataset_builders = original_od_main_get_dataset_builders
    seg_train.get_dataset_builders = original_seg_train_get_dataset_builders
    seg_eval.get_dataset_builders = original_seg_eval_get_dataset_builders


def convert_to_argv(args):
    return " ".join(key if val is None else "{} {}".format(key, val) for key, val in args.items()).split()


SAMPLE_TYPES = [
    "classification",
    "object_detection",
    "segmentation",
]

SAMPLES = {
    "classification": {"train-test-export": cls_main.main},
    "object_detection": {"train-test-export": od_main.main},
    "segmentation": {"train": seg_train.main, "test-export": seg_eval.main},
}

DATASETS = {
    "classification": [("cifar10", "tfrecords"), ("cifar10", "tfrecords"), ("cifar10", "tfrecords")],
    "object_detection": [("coco2017", "tfrecords")],
    "segmentation": [("coco2017", "tfrecords")],
}

TEST_CONFIG_ROOT = TEST_ROOT.joinpath("tensorflow", "data", "configs")
CONFIGS = {
    "classification": [
        TEST_CONFIG_ROOT.joinpath("resnet50_cifar10_magnitude_sparsity_int8.json"),
        TEST_CONFIG_ROOT.joinpath("sequential_model_cifar10_magnitude_sparsity_int8.json"),
        TEST_CONFIG_ROOT.joinpath("sequential_model_no_input_cifar10_magnitude_sparsity_int8.json"),
    ],
    "object_detection": [
        TEST_CONFIG_ROOT.joinpath("retinanet_coco2017_magnitude_sparsity_int8.json"),
    ],
    "segmentation": [
        TEST_CONFIG_ROOT.joinpath("mask_rcnn_coco2017_magnitude_sparsity_int8.json"),
    ],
}

BATCH_SIZE_PER_GPU = {
    "classification": [1, 1, 1],
    "object_detection": [1],
    "segmentation": [1],
}


def get_global_batch_size():
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    coeff = num_gpus if num_gpus else 1
    global_batch_size = {}
    for sample_type, batch_sizes in BATCH_SIZE_PER_GPU.items():
        global_batch_size[sample_type] = [coeff * bs for bs in batch_sizes]
    return global_batch_size


GLOBAL_BATCH_SIZE = get_global_batch_size()

DATASET_PATHS = {
    "classification": {
        x: lambda dataset_root, dataset_name=x: (
            os.path.join(dataset_root, dataset_name)
            if dataset_root
            else os.path.join(tempfile.gettempdir(), dataset_name)
        )
        for x, _ in DATASETS["classification"]
    },
    "object_detection": {
        "coco2017": lambda dataset_root: TEST_ROOT.joinpath("tensorflow", "data", "mock_datasets", "coco2017")
    },
    "segmentation": {
        "coco2017": lambda dataset_root: TEST_ROOT.joinpath("tensorflow", "data", "mock_datasets", "coco2017")
    },
}

DATASET_PATHS["classification"]["cifar10"] = lambda dataset_root: TEST_ROOT.joinpath(
    "tensorflow", "data", "mock_datasets", "cifar10"
)


def get_sample_fn(sample_type, modes):
    variants = []
    for key in SAMPLES[sample_type]:
        supported_modes = set(key.split("-"))
        if set(modes).issubset(supported_modes):
            variants.append(key)

    if len(variants) != 1:
        raise Exception("Can not choose a function for given arguments")

    return SAMPLES[sample_type][variants[0]]


def generate_config_params():
    config_params = []
    for sample_id, sample_type in enumerate(SAMPLE_TYPES):
        config_paths, batch_sizes = CONFIGS[sample_type], GLOBAL_BATCH_SIZE[sample_type]
        dataset_names, dataset_types = zip(*DATASETS[sample_type])

        for params_id, params in enumerate(zip(config_paths, dataset_names, dataset_types, batch_sizes)):
            config_params.append((sample_type, *params, "{}_{}".format(sample_id, params_id)))
    return config_params


def generate_id(value):
    sample_type, config_path, dataset_name, dataset_type, batch_size, _ = value
    filename = config_path.name
    return "-".join([sample_type, filename, dataset_name, dataset_type, str(batch_size)])


CONFIG_PARAMS = generate_config_params()


@pytest.fixture(params=CONFIG_PARAMS, ids=generate_id)
def _config(request, dataset_dir):
    sample_type, config_path, dataset_name, dataset_type, batch_size, tid = request.param
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)

    with config_path.open() as f:
        jconfig = json.load(f)

    if "checkpoint_save_dir" in jconfig:
        del jconfig["checkpoint_save_dir"]

    jconfig["dataset"] = dataset_name
    jconfig["dataset_type"] = dataset_type

    return {
        "sample_type": sample_type,
        "nncf_config": jconfig,
        "model_name": jconfig["model"],
        "dataset_path": dataset_path,
        "batch_size": batch_size,
        "tid": tid,
    }


@pytest.fixture(scope="module")
def _case_common_dirs(tmp_path_factory):
    return {
        "checkpoint_save_dir": str(tmp_path_factory.mktemp("models")),
        "optimized_checkpoint_save_dir": str(tmp_path_factory.mktemp("optimized_models")),
    }


def test_model_eval(_config, tmp_path):
    if _config["sample_type"] == "segmentation":
        pytest.skip("ticket #58759")
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    args = {
        "--mode": "test",
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _config["batch_size"],
    }
    main = get_sample_fn(_config["sample_type"], modes=["test"])
    main(convert_to_argv(args))


@pytest.mark.nightly
@pytest.mark.dependency(name="tf_test_model_train")
def test_model_train(_config, tmp_path, _case_common_dirs):
    context._reset_context()
    if _config["sample_type"] == "segmentation":
        pytest.skip("ticket #58759")
    checkpoint_save_dir = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    args = {
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _config["batch_size"],
        "--epochs": 1,
        "--checkpoint-save-dir": checkpoint_save_dir,
    }

    if _config["sample_type"] != "segmentation":
        args["--mode"] = "train"

    main = get_sample_fn(_config["sample_type"], modes=["train"])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.nightly
@pytest.mark.dependency(depends=["tf_test_model_train"])
def test_trained_model_eval(_config, tmp_path, _case_common_dirs):
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    ckpt_path = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])
    args = {
        "--mode": "test",
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _config["batch_size"],
        "--resume": ckpt_path,
    }

    main = get_sample_fn(_config["sample_type"], modes=["test"])
    main(convert_to_argv(args))


@pytest.mark.nightly
@pytest.mark.dependency(depends=["tf_test_model_train"])
def test_resume(_config, tmp_path, _case_common_dirs):
    checkpoint_save_dir = os.path.join(str(tmp_path), "models")
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    ckpt_path = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])

    args = {
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _config["batch_size"],
        "--epochs": 2,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--resume": ckpt_path,
    }

    if _config["sample_type"] != "segmentation":
        args["--mode"] = "train"

    main = get_sample_fn(_config["sample_type"], modes=["train"])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.nightly
@pytest.mark.dependency(depends=["tf_test_model_train"])
def test_trained_model_resume_train_test_export_last_ckpt(_config, tmp_path, _case_common_dirs):
    if _config["sample_type"] == "segmentation":
        pytest.skip()

    checkpoint_save_dir = os.path.join(str(tmp_path), "models")
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    ckpt_path = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])

    export_path = os.path.join(str(tmp_path), "model.pb")
    args = {
        "--mode": "train test export",
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _config["batch_size"],
        "--epochs": 2,
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--resume": ckpt_path,
        "--to-frozen-graph": export_path,
    }

    main = get_sample_fn(_config["sample_type"], modes=["train", "test", "export"])
    main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)
    assert os.path.exists(export_path)


FORMATS = ["frozen-graph", "saved-model", "h5"]


def get_export_model_name(export_format):
    model_name = "model"
    if export_format == "frozen-graph":
        model_name = "model.pb"
    elif export_format == "h5":
        model_name = "model.h5"
    return model_name


@pytest.mark.nightly
@pytest.mark.dependency(depends=["tf_test_model_train"])
@pytest.mark.parametrize("export_format", FORMATS, ids=FORMATS)
def test_export_with_resume(_config, tmp_path, export_format, _case_common_dirs):
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    ckpt_path = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])

    if export_format == "saved-model":
        compression_config = _config["nncf_config"].get("compression", {})
        if isinstance(compression_config, dict):
            compression_config = [compression_config]
        for config in compression_config:
            if config.get("algorithm", "") == "quantization":
                pytest.skip()

    if _config["sample_type"] == "segmentation" and export_format == "h5":
        pytest.skip("The {} sample does not support export to {} format.".format(_config["sample_type"], export_format))

    export_path = os.path.join(str(tmp_path), get_export_model_name(export_format))
    args = {
        "--mode": "export",
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--resume": ckpt_path,
        "--to-{}".format(export_format): export_path,
    }

    main = get_sample_fn(_config["sample_type"], modes=["export"])
    main(convert_to_argv(args))

    model_path = os.path.join(export_path, "saved_model.pb") if export_format == "saved-model" else export_path
    assert os.path.exists(model_path)


PREPARE_CHECKPOINTS_SUPPORTED_SAMPLE_TYPES = ["object_detection", "segmentation"]


@pytest.mark.nightly
@pytest.mark.dependency(name="tf_test_prepare_checkpoint", depends=["tf_test_model_train"])
def test_prepare_checkpoint(_config, tmp_path, _case_common_dirs):
    if _config["sample_type"] not in PREPARE_CHECKPOINTS_SUPPORTED_SAMPLE_TYPES:
        pytest.skip("Unsupported sample type for test_prepare_checkpoints")

    resume_path = os.path.join(_case_common_dirs["checkpoint_save_dir"], _config["tid"])
    checkpoint_save_dir = os.path.join(_case_common_dirs["optimized_checkpoint_save_dir"], _config["tid"])
    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    args = {
        "--model-type": _config["sample_type"],
        "--config": config_factory.serialize(),
        "--checkpoint-save-dir": checkpoint_save_dir,
        "--resume": resume_path,
    }

    prepare_checkpoint_main(convert_to_argv(args))

    assert tf.io.gfile.isdir(checkpoint_save_dir)
    assert tf.train.latest_checkpoint(checkpoint_save_dir)


@pytest.mark.nightly
@pytest.mark.dependency(depends=["tf_test_prepare_checkpoint"])
def test_eval_prepared_checkpoint(_config, tmp_path, _case_common_dirs):
    if _config["sample_type"] not in PREPARE_CHECKPOINTS_SUPPORTED_SAMPLE_TYPES:
        pytest.skip("Unsupported sample type for test_prepare_checkpoints")

    config_factory = ConfigFactory(_config["nncf_config"], tmp_path / "config.json")
    resume_path = os.path.join(_case_common_dirs["optimized_checkpoint_save_dir"], _config["tid"])

    args = {
        "--mode": "test",
        "--data": _config["dataset_path"],
        "--config": config_factory.serialize(),
        "--batch-size": _config["batch_size"],
        "--resume": resume_path,
    }

    main = get_sample_fn(_config["sample_type"], modes=["test"])
    main(convert_to_argv(args))


@pytest.fixture(
    params=[
        TEST_ROOT.joinpath("tensorflow", "data", "configs", "sequential_pruning_accuracy_aware.json"),
        TEST_ROOT.joinpath("tensorflow", "data", "configs", "sequential_int8_accuracy_aware.json"),
    ]
)
def _accuracy_aware_config(request, dataset_dir):
    config_path = request.param
    sample_type = "classification"
    dataset_name, dataset_type = "cifar10", "tfrecords"
    dataset_path = DATASET_PATHS[sample_type][dataset_name](dataset_dir)
    with config_path.open() as f:
        jconfig = json.load(f)

    jconfig["dataset"] = dataset_name
    jconfig["dataset_type"] = dataset_type

    num_gpus = len(tf.config.list_physical_devices("GPU"))
    batch_size = num_gpus if num_gpus else 1

    return {
        "sample_type": sample_type,
        "nncf_config": jconfig,
        "model_name": jconfig["model"],
        "dataset_path": dataset_path,
        "batch_size": batch_size,
    }


@pytest.mark.nightly
@pytest.mark.dependency(name="tf_test_model_train")
def test_model_accuracy_aware_train(_accuracy_aware_config, tmp_path):
    checkpoint_save_dir = Path(tmp_path)
    config_factory = ConfigFactory(_accuracy_aware_config["nncf_config"], tmp_path / "config.json")
    args = {
        "--data": _accuracy_aware_config["dataset_path"],
        "--config": config_factory.serialize(),
        "--log-dir": tmp_path,
        "--batch-size": _accuracy_aware_config["batch_size"],
        "--epochs": 1,
        "--checkpoint-save-dir": tmp_path,
    }

    main = get_sample_fn(_accuracy_aware_config["sample_type"], modes=["train"])
    main(convert_to_argv(args))

    model_dirs = [x for x in checkpoint_save_dir.glob("*/") if x.is_dir()]
    assert len(model_dirs) == 1
    model_specific_run_dir = model_dirs[0]
    aa_checkpoint_path = get_accuracy_aware_checkpoint_dir_path(model_specific_run_dir)
    assert tf.train.latest_checkpoint(str(aa_checkpoint_path))


@pytest.mark.nightly
@pytest.mark.parametrize("sample_type", SAMPLE_TYPES)
def test_eval_only_config_fails_to_train(tmp_path, sample_type):
    config_factory = ConfigFactory(
        {"model": "mock", "input_info": {"sample_size": [1, 1, 1, 1]}, "eval_only": True}, tmp_path / "config.json"
    )
    args = {
        "--config": config_factory.serialize(),
    }

    main = get_sample_fn(sample_type, modes=["train"])
    with pytest.raises(nncf.ValidationError) as e_info:
        main(convert_to_argv(args))
    assert remove_line_breaks(EVAL_ONLY_ERROR_TEXT) in remove_line_breaks(e_info.value.args[0])
