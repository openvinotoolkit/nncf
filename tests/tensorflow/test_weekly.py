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

import pytest
from pytest import approx

from tests.cross_fw.shared.paths import PROJECT_ROOT
from tests.tensorflow.test_sanity_sample import get_sample_fn

EXAMPLES_DIR = PROJECT_ROOT.joinpath("examples", "tensorflow")

# sample
# ├── dataset
# │   ├── path
# │   ├── batch
# │   ├── configs
# │   │     ├─── config_filename
# │   │     │       ├── expected_accuracy
# │   │     │       ├── absolute_tolerance_train
# │   │     │       ├── absolute_tolerance_test
# │   │     │       ├── execution_arg
# │   │     │       ├── weights
GLOBAL_CONFIG = {
    "classification": {
        "imagenet2012": {
            "configs": {
                "quantization/inception_v3_imagenet_int8.json": {
                    "expected_accuracy": 78.35,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json": {
                    "expected_accuracy": 77.58,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity/inception_v3_imagenet_magnitude_sparsity.json": {
                    "expected_accuracy": 77.87,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "quantization/mobilenet_v2_imagenet_int8.json": {
                    "expected_accuracy": 71.66,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json": {
                    "expected_accuracy": 71.00,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity/mobilenet_v2_imagenet_rb_sparsity.json": {
                    "expected_accuracy": 71.34,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json": {
                    "expected_accuracy": 71.83,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "quantization/mobilenet_v3_small_imagenet_int8.json": {
                    "expected_accuracy": 67.70,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity_quantization/mobilenet_v3_small_imagenet_rb_sparsity_int8.json": {
                    "expected_accuracy": 67.70,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "quantization/mobilenet_v3_large_imagenet_int8.json": {
                    "expected_accuracy": 75.00,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity_quantization/mobilenet_v3_large_imagenet_rb_sparsity_int8.json": {
                    "expected_accuracy": 75.15,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "quantization/resnet50_imagenet_int8.json": {
                    "expected_accuracy": 75.00,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "quantization/resnet50_int8_accuracy_aware.json": {
                    "expected_accuracy": 74.88,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json": {
                    "expected_accuracy": 74.30,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "sparsity/resnet50_imagenet_rb_sparsity.json": {
                    "expected_accuracy": 74.36,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "pruning/resnet50_imagenet_pruning_geometric_median.json": {
                    "expected_accuracy": 74.98,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
                "pruning_quantization/resnet50_imagenet_pruning_geometric_median_int8.json": {
                    "expected_accuracy": 75.08,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                },
            }
        }
    },
    "object_detection": {
        "coco2017": {
            "configs": {
                "quantization/retinanet_coco_int8.json": {
                    "expected_accuracy": 33.22,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "retinanet_coco/retinanet_coco.h5",
                },
                "sparsity/retinanet_coco_magnitude_sparsity.json": {
                    "expected_accuracy": 33.13,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "retinanet_coco/retinanet_coco.h5",
                },
                "pruning/retinanet_coco_pruning_geometric_median.json": {
                    "expected_accuracy": 32.70,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "retinanet_coco/retinanet_coco.h5",
                },
                "pruning_quantization/retinanet_coco_pruning_geometric_median_int8.json": {
                    "expected_accuracy": 32.53,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "retinanet_coco/retinanet_coco.h5",
                },
                "quantization/yolo_v4_coco_int8.json": {
                    "expected_accuracy": 46.15,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "yolo_v4_coco/yolo_v4_coco.h5",
                },
                "sparsity/yolo_v4_coco_magnitude_sparsity.json": {
                    "expected_accuracy": 46.54,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "yolo_v4_coco/yolo_v4_coco.h5",
                },
            }
        }
    },
    "segmentation": {
        "coco2017": {
            "configs": {
                "quantization/mask_rcnn_coco_int8.json": {
                    "expected_accuracy": 37.12,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "mask_rcnn_coco",
                },
                "sparsity/mask_rcnn_coco_magnitude_sparsity.json": {
                    "expected_accuracy": 36.93,
                    "absolute_tolerance_train": 0.5,
                    "absolute_tolerance_test": 0.5,
                    "weights": "mask_rcnn_coco",
                },
            }
        }
    },
}


def convert_to_argv(args):
    return " ".join(f"--{key}" if val is None else f"--{key} {val}" for key, val in args.items()).split()


CONFIG_PARAMS = []
for sample_type_, datasets in GLOBAL_CONFIG.items():
    for dataset_name_, dataset in datasets.items():
        dataset_path = dataset.get("path", os.path.join(tempfile.gettempdir(), dataset_name_))
        configs = dataset.get("configs", {})
        for config_name in configs:
            config_params = configs[config_name]
            execution_args = config_params.get("execution_arg", [""])
            expected_accuracy_ = config_params.get("expected_accuracy", 100)
            absolute_tolerance_train_ = config_params.get("absolute_tolerance_train", 1)
            absolute_tolerance_test_ = config_params.get("absolute_tolerance_test", 0.5)
            weights_path_ = config_params.get("weights", None)
            for execution_arg_ in execution_args:
                config_path_ = EXAMPLES_DIR.joinpath(sample_type_, "configs", config_name)
                args_ = {"data": dataset_path, "weights": weights_path_, "config": str(config_path_)}

                test_config_ = {
                    "sample_type": sample_type_,
                    "expected_accuracy": expected_accuracy_,
                    "absolute_tolerance_train": absolute_tolerance_train_,
                    "absolute_tolerance_test": absolute_tolerance_test_,
                }
                CONFIG_PARAMS.append(tuple([test_config_, args_, execution_arg_, dataset_name_]))


def get_config_name(config_path):
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


def get_actual_acc(metrics_path):
    assert os.path.exists(metrics_path)
    with open(metrics_path, encoding="utf8") as metrics_file:
        metrics = json.load(metrics_file)
        actual_acc = metrics["Accuracy"]
    return actual_acc


@pytest.fixture(
    scope="module",
    params=CONFIG_PARAMS,
    ids=["-".join([p[0]["sample_type"], get_config_name(p[1]["config"])]) for p in CONFIG_PARAMS],
)
def _params(request, tmp_path_factory, dataset_dir, models_dir, weekly_tests):
    if not weekly_tests:
        pytest.skip("For weekly testing use --run-weekly-tests option.")
    test_config, args, execution_arg, _ = request.param
    if dataset_dir:
        args["data"] = os.path.join(dataset_dir, os.path.split(args["data"])[-1])
    if args["weights"]:
        if models_dir:
            args["weights"] = os.path.join(models_dir, args["weights"])
        if not os.path.exists(args["weights"]):
            raise FileExistsError("Weights file does not exist: {}".format(args["weights"]))
    else:
        del args["weights"]
    if execution_arg:
        args[execution_arg] = None
    checkpoint_save_dir = str(tmp_path_factory.mktemp("models"))
    checkpoint_save_dir = os.path.join(checkpoint_save_dir, execution_arg.replace("-", "_"))
    metric_save_dir = str(tmp_path_factory.mktemp("metrics"))
    metric_save_dir = os.path.join(metric_save_dir, execution_arg.replace("-", "_"))
    model_name = get_config_name(args["config"])
    args["metrics-dump"] = os.path.join(metric_save_dir, f"{model_name}_metrics.json")
    args["checkpoint-save-dir"] = os.path.join(checkpoint_save_dir, model_name)
    return {
        "test_config": test_config,
        "args": args,
    }


def run_sample(tc, args):
    mode = args["mode"]
    main = get_sample_fn(tc["sample_type"], modes=[mode])

    if tc["sample_type"] == "segmentation" and args["mode"] == "train":
        del args["metrics-dump"]
        del args["mode"]

    main(convert_to_argv(args))

    if "metrics-dump" in args:
        actual_acc = get_actual_acc(args["metrics-dump"])
        ref_acc = tc["expected_accuracy"]
        assert actual_acc == approx(
            ref_acc, abs=tc["absolute_tolerance_{}".format(mode)]
        ), "Test accuracy doesn't meet the expected accuracy within threshold."


def test_weekly_train_eval(_params, tmp_path):
    p = _params
    args = p["args"]
    tc = p["test_config"]

    args["mode"] = "train"
    run_sample(tc, args.copy())

    args["mode"] = "test"
    assert os.path.exists(args["checkpoint-save-dir"])
    args["resume"] = args["checkpoint-save-dir"]
    if "weights" in args:
        del args["weights"]

    run_sample(tc, args.copy())
