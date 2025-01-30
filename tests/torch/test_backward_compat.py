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

import pytest
import torch

from examples.common.sample_config import SampleConfig
from examples.torch.common.distributed import configure_distributed
from examples.torch.common.execution import ExecutionMode
from examples.torch.common.execution import get_device
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.model_loader import load_model
from nncf.api.compression import CompressionStage
from nncf.common.logging.logger import NNCFDeprecationWarning
from nncf.config import NNCFConfig
from nncf.torch import register_default_init_args
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.quantization.algo import QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME
from nncf.torch.quantization.algo import QuantizerBuilderStateVersion
from nncf.torch.quantization.external_quantizer import EXTERNAL_QUANTIZERS_STORAGE_PREFIX
from tests.cross_fw.shared.helpers import get_cli_dict_args
from tests.cross_fw.shared.paths import ROOT_PYTHONPATH_ENV
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import Command
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import get_basic_quantization_config
from tests.torch.test_sanity_sample import create_command_line

GLOBAL_CONFIG = {
    TEST_ROOT
    / "torch"
    / "data"
    / "configs"
    / "squeezenet1_1_cifar10_rb_sparsity_int8.json": [
        {
            "checkpoint_name": "squeezenet1_1_custom_cifar10_rb_sparsity_int8_dp.pth",
            "dataset": "cifar10",
            "execution_mode": ExecutionMode.GPU_DATAPARALLEL,
        },
        {
            "checkpoint_name": "squeezenet1_1_custom_cifar10_rb_sparsity_int8_ddp.pth",
            "dataset": "cifar10",
            "execution_mode": ExecutionMode.MULTIPROCESSING_DISTRIBUTED,
        },
    ],
}

CONFIG_PARAMS = []
for config_path_, cases_list_ in GLOBAL_CONFIG.items():
    for case_params_ in cases_list_:
        CONFIG_PARAMS.append(
            (
                config_path_,
                case_params_,
            )
        )


@pytest.fixture(
    scope="module", params=CONFIG_PARAMS, ids=["-".join([str(p[0]), p[1]["execution_mode"]]) for p in CONFIG_PARAMS]
)
def _params(request, backward_compat_models_path):
    if backward_compat_models_path is None:
        pytest.skip(
            "Path to models weights for backward compatibility testing is not set,"
            " use --backward-compat-models option."
        )
    config_path, case_params = request.param
    checkpoint_path = str(os.path.join(backward_compat_models_path, case_params["checkpoint_name"]))
    return {
        "sample_config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "execution_mode": case_params["execution_mode"],
        "dataset": case_params["dataset"],
    }


def test_model_can_be_loaded_with_resume(_params):
    p = _params
    sample_config_path = p["sample_config_path"]
    checkpoint_path = p["checkpoint_path"]

    config = SampleConfig.from_json(str(sample_config_path))
    nncf_config = NNCFConfig.from_json(str(sample_config_path))

    config.execution_mode = p["execution_mode"]

    config.current_gpu = 0
    config.device = get_device(config)
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        config.dist_url = "tcp://127.0.0.1:9898"
        config.dist_backend = "nccl"
        config.rank = 0
        config.world_size = 1
        configure_distributed(config)

    model_name = config["model"]
    model = load_model(
        model_name,
        pretrained=False,
        num_classes=config.get("num_classes", 1000),
        model_params=config.get("model_params"),
    )
    nncf_config = register_default_init_args(nncf_config, train_loader=create_ones_mock_dataloader(nncf_config))

    model.to(config.device)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    model, _ = prepare_model_for_execution(model, config)

    if config.distributed:
        compression_ctrl.distributed()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_state(model, checkpoint["state_dict"], is_resume=True)


def test_loaded_model_evals_according_to_saved_acc(_params, tmp_path, dataset_dir):
    p = _params
    config_path = p["sample_config_path"]
    checkpoint_path = p["checkpoint_path"]

    metrics_path = str(tmp_path.joinpath("metrics.json"))
    tmp_path = str(tmp_path)
    args = {}
    if not dataset_dir:
        dataset_dir = tmp_path
    args["data"] = dataset_dir
    args["dataset"] = p["dataset"]
    args["config"] = str(config_path)
    args["mode"] = "test"
    args["log-dir"] = tmp_path
    args["workers"] = 0  # Workaroundr the PyTorch MultiProcessingDataLoader issue
    args["seed"] = 1
    args["resume"] = checkpoint_path
    args["metrics-dump"] = metrics_path

    if p["execution_mode"] == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        args["multiprocessing-distributed"] = ""
    else:
        pytest.skip("DataParallel eval takes too long for this test to be run during pre-commit")

    runner = Command(create_command_line(get_cli_dict_args(args), "classification"), env=ROOT_PYTHONPATH_ENV)
    runner.run()

    with open(metrics_path, encoding="utf8") as metric_file:
        metrics = json.load(metric_file)
        # accuracy is rounded to hundredths
        assert torch.load(checkpoint_path)["best_acc1"] == pytest.approx(metrics["Accuracy"], abs=1e-2)


# BN Wrapping backward compatibility test


class ConvBNLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, (3, 3))
        self.bn = torch.nn.BatchNorm2d(9)
        self.conv1 = torch.nn.Conv2d(9, 3, (3, 3))
        self.bn1 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.bn1(self.conv1(x))


sd_without_nncf_bn_wrapping = {
    "conv.weight": torch.ones([9, 3, 3, 3]),
    "conv.bias": torch.ones([9]),
    "conv.nncf_padding_value": torch.ones([1]),
    "conv.pre_ops.0.op._num_bits": torch.ones([1]),
    "conv.pre_ops.0.op.signed_tensor": torch.ones([1]),
    "conv.pre_ops.0.op.enabled": torch.ones([1]),
    "conv.pre_ops.0.op.scale": torch.ones([9, 1, 1, 1]),
    "bn.weight": torch.ones([9]),
    "bn.bias": torch.ones([9]),
    "bn.running_mean": torch.ones([9]),
    "bn.running_var": torch.ones([9]),
    "bn.num_batches_tracked": torch.ones([]),
    "conv1.weight": torch.ones([3, 9, 3, 3]),
    "conv1.bias": torch.ones([3]),
    "conv1.nncf_padding_value": torch.ones([1]),
    "conv1.pre_ops.0.op._num_bits": torch.ones([1]),
    "conv1.pre_ops.0.op.signed_tensor": torch.ones([1]),
    "conv1.pre_ops.0.op.enabled": torch.ones([1]),
    "conv1.pre_ops.0.op.scale": torch.ones([3, 1, 1, 1]),
    "bn1.weight": torch.ones([3]),
    "bn1.bias": torch.ones([3]),
    "bn1.running_mean": torch.ones([3]),
    "bn1.running_var": torch.ones([3]),
    "bn1.num_batches_tracked": torch.ones([]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}./nncf_model_input_0|OUTPUT._num_bits": torch.ones([1]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}./nncf_model_input_0|OUTPUT.signed_tensor": torch.ones([1]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}./nncf_model_input_0|OUTPUT.enabled": torch.ones([1]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}./nncf_model_input_0|OUTPUT.scale": torch.ones([1]),
    # Old bn layer names:            |||||||||||
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}.ConvBNLayer/BatchNorm2d[bn]/batch_norm_0|OUTPUT._num_bits": torch.ones([1]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}.ConvBNLayer/BatchNorm2d[bn]/batch_norm_0|OUTPUT.signed_tensor": torch.ones(
        [1]
    ),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}.ConvBNLayer/BatchNorm2d[bn]/batch_norm_0|OUTPUT.enabled": torch.ones([1]),
    f"{EXTERNAL_QUANTIZERS_STORAGE_PREFIX}.ConvBNLayer/BatchNorm2d[bn]/batch_norm_0|OUTPUT.scale": torch.ones([1]),
}

compression_state_without_bn_wrapping = {
    "builder_state": {
        "quantization": {
            "quantizer_setup": {
                "quantization_points": {
                    1: {
                        "qip": {"target_node_name": "/nncf_model_input_0", "input_port_id": None},
                        "qip_class": "ActivationQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": None,
                            "per_channel": False,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                    },
                    # Old bn layer name:                         |||||||||||
                    2: {
                        "qip": {"target_node_name": "ConvBNLayer/BatchNorm2d[bn]/batch_norm_0", "input_port_id": None},
                        "qip_class": "ActivationQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": None,
                            "per_channel": False,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                    },
                    4: {
                        "qip": {"target_node_name": "ConvBNLayer/NNCFConv2d[conv]/conv2d_0"},
                        "qip_class": "WeightQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": True,
                            "per_channel": True,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                    },
                    5: {
                        "qip": {"target_node_name": "ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"},
                        "qip_class": "WeightQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": True,
                            "per_channel": True,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                    },
                },
                "unified_scale_groups": {},
                "shared_input_operation_set_groups": {0: [1, 4], 1: [2, 5]},
            },
            "build_time_metric_infos": {"aq_potential_num": 3, "wq_potential_num": 4},
        }
    },
    "ctrl_state": {
        "quantization": {
            "loss_state": None,
            "scheduler_state": {"current_step": -1, "current_epoch": -1},
            "compression_stage": CompressionStage.FULLY_COMPRESSED,
        }
    },
}


def test_quantization_ckpt_without_wrapped_bn_loading():
    model = ConvBNLayer()
    config = get_basic_quantization_config(input_info={"sample_size": [1, 3, 100, 100]})
    register_bn_adaptation_init_args(config)
    with pytest.warns(NNCFDeprecationWarning):
        compressed_model, _ = create_compressed_model_and_algo_for_test(
            model, config, compression_state=compression_state_without_bn_wrapping
        )
    with pytest.warns(NNCFDeprecationWarning):
        _ = load_state(compressed_model, sd_without_nncf_bn_wrapping, is_resume=True)


old_comp_state = {
    "ctrl_state": {
        "quantization": {
            "loss_state": None,
            "scheduler_state": {"current_step": -1, "current_epoch": -1},
            "compression_stage": CompressionStage.FULLY_COMPRESSED,
        }
    },
    "builder_state": {
        "quantization": {
            "quantizer_setup": {
                "quantization_points": {
                    1: {
                        "qip": {"target_node_name": "/nncf_model_input_0", "input_port_id": None},
                        "qip_class": "ActivationQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": None,
                            "per_channel": False,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                    },
                    2: {
                        "qip": {
                            "target_node_name": "ConvBNLayer/NNCFBatchNorm2d[bn]/batch_norm_0",
                            "input_port_id": None,
                        },
                        "qip_class": "ActivationQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": None,
                            "per_channel": False,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                    },
                    4: {
                        "qip": {"target_node_name": "ConvBNLayer/NNCFConv2d[conv]/conv2d_0"},
                        "qip_class": "WeightQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": True,
                            "per_channel": True,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                    },
                    5: {
                        "qip": {"target_node_name": "ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"},
                        "qip_class": "WeightQuantizationInsertionPoint",
                        "qconfig": {
                            "num_bits": 8,
                            "mode": "symmetric",
                            "signedness_to_force": True,
                            "per_channel": True,
                        },
                        "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                    },
                },
                "unified_scale_groups": {},
                "shared_input_operation_set_groups": {0: [1, 4], 1: [2, 5]},
            },
            "build_time_metric_infos": {"aq_potential_num": 3, "wq_potential_num": 4},
        }
    },
}

reference_new_builder_state = {
    "quantization": {
        "quantizer_setup": {
            "quantization_points": {
                1: {
                    "target_point": {
                        "target_type": {"name": "OPERATOR_POST_HOOK"},
                        "input_port_id": None,
                        "target_node_name": "/nncf_model_input_0",
                    },
                    "qspec": {
                        "num_bits": 8,
                        "mode": "symmetric",
                        "signedness_to_force": None,
                        "narrow_range": False,
                        "half_range": False,
                        "scale_shape": (1,),
                        "logarithm_scale": False,
                        "is_quantized_on_export": False,
                        "compression_lr_multiplier": None,
                    },
                    "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                },
                2: {
                    "target_point": {
                        "target_type": {"name": "OPERATOR_POST_HOOK"},
                        "input_port_id": None,
                        "target_node_name": "ConvBNLayer/NNCFBatchNorm2d[bn]/batch_norm_0",
                    },
                    "qspec": {
                        "num_bits": 8,
                        "mode": "symmetric",
                        "signedness_to_force": None,
                        "narrow_range": False,
                        "half_range": False,
                        "scale_shape": (1,),
                        "logarithm_scale": False,
                        "is_quantized_on_export": False,
                        "compression_lr_multiplier": None,
                    },
                    "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                },
                4: {
                    "target_point": {
                        "target_type": {"name": "OPERATION_WITH_WEIGHTS"},
                        "input_port_id": None,
                        "target_node_name": "ConvBNLayer/NNCFConv2d[conv]/conv2d_0",
                    },
                    "qspec": {
                        "num_bits": 8,
                        "mode": "symmetric",
                        "signedness_to_force": True,
                        "narrow_range": True,
                        "half_range": False,
                        "scale_shape": (9, 1, 1, 1),
                        "logarithm_scale": False,
                        "is_quantized_on_export": True,
                        "compression_lr_multiplier": None,
                    },
                    "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv]/conv2d_0"],
                },
                5: {
                    "target_point": {
                        "target_type": {"name": "OPERATION_WITH_WEIGHTS"},
                        "input_port_id": None,
                        "target_node_name": "ConvBNLayer/NNCFConv2d[conv1]/conv2d_0",
                    },
                    "qspec": {
                        "num_bits": 8,
                        "mode": "symmetric",
                        "signedness_to_force": True,
                        "narrow_range": True,
                        "half_range": False,
                        "scale_shape": (3, 1, 1, 1),
                        "logarithm_scale": False,
                        "is_quantized_on_export": True,
                        "compression_lr_multiplier": None,
                    },
                    "directly_quantized_operator_node_names": ["ConvBNLayer/NNCFConv2d[conv1]/conv2d_0"],
                },
            },
            "unified_scale_groups": {},
            "shared_input_operation_set_groups": {0: [1, 4], 1: [2, 5]},
        },
        "build_time_metric_infos": {"aq_potential_num": 3, "wq_potential_num": 4},
        QUANTIZER_BUILDER_STATE_VERSION_SAVE_NAME: max(QuantizerBuilderStateVersion).value,
    }
}


def test_comp_state_without_qspec():
    model = ConvBNLayer()
    nncf_config = get_basic_quantization_config(input_info={"sample_size": [1, 3, 100, 100]})
    nncf_config["compression"]["overflow_fix"] = "disable"
    register_bn_adaptation_init_args(nncf_config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, nncf_config, compression_state=old_comp_state
    )
    curr_comp_state = compression_ctrl.get_compression_state()
    assert curr_comp_state["builder_state"] == reference_new_builder_state
