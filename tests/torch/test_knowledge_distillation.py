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

from collections.abc import Iterable
from copy import deepcopy
from functools import reduce
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

from nncf import NNCFConfig
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import get_model_device
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import fill_params_of_model_by_normal
from tests.torch.helpers import get_empty_config
from tests.torch.quantization.quantization_helpers import create_rank_dataloader
from tests.torch.quantization.quantization_helpers import distributed_init_test_default
from tests.torch.quantization.quantization_helpers import post_compression_test_distr_init
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.torch.test_models.synthetic import ContainersOutputsModel
from tests.torch.test_models.synthetic import PartlyNonDifferentialOutputsModel

KEY_TO_KD_PARAMETERS = "kd"


def get_device_str(inference_type: str, gpu_id: int):
    if inference_type == "cpu":
        return "cpu"
    if gpu_id is not None:
        return "cuda:{}".format(gpu_id)

    return "cuda"


def get_kd_config(config: NNCFConfig, kd_type="mse", scale=1, temperature=None) -> NNCFConfig:
    if isinstance(config.get("compression", {}), dict):
        config["compression"] = [config["compression"]] if config.get("compression", None) is not None else []
    kd_algo_dict = {"algorithm": "knowledge_distillation", "type": kd_type, "scale": scale}
    if temperature is not None:
        kd_algo_dict["temperature"] = temperature
    config["compression"].append(kd_algo_dict)
    return config


def get_sparsity_config_with_sparsity_init(config: NNCFConfig, sparsity_init=0.5) -> NNCFConfig:
    config["compression"]["sparsity_init"] = sparsity_init
    return config


@pytest.mark.parametrize(
    "inference_type",
    [
        "cpu",
        pytest.param("single_GPU", marks=pytest.mark.cuda),
        pytest.param("DP", marks=pytest.mark.cuda),
        pytest.param("DDP", marks=pytest.mark.cuda),
    ],
)
def test_knowledge_distillation_training_process(inference_type: str):
    if not torch.cuda.is_available() and inference_type != "cpu":
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    torch.manual_seed(1)
    input_size = [1, 1, 8, 8]
    sparsity_level = 0.3
    config = get_sparsity_config_with_sparsity_init(
        get_basic_magnitude_sparsity_config(input_sample_size=input_size), sparsity_level
    )
    if inference_type == "DDP":
        ngpus_per_node = torch.cuda.device_count()
        config.world_size = ngpus_per_node
        torch.multiprocessing.spawn(
            run_test_training, nprocs=ngpus_per_node, args=(config, inference_type, ngpus_per_node), join=True
        )
    else:
        run_test_training(None, config, inference_type, None)


def run_actual(
    model: nn.Module, config: NNCFConfig, inference_type: str, mock_dataloader: Iterable, ngpus_per_node=None
) -> Tuple[List[torch.Tensor], NNCFNetwork]:
    config = get_kd_config(config)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if inference_type == "DDP":
        model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, model)
    elif inference_type == "DP":
        model = torch.nn.DataParallel(model)
    optimizer = SGD(model.parameters(), lr=1e-02, weight_decay=1e-02)
    model.train()
    output_storage = []
    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        output = model(input_)
        output_storage.append(output)
        loss = compression_ctrl.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output_storage, model


def run_reference(
    model: nn.Module, config: NNCFConfig, inference_type: str, mock_dataloader: Iterable, ngpus_per_node=None
) -> List[torch.Tensor]:
    model = deepcopy(model)
    kd_model = deepcopy(model)
    mse = torch.nn.MSELoss().cuda()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if inference_type == "DDP":
        model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, model)
    elif inference_type == "DP":
        model = torch.nn.DataParallel(model)
        kd_model = torch.nn.DataParallel(kd_model)
    optimizer = SGD(model.parameters(), lr=1e-02, weight_decay=1e-02)
    model.train()
    kd_model.train()
    output_storage = []
    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        output = model(input_)
        kd_output = kd_model(input_)
        output_storage.append(output)
        loss = mse(output, kd_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output_storage


def run_test_training(gpu, config: NNCFConfig, inference_type: str, ngpus_per_node: int):
    torch.manual_seed(2)
    number_of_iters = 10
    batch_size = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()
    config["input_info"]["sample_size"] = [1, 1, 8, 8]
    if inference_type == "DDP":
        distributed_init_test_default(gpu, ngpus_per_node, config)
        mock_dataloader = create_rank_dataloader(config, gpu, batch_size * number_of_iters, batch_size=batch_size)
    else:
        mock_dataloader = create_ones_mock_dataloader(
            config, num_samples=batch_size * number_of_iters, batch_size=batch_size
        )
    model_device = get_device_str(inference_type, gpu)
    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model, std=0.5)
    model.to(model_device)
    dumped_orig_model = deepcopy(model)

    actual_outputs, actual_model = run_actual(deepcopy(model), config, inference_type, mock_dataloader, ngpus_per_node)
    reference_outputs = run_reference(model, config, inference_type, mock_dataloader, ngpus_per_node)
    assert reduce(lambda a, b: a and torch.allclose(b[0], b[1]), zip(actual_outputs, reference_outputs), True), (
        "Outputs of model with actual KD implementation doesn't match outputs from model with reference "
        "Knowledge Distillation implementation"
    )

    for param1, param2 in zip(
        [param for name, param in filter(lambda x: KEY_TO_KD_PARAMETERS in x[0], actual_model.named_parameters())],
        dumped_orig_model.parameters(),
    ):
        assert torch.allclose(param1, param2), (
            "Weights of dumped original model doesn't match weights of original "
            "model used for distillation (most likely weights of original model"
            " are being corrupted due training)"
        )


def test_loss_outputs_parsing():
    mse = torch.nn.MSELoss()
    input_size = [1, 1, 8, 8]
    model = PartlyNonDifferentialOutputsModel(input_size)
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    batch_size = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(
            get_basic_magnitude_sparsity_config(input_sample_size=input_size), sparsity_level
        )
    )
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(), batch_size=batch_size)
    compression_ctrl.scheduler.epoch_step()
    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)
        loss_outputs = []
        for tensor1, tensor2 in zip(outputs, kd_outputs):
            if tensor1.requires_grad:
                loss_outputs.append((tensor1, tensor2))

        reference_kd_loss = sum(mse(item[0], item[1]) for item in loss_outputs)
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)


def test_knowledge_distillation_outputs_containers_parsing():
    mse = torch.nn.MSELoss()
    input_size = [1, 1, 8, 8]
    model = ContainersOutputsModel(input_size)
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    batch_size = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(
            get_basic_magnitude_sparsity_config(input_sample_size=input_size), sparsity_level
        )
    )
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(), batch_size=batch_size)
    compression_ctrl.scheduler.epoch_step()
    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)

        reference_kd_loss = (
            mse(outputs["xa"], kd_outputs["xa"])
            + mse(outputs["xb_and_xc"][0], kd_outputs["xb_and_xc"][0])
            + mse(outputs["xb_and_xc"][1], kd_outputs["xb_and_xc"][1])
        )
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)


@pytest.mark.parametrize(
    ("kd_loss_type", "scale", "temperature"),
    [("mse", 1, None), ("softmax", 1, 1), ("mse", 10, None), ("softmax", 1, 5), ("softmax", 2, 5)],
)
def test_knowledge_distillation_loss_types(kd_loss_type: str, scale, temperature):
    torch.manual_seed(2)
    if kd_loss_type == "softmax":

        def kd_loss_fn(ref_outputs, compressed_model_outputs) -> torch.Tensor:
            return (
                scale
                * -(
                    nn.functional.log_softmax(compressed_model_outputs / temperature, dim=1)
                    * nn.functional.softmax(ref_outputs / temperature, dim=1)
                ).mean()
                * (temperature * temperature * compressed_model_outputs.shape[1])
            )

    else:

        def mse_loss(x, y):
            return scale * F.mse_loss(x, y)

        kd_loss_fn = mse_loss
    input_size = [1, 100]
    batch_size = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()

    model = nn.Sequential(nn.Linear(in_features=input_size[-1], out_features=10), nn.Sigmoid())

    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.5
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(
            get_basic_magnitude_sparsity_config(input_sample_size=input_size), sparsity_level
        ),
        kd_type=kd_loss_type,
        scale=scale,
        temperature=temperature,
    )
    config["compression"][-1]["type"] = kd_loss_type
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(), batch_size=batch_size)
    compression_ctrl.scheduler.epoch_step()
    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)
        reference_kd_loss = kd_loss_fn(kd_outputs, outputs)
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)


@pytest.mark.parametrize("algo", ("magnitude_sparsity", "rb_sparsity"))
def test_kd_sparsity_statistics(algo: str):
    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model)
    model_with_kd = deepcopy(model)
    config = get_empty_config()
    sparsity_init = 0.5
    config["compression"] = {"algorithm": algo, "sparsity_init": sparsity_init}
    config_with_kd = deepcopy(config)
    config_with_kd = get_kd_config(config_with_kd)

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model_with_kd, compression_ctrl_with_kd = create_compressed_model_and_algo_for_test(model_with_kd, config_with_kd)
    statistics = compression_ctrl.statistics()
    statistics_with_kd = compression_ctrl_with_kd.statistics()
    assert (
        getattr(statistics, algo).model_statistics.sparsity_level
        == getattr(statistics_with_kd, algo).model_statistics.sparsity_level
    )
    assert (
        getattr(statistics, algo).model_statistics.sparsity_level_for_layers
        == getattr(statistics_with_kd, algo).model_statistics.sparsity_level_for_layers
    )


@pytest.mark.parametrize("device_placing", ["before", "after"])
@pytest.mark.parametrize(
    "inference_type",
    [
        "cpu",
        pytest.param("single_GPU", marks=pytest.mark.cuda),
        pytest.param("DP", marks=pytest.mark.cuda),
        pytest.param("DDP", marks=pytest.mark.cuda),
    ],
)
def test_model_device_before_create_compressed_model(device_placing, inference_type):
    if not torch.cuda.is_available() and inference_type != "cpu":
        pytest.skip("Skipping CUDA test cases for CPU only setups")
    input_size = [1, 1, 8, 8]
    config = NNCFConfig()
    config = get_kd_config(config)
    config.update(
        {
            "input_info": {
                "sample_size": input_size,
            },
        }
    )
    if inference_type == "DDP":
        ngpus_per_node = torch.cuda.device_count()
        config.world_size = ngpus_per_node
        torch.multiprocessing.spawn(
            run_training_for_device_testing,
            nprocs=ngpus_per_node,
            args=(config, inference_type, ngpus_per_node, device_placing),
            join=True,
        )
    else:
        run_training_for_device_testing(None, config, inference_type, None, device_placing=device_placing)


def run_training_for_device_testing(
    gpu, config: NNCFConfig, inference_type: str, ngpus_per_node: int, device_placing: str
):
    number_of_iters = 1
    batch_size = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()
    config["input_info"]["sample_size"] = [1, 1, 8, 8]
    if inference_type == "DDP":
        distributed_init_test_default(gpu, ngpus_per_node, config)
        mock_dataloader = create_rank_dataloader(config, gpu, batch_size * number_of_iters, batch_size=batch_size)
    else:
        mock_dataloader = create_ones_mock_dataloader(
            config, num_samples=batch_size * number_of_iters, batch_size=batch_size
        )
    model_device = get_device_str(inference_type, gpu)
    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model, std=0.5)

    if device_placing == "before":
        model.to(model_device)

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    if inference_type == "DDP":
        model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, model)
    elif inference_type == "DP":
        model = torch.nn.DataParallel(model)

    optimizer = SGD(model.parameters(), lr=1e-02)
    model.train()
    output_storage = []

    if device_placing == "after":
        model.to(model_device)

    for _, (input_, __) in enumerate(mock_dataloader):
        input_ = input_.to(get_model_device(model))
        output = model(input_)
        output_storage.append(output)
        loss = compression_ctrl.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class KDOutputModel(torch.nn.Module):
    def __init__(self, target_shapes: List[Tuple[int]]):
        super().__init__()
        self.mock_param = torch.nn.Parameter(torch.ones([1]))
        self.target_shapes = target_shapes

    def forward(self, *args, **kwargs):
        retval = []
        for shape in self.target_shapes:
            retval.append(torch.ones(shape).to(self.mock_param.device) * self.mock_param)
        return retval


class CustomOutputWeightedModel(torch.nn.Module):
    def __init__(self, input_shape: List[int], outputs_dim_numbers_list: List[int]):
        super().__init__()
        self.outputs_dim_numbers_list = outputs_dim_numbers_list
        # linear layer would be compressed and will lead to different teacher (FP) and student (compressed) model
        # outputs and hence non zero KD loss value (if outputs are not ignored through special logic)
        self.linear = torch.nn.Linear(in_features=input_shape[3], out_features=input_shape[3])

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        # creating outputs with different number of dims
        output = {
            4: x,
            3: x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]),
            2: x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]]),
            1: x.view([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]]),
        }
        return tuple(filter(lambda item: len(item.size()) in self.outputs_dim_numbers_list, output.values()))


@pytest.mark.parametrize(
    "outputs_dim_numbers_list, kd_type, is_zero",
    [
        ([4], "softmax", True),
        ([3], "softmax", True),
        ([1], "softmax", True),
        ([2], "softmax", False),
        ([3, 2, 1], "softmax", False),
        ([4], "mse", False),
        ([3], "mse", False),
        ([2], "mse", False),
        ([1], "mse", True),
        ([4, 2, 1], "mse", False),
    ],
)
def test_kd_incompatible_output_shapes_handling(outputs_dim_numbers_list, kd_type, is_zero):
    """
    Checks ignorance behavior (kd loss is zero) for different model output shape sizes
    :param dim_numbers_list: a list of dim numbers of model output tensors
        examples: dim_numbers_list = [4] -> model outputs should be [*, *, *, *]
        dim_numbers_list = [4, 2] -> model outputs should be ([*, *, *, *], [*, *])
    :param kd_type: type of knowledge distillation loss
    "param is_zero: if given type of model outputs should be ignored than kd loss value should be zero
    """
    input_size = [1, 2, 3, 4]
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(
            get_basic_magnitude_sparsity_config(input_sample_size=input_size), sparsity_init=0.5
        ),
        kd_type=kd_type,
    )
    model = CustomOutputWeightedModel(input_size, outputs_dim_numbers_list)
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.scheduler.epoch_step()
    compressed_model.train()
    input_ = torch.normal(0, std=0.5, size=input_size)
    compressed_model.forward(input_)
    kd_loss = compression_ctrl.loss()
    assert torch.allclose(kd_loss, torch.zeros([1])) == is_zero
