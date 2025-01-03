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

import copy
from typing import Callable, Dict, List, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from nncf import NNCFConfig
from nncf.torch.layer_utils import CompressionParameter
from tests.torch.helpers import LeNet
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import RandomDatasetMock
from tests.torch.helpers import create_initialized_compressed_model
from tests.torch.helpers import create_random_mock_dataloader
from tests.torch.helpers import get_grads
from tests.torch.helpers import set_torch_seed
from tests.torch.quantization.test_algo_quantization import get_quantization_config_without_range_init
from tests.torch.sparsity.rb.test_algo import get_basic_sparsity_config

ALGO_NAME_TO_PATH_MAP = {
    "quantization": "nncf.torch.quantization",
    "rb_sparsity": "nncf.torch.sparsity.rb",
}


def get_quantization_config() -> NNCFConfig:
    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    config["compression"]["initializer"] = {"range": {"num_init_samples": 10}}
    return config


def get_sparsity_config() -> NNCFConfig:
    config = get_basic_sparsity_config([1, *LeNet.INPUT_SIZE])
    return config


def get_config_algorithms(config: NNCFConfig) -> List[Dict]:
    if isinstance(config["compression"], list):
        algorithms = config["compression"]
    else:
        algorithms = [config["compression"]]
    return algorithms


def add_multiplier_to_config(
    config: NNCFConfig, local_multiplier: float = None, global_multiplier: float = None
) -> NNCFConfig:
    config = copy.deepcopy(config)

    if local_multiplier is not None:
        algorithms = get_config_algorithms(config)

        for algo in algorithms:
            algo.update({"compression_lr_multiplier": local_multiplier})

    if global_multiplier is not None:
        config["compression_lr_multiplier"] = global_multiplier

    return config


def get_multipliers_from_config(config: NNCFConfig) -> Dict[str, float]:
    algo_to_multipliers = {}

    algorithms = get_config_algorithms(config)
    global_multiplier = config.get("compression_lr_multiplier", 1)
    for algo in algorithms:
        algo_name = algo["algorithm"]
        algo_to_multipliers[algo_name] = algo.get("compression_lr_multiplier", global_multiplier)

    return algo_to_multipliers


def merge_configs(configs: List[NNCFConfig], use_algo_list: bool = True) -> NNCFConfig:
    res_config = None
    algorithms = []

    for source_config in configs:
        source_config = copy.deepcopy(source_config)

        algorithms.extend(get_config_algorithms(source_config))
        del source_config["compression"]

        if res_config is None:
            res_config = source_config
        res_config.update(source_config)

    if not use_algo_list:
        if len(algorithms) > 1:
            raise Exception("If there is more than one algorithm you could use only use_algo_list=True")
        res_config["compression"] = algorithms[0]
    else:
        res_config["compression"] = algorithms

    res_config["model"] = "merged_model"
    return res_config


def get_configs_building_params() -> List[Dict]:
    res = []
    get_orig_config_fns = [get_quantization_config, get_sparsity_config]
    num_orig_configs = len(get_orig_config_fns)

    for global_multiplier in [0, 1, 10]:
        res.append(
            {
                "get_orig_config_fns": get_orig_config_fns,
                "multipliers": [None] * num_orig_configs,
                "global_multiplier": global_multiplier,
                "use_algo_list": True,
            }
        )

    global_multiplier = 10
    multipliers = [global_multiplier * (1.1**i) for i in range(num_orig_configs)]

    res.append(
        {
            "get_orig_config_fns": get_orig_config_fns,
            "multipliers": multipliers,
            "global_multiplier": global_multiplier,
            "use_algo_list": True,
        }
    )

    for i in range(num_orig_configs):
        cur_multipliers = copy.deepcopy(multipliers)
        cur_multipliers[i] = None
        res.append(
            {
                "get_orig_config_fns": get_orig_config_fns,
                "multipliers": cur_multipliers,
                "global_multiplier": None,
                "use_algo_list": True,
            }
        )

    for get_orig_config_fn in get_orig_config_fns:
        for use_algo_list in [False, True]:
            for global_multiplier, multiplier in [(11, 10), (11, None), (None, 10)]:
                res.append(
                    {
                        "get_orig_config_fns": [get_orig_config_fn],
                        "multipliers": [multiplier],
                        "global_multiplier": global_multiplier,
                        "use_algo_list": use_algo_list,
                    }
                )

    return res


def create_initialized_lenet_model_and_dataloader(config: NNCFConfig) -> Tuple[nn.Module, DataLoader]:
    with set_torch_seed():
        train_loader = create_random_mock_dataloader(config, num_samples=10)
        model = LeNet()
        for param in model.parameters():
            nn.init.normal_(param)
        model = create_initialized_compressed_model(model, config, train_loader)
    return model, train_loader


@pytest.fixture(name="configs_building_params", params=get_configs_building_params())
def configs_building_params_(request) -> Dict:
    return request.param


@pytest.fixture(name="ref_configs")
def ref_configs_(configs_building_params: Dict) -> List[NNCFConfig]:
    return [get_ref_config_fn() for get_ref_config_fn in configs_building_params["get_orig_config_fns"]]


@pytest.fixture(name="ref_config")
def ref_config_(ref_configs, configs_building_params) -> NNCFConfig:
    return merge_configs(ref_configs, configs_building_params["use_algo_list"])


@pytest.fixture(name="target_configs")
def target_configs_(ref_configs: List[NNCFConfig], configs_building_params: Dict) -> List[NNCFConfig]:
    return [
        add_multiplier_to_config(config, local_multiplier=multiplier)
        for config, multiplier in zip(ref_configs, configs_building_params["multipliers"])
    ]


@pytest.fixture(name="target_config")
def target_config_(target_configs: List[NNCFConfig], configs_building_params: Dict) -> NNCFConfig:
    target_config = merge_configs(target_configs, configs_building_params["use_algo_list"])
    return add_multiplier_to_config(target_config, global_multiplier=configs_building_params["global_multiplier"])


@pytest.fixture(name="get_ref_lenet_model_and_dataloader")
def get_ref_lenet_model_and_dataloader_(ref_config: NNCFConfig) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_lenet_model_and_dataloader(ref_config)

    return f


@pytest.fixture(name="get_target_lenet_model_and_dataloader")
def get_target_lenet_model_and_dataloader_(target_config: NNCFConfig) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_lenet_model_and_dataloader(target_config)

    return f


class OneParameterModel(nn.Module):
    INPUT_SIZE = (0,)

    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, _x):
        return self.param.sum()


def get_one_parameter_model_creation_params(for_training: bool = False) -> List[Dict]:
    params = []
    for init_requires_grad in [False, True]:
        requires_grad_settings_list = [
            [],
            [("attr", False)],
            [("attr", True)],
            [("fn", False)],
            [("fn", True)],
            [("attr", not init_requires_grad), ("attr", True)],
            [("fn", not init_requires_grad), ("fn", True)],
            [("attr", not init_requires_grad), ("fn", True)],
            [("fn", not init_requires_grad), ("attr", True)],
        ]

        for requires_grad_settings in requires_grad_settings_list:
            trainable = init_requires_grad if len(requires_grad_settings) == 0 else requires_grad_settings[-1][1]
            if for_training and not trainable:
                continue
            multipliers = [0.1, 1, 10] if trainable else [0.1]

            for multiplier in multipliers:
                params.append(
                    {
                        "init_requires_grad": init_requires_grad,
                        "requires_grad_settings": requires_grad_settings,
                        "multiplier": multiplier,
                    }
                )
    return params


def create_initialized_one_parameter_model_and_dataloader(
    parameter_cls: type,
    init_requires_grad: bool,
    requires_grad_settings: List[Tuple[str, bool]],
    multiplier: float = None,
) -> [nn.Module, DataLoader]:
    with set_torch_seed():
        data = torch.randn(size=(1, 1, 5, 5))
        if parameter_cls is nn.Parameter:
            param = parameter_cls(data, requires_grad=init_requires_grad)
        elif parameter_cls is CompressionParameter:
            param = parameter_cls(data, requires_grad=init_requires_grad, compression_lr_multiplier=multiplier)
        else:
            raise Exception(f"Unsupported parameter type: {parameter_cls}")

    for setting_type, requires_grad in requires_grad_settings:
        if setting_type == "attr":
            param.requires_grad = requires_grad
        elif setting_type == "fn":
            param.requires_grad_(requires_grad)
        else:
            raise Exception(f"Unsupported setting type: {setting_type}")

    model = OneParameterModel(param)
    train_loader = DataLoader(
        RandomDatasetMock(model.INPUT_SIZE), batch_size=1, shuffle=False, num_workers=0, drop_last=True
    )
    return model, train_loader


@pytest.fixture(name="get_ref_one_parameter_model_and_dataloader")
def get_ref_one_parameter_model_and_dataloader_(
    one_parameter_model_creation_params: Dict,
) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_one_parameter_model_and_dataloader(
            nn.Parameter, **one_parameter_model_creation_params
        )

    return f


@pytest.fixture(name="get_target_one_parameter_model_and_dataloader")
def get_target_one_parameter_model_and_dataloader_(
    one_parameter_model_creation_params: Dict,
) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_one_parameter_model_and_dataloader(
            CompressionParameter, **one_parameter_model_creation_params
        )

    return f


def perform_model_training_steps(model: nn.Module, train_loader: DataLoader, num_steps: int = 1) -> nn.Module:
    with set_torch_seed():
        train_loader = iter(train_loader)
        optimizer = SGD(model.parameters(), lr=0.1)

        for _ in range(num_steps):
            optimizer.zero_grad()
            x, y_gt = next(train_loader)
            y = model(x)
            loss = F.mse_loss(y.sum(), y_gt)

            loss.backward()
            optimizer.step()

    return model


def get_params_grouped_by_algorithms(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    cls_name_to_params = {}
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        full_cls_name = ".".join([module.__class__.__module__, module.__class__.__name__])
        if full_cls_name not in cls_name_to_params:
            cls_name_to_params[full_cls_name] = []
        cls_name_to_params[full_cls_name].extend(params)

    algo_name_to_params = {}
    for cls_name, params in cls_name_to_params.items():
        params = [param for param in params if param.requires_grad]
        if len(params) == 0:
            continue

        algo_name = "regular"
        for cur_algo_name, cur_algo_path in ALGO_NAME_TO_PATH_MAP.items():
            if cur_algo_path in cls_name:
                algo_name = cur_algo_name

        if algo_name not in algo_name_to_params:
            algo_name_to_params[algo_name] = []
        algo_name_to_params[algo_name].extend(params)

    return algo_name_to_params


def get_lenet_params_after_training_steps(
    model: nn.Module, train_loader: DataLoader, num_steps: int = 1
) -> Dict[str, List[nn.Parameter]]:
    with set_torch_seed():
        model = perform_model_training_steps(model, train_loader, num_steps)
    return get_params_grouped_by_algorithms(model)


def get_one_parameter_model_params_after_training_steps(
    model: nn.Module, train_loader: DataLoader, num_steps: int = 1
) -> List[nn.Parameter]:
    with set_torch_seed():
        model = perform_model_training_steps(model, train_loader, num_steps)
    return list(model.parameters())


def test_if_algorithms_add_params(
    get_target_lenet_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]], ref_config: NNCFConfig
):
    algo_to_params = get_lenet_params_after_training_steps(*get_target_lenet_model_and_dataloader(), num_steps=0)
    algo_names = get_multipliers_from_config(ref_config).keys()

    assert sorted(algo_to_params.keys()) == sorted(list(algo_names) + ["regular"])


@pytest.mark.parametrize("one_parameter_model_creation_params", get_one_parameter_model_creation_params())
def test_if_parameter_is_initialized_correctly(
    get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
):
    ref_model, _ref_loader = get_ref_one_parameter_model_and_dataloader()
    target_model, target_loader = get_target_one_parameter_model_and_dataloader()

    assert pytest.approx(ref_model.param.data) == target_model.param.data
    assert ref_model.param.requires_grad == target_model.param.requires_grad

    if ref_model.param.requires_grad:
        get_one_parameter_model_params_after_training_steps(target_model, target_loader)
    else:
        with pytest.raises(Exception):
            get_one_parameter_model_params_after_training_steps(target_model, target_loader)


def check_if_grads_are_multiplied(ref_params: List[nn.Parameter], target_params: List[nn.Parameter], multiplier: float):
    ref_grads = get_grads(ref_params)
    ref_grads = [multiplier * grad for grad in ref_grads]
    target_grads = get_grads(target_params)

    PTTensorListComparator.check_equal(ref_grads, target_grads)


def test_if_setting_multipliers_in_config_multiplies_grads_values(
    get_ref_lenet_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    get_target_lenet_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    target_config: NNCFConfig,
):
    ref_params = get_lenet_params_after_training_steps(*get_ref_lenet_model_and_dataloader())
    target_params = get_lenet_params_after_training_steps(*get_target_lenet_model_and_dataloader())
    multipliers = get_multipliers_from_config(target_config)
    multipliers["regular"] = 1

    for algo, val in ref_params.items():
        check_if_grads_are_multiplied(val, target_params[algo], multipliers[algo])


@pytest.mark.parametrize(
    "one_parameter_model_creation_params", get_one_parameter_model_creation_params(for_training=True)
)
def test_if_setting_multiplier_in_parameter_multiplies_grads_values(
    get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    one_parameter_model_creation_params: Dict,
):
    ref_params = get_one_parameter_model_params_after_training_steps(*get_ref_one_parameter_model_and_dataloader())
    target_params = get_one_parameter_model_params_after_training_steps(
        *get_target_one_parameter_model_and_dataloader()
    )

    assert target_params[0].requires_grad
    check_if_grads_are_multiplied(ref_params, target_params, one_parameter_model_creation_params["multiplier"])


def check_if_zero_multiplier_freezes_training(
    orig_params: List[nn.Parameter], params: List[nn.Parameter], multiplier: float
):
    if multiplier == 0:
        PTTensorListComparator.check_equal(orig_params, params)
    else:
        PTTensorListComparator.check_not_equal(orig_params, params)


def get_params_diff(orig_params: List[nn.Parameter], params: List[nn.Parameter]) -> List[torch.Tensor]:
    param_diffs = []
    for param, orig_param in zip(params, orig_params):
        param_diffs.append((param - orig_param).abs())
    return param_diffs


def check_params_affect_training_speed(
    orig_params: List[nn.Parameter],
    ref_params: List[nn.Parameter],
    target_params: List[nn.Parameter],
    compression_lr_multiplier: float,
):
    assert len(ref_params) == len(orig_params)
    assert len(target_params) == len(orig_params)

    ref_diff = get_params_diff(ref_params, orig_params)
    target_diff = get_params_diff(target_params, orig_params)

    if pytest.approx(compression_lr_multiplier) == 1:
        PTTensorListComparator.check_equal(target_diff, ref_diff)
    elif compression_lr_multiplier < 1:
        PTTensorListComparator.check_less(target_diff, ref_diff)
    else:
        PTTensorListComparator.check_greater(target_diff, ref_diff)


def test_if_setting_multipliers_in_config_affect_training_speed(
    get_ref_lenet_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    get_target_lenet_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    target_config: NNCFConfig,
):
    orig_params = get_lenet_params_after_training_steps(*get_ref_lenet_model_and_dataloader(), num_steps=0)
    target_params = get_lenet_params_after_training_steps(*get_target_lenet_model_and_dataloader(), num_steps=1)
    multipliers = get_multipliers_from_config(target_config)
    multipliers["regular"] = 1

    for algo, val in orig_params.items():
        check_if_zero_multiplier_freezes_training(val, target_params[algo], multipliers[algo])


@pytest.mark.parametrize(
    "one_parameter_model_creation_params", get_one_parameter_model_creation_params(for_training=True)
)
def test_if_setting_multiplier_in_parameter_affect_training_speed(
    get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
    one_parameter_model_creation_params: Dict,
):
    orig_params = get_one_parameter_model_params_after_training_steps(
        *get_ref_one_parameter_model_and_dataloader(), num_steps=0
    )
    ref_params = get_one_parameter_model_params_after_training_steps(
        *get_ref_one_parameter_model_and_dataloader(), num_steps=1
    )
    target_params = get_one_parameter_model_params_after_training_steps(
        *get_target_one_parameter_model_and_dataloader(), num_steps=1
    )

    assert target_params[0].requires_grad
    check_if_zero_multiplier_freezes_training(
        orig_params, target_params, one_parameter_model_creation_params["multiplier"]
    )
    check_params_affect_training_speed(
        orig_params, ref_params, target_params, one_parameter_model_creation_params["multiplier"]
    )
