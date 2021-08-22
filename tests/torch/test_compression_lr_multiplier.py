"""
 Copyright (c) 2019-2020 Intel Corporation
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

from typing import Callable, Dict, Generator, List, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from nncf import NNCFConfig
from nncf.torch.layer_utils import CompressionParameter
from tests.common.test_compression_lr_multiplier import BaseCompressionLRMultiplierTester
from tests.common.test_compression_lr_multiplier import get_configs_building_params
from tests.common.test_compression_lr_multiplier import get_quantization_config
from tests.common.test_compression_lr_multiplier import get_rb_sparsity_config
from tests.common.test_compression_lr_multiplier import get_binarization_config
from tests.torch.helpers import create_initialized_compressed_model
from tests.torch.helpers import create_random_mock_dataloader
from tests.torch.helpers import get_grads
from tests.torch.helpers import LeNet
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import RandomDatasetMock
from tests.torch.helpers import set_torch_seed


def create_initialized_model_and_dataset(config: NNCFConfig) -> Tuple[nn.Module, DataLoader]:
    with set_torch_seed():
        train_loader = create_random_mock_dataloader(config, num_samples=10)
        model = LeNet()
        for param in model.parameters():
            nn.init.normal_(param)
        model = create_initialized_compressed_model(model, config, train_loader)
    return model, train_loader


@pytest.fixture(name='sample_size')
def sample_size_():
    return list((1,) + LeNet.INPUT_SIZE)


@pytest.fixture(name='get_ref_model_and_dataset')
def get_ref_model_and_dataset_(ref_config: NNCFConfig) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_model_and_dataset(ref_config)
    return f


@pytest.fixture(name='get_target_model_and_dataset')
def get_target_model_and_dataset_(target_config: NNCFConfig) -> Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_model_and_dataset(target_config)
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
            [], [('attr', False)], [('attr', True)], [('fn', False)], [('fn', True)],
            [('attr', not init_requires_grad), ('attr', True)], [('fn', not init_requires_grad), ('fn', True)],
            [('attr', not init_requires_grad), ('fn', True)], [('fn', not init_requires_grad), ('attr', True)]
        ]

        for requires_grad_settings in requires_grad_settings_list:
            trainable = init_requires_grad if len(requires_grad_settings) == 0 else requires_grad_settings[-1][1]
            if for_training and not trainable:
                continue
            multipliers = [0.1, 1, 10] if trainable else [0.1]

            for multiplier in multipliers:
                params.append({
                    'init_requires_grad': init_requires_grad,
                    'requires_grad_settings': requires_grad_settings,
                    'multiplier': multiplier
                })
    return params


def create_initialized_one_parameter_model_and_dataset(parameter_cls: type, init_requires_grad: bool,
                                                       requires_grad_settings: List[Tuple[str, bool]],
                                                       multiplier: float = None) -> [nn.Module, DataLoader]:
    with set_torch_seed():
        data = torch.randn(size=(1, 1, 5, 5))
        if parameter_cls is nn.Parameter:
            param = parameter_cls(data, requires_grad=init_requires_grad)
        elif parameter_cls is CompressionParameter:
            param = parameter_cls(data, requires_grad=init_requires_grad,
                                  compression_lr_multiplier=multiplier)
        else:
            raise Exception(f'Unsupported parameter type: {parameter_cls}')

    for setting_type, requires_grad in requires_grad_settings:
        if setting_type == 'attr':
            param.requires_grad = requires_grad
        elif setting_type == 'fn':
            param.requires_grad_(requires_grad)
        else:
            raise Exception(f'Unsupported setting type: {setting_type}')

    model = OneParameterModel(param)
    train_loader = DataLoader(RandomDatasetMock(model.INPUT_SIZE),
                              batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    return model, train_loader


@pytest.fixture(name='get_ref_one_parameter_model_and_dataloader')
def get_ref_one_parameter_model_and_dataloader_(one_parameter_model_creation_params: Dict) -> \
        Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_one_parameter_model_and_dataset(nn.Parameter,
                                                                  **one_parameter_model_creation_params)
    return f


@pytest.fixture(name='get_target_one_parameter_model_and_dataloader')
def get_target_one_parameter_model_and_dataloader_(one_parameter_model_creation_params: Dict) -> \
        Callable[[], Tuple[nn.Module, DataLoader]]:
    def f():
        return create_initialized_one_parameter_model_and_dataset(CompressionParameter,
                                                                  **one_parameter_model_creation_params)
    return f


@pytest.mark.usefixtures('ref_config')
class TestPTCompressionLRMultiplier(BaseCompressionLRMultiplierTester):
    ALGO_NAME_TO_PATH_MAP = {
        'quantization': 'nncf.torch.quantization',
        'rb_sparsity': 'nncf.torch.sparsity.rb',
        'binarization': 'nncf.torch.binarization'
    }

    TensorListComparator = PTTensorListComparator

    @pytest.fixture(name='configs_building_params',
                    params=get_configs_building_params([
                        get_quantization_config, get_rb_sparsity_config, get_binarization_config
                    ]))
    def configs_building_params_(self, request) -> Dict:
        return request.param

    @classmethod
    def _perform_model_training_steps(cls, model: nn.Module, dataset: DataLoader,
                                      num_steps: int = 1) -> nn.Module:
        with set_torch_seed():
            dataset = iter(dataset)
            optimizer = SGD(model.parameters(), lr=0.1)

            # This block of code is needed to initialize scale in the binarization algorithm
            # TODO: perform binarization scale init in the same way as for quantization
            with torch.no_grad():
                x, y_gt = next(dataset)
                model(x)

            for _ in range(num_steps):
                optimizer.zero_grad()
                x, y_gt = next(dataset)
                y = model(x)
                loss = F.mse_loss(y.sum(), y_gt)

                loss.backward()
                optimizer.step()

        return model

    @classmethod
    def _get_layer_cls_and_params(cls, model: nn.Module) -> Generator[Tuple[type, List[nn.Parameter]], None, None]:
        for module in model.modules():
            params = list(filter(lambda param: param.requires_grad, module.parameters(recurse=False)))
            yield module.__class__, params

    @classmethod
    def _get_grads(cls, params: Dict[str, List[nn.Parameter]]) -> Dict[str, List[torch.Tensor]]:
        return {k: get_grads(v) for k, v in params.items()}

    @classmethod
    def _get_params_and_grads_after_training_steps(cls, model: nn.Module, dataset: DataLoader,
                                                   num_steps: int = 1) -> Tuple[Dict[str, List[nn.Parameter]],
                                                                                Dict[str, List[torch.Tensor]]]:
        with set_torch_seed():
            model = cls._perform_model_training_steps(model, dataset, num_steps)
        params = cls._get_params_grouped_by_algos(model)
        grads = cls._get_grads(params)
        params = {algo: [param.cpu().detach() for param in params[algo]] for algo in params}
        return params, grads

    @pytest.mark.parametrize('one_parameter_model_creation_params',
                             get_one_parameter_model_creation_params())
    def test_compression_parameter_is_initialized_correctly(
            self,
            get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
            get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]]
    ):
        ref_model, _ref_loader = get_ref_one_parameter_model_and_dataloader()
        target_model, target_loader = get_target_one_parameter_model_and_dataloader()

        assert pytest.approx(ref_model.param.data) == target_model.param.data
        assert ref_model.param.requires_grad == target_model.param.requires_grad

        if ref_model.param.requires_grad:
            self._perform_model_training_steps(target_model, target_loader)
        else:
            with pytest.raises(Exception):
                self._perform_model_training_steps(target_model, target_loader)

    @pytest.mark.parametrize('one_parameter_model_creation_params',
                             get_one_parameter_model_creation_params(for_training=True))
    def test_multiplier_in_parameter_multiplies_grads(
            self,
            get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
            get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
            one_parameter_model_creation_params: Dict
    ):
        multipliers = {'regular': one_parameter_model_creation_params['multiplier']}
        self._check_multipliers_in_config_multiplies_grads(get_ref_one_parameter_model_and_dataloader,
                                                           get_target_one_parameter_model_and_dataloader,
                                                           multipliers)

    @pytest.mark.parametrize('one_parameter_model_creation_params',
                             get_one_parameter_model_creation_params(for_training=True))
    def test_multiplier_in_parameter_affect_training_speed(
            self,
            get_ref_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
            get_target_one_parameter_model_and_dataloader: Callable[[], Tuple[nn.Module, DataLoader]],
            one_parameter_model_creation_params: Dict,
    ):
        multipliers = {'regular': one_parameter_model_creation_params['multiplier']}
        self._check_multipliers_in_config_affect_training_speed(get_ref_one_parameter_model_and_dataloader,
                                                                get_target_one_parameter_model_and_dataloader,
                                                                multipliers)
