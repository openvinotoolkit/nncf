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
import pytest
import re
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
from pytest import approx
from torch.utils.data import DataLoader

from examples.common.models.classification import squeezenet1_1_custom
from nncf import utils
from nncf.checkpoint_loading import load_state
from nncf.config import NNCFConfig
from nncf.initialization import InitializingDataLoader, register_default_init_args
from nncf.quantization.layers import SymmetricQuantizer, AsymmetricQuantizer, \
    BaseQuantizer
from nncf.structures import QuantizationRangeInitArgs
from nncf.utils import get_all_modules_by_type, safe_thread_call
from tests.quantization.test_precision_init import create_hawq_test_config
from tests.quantization.test_quantization_helpers import compare_multi_gpu_dump, get_squeezenet_quantization_config, \
    distributed_init_test_default, post_compression_test_distr_init
from tests.helpers import TwoConvTestModel, get_empty_config, \
    create_compressed_model_and_algo_for_test, create_mock_dataloader, MockModel


def scale_signed_dumping_worker(gpu, ngpus_per_node, config, tmp_path):
    data_loader = distributed_init_test_default(gpu, ngpus_per_node, config)
    model = safe_thread_call(partial(squeezenet1_1_custom, pretrained=True))

    config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_scheduler = compression_ctrl.scheduler

    quant_model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, quant_model)

    criterion = torch.nn.MSELoss().cuda(config.gpu)
    optimizer = torch.optim.Adam(quant_model.parameters(), lr=0.01)

    torch.backends.cudnn.benchmark = True

    # just to reproduce the same scale values without Dropout
    quant_model.eval()

    act_sum = 0
    for layer in get_all_modules_by_type(quant_model, "SymmetricQuantizer").values():
        act_sum += layer.scale
    ref_sum = 3467.322
    assert act_sum.item() == approx(ref_sum, 0.01), \
        'sum of scales is not expected {} vs {} rank {}'.format(act_sum.item(), ref_sum, config.rank)

    out_file_path = get_path_after_broadcast(tmp_path, config.rank)
    save_params(quant_model, out_file_path)
    compression_scheduler.step()
    for i, (input_, _) in enumerate(data_loader):
        if i > 5:
            break
        output = quant_model(input_)
        optimizer.zero_grad()
        dummy_target = torch.randn(1000).cuda(config.gpu, non_blocking=True)
        loss = criterion(output, dummy_target)
        compression_scheduler.step()
        loss.backward()
        optimizer.step()
        compression_scheduler.step()

    out_file_path = get_path_path_after_train_iters(tmp_path, config.rank)
    save_params(quant_model, out_file_path)


def get_path_path_after_train_iters(tmp_path, rank):
    out_file_path = tmp_path / 'scale_signed_after_1_train_iter_gpu{}.pt'.format(rank)
    return out_file_path


def get_path_after_broadcast(tmp_path, rank):
    out_file_path = tmp_path / 'scale_signed_after_broadcast_gpu{}.pt'.format(rank)
    return out_file_path


def save_params(model, out_file_path):
    gpu_scale_signed_params = []
    for _, layer in utils.get_all_modules_by_type(model, 'SymmetricQuantizer').items():
        gpu_scale_signed_params.append((layer.scale.to(torch.device('cpu')),
                                        layer.signed_tensor.to(torch.device('cpu'))))
    with out_file_path.open('wb') as out_file:
        torch.save(gpu_scale_signed_params, out_file)


def test_multiprocessing_distributed_shares_init_scales_signedness_across_gpus(tmp_path):
    num_init_steps = 10

    config = get_squeezenet_quantization_config()
    config['compression']['initializer'] = {'range': {'num_init_steps': num_init_steps}}

    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(scale_signed_dumping_worker,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config, tmp_path),
                                join=True)

    assert not compare_multi_gpu_dump(config, tmp_path, get_path_after_broadcast)
    assert not compare_multi_gpu_dump(config, tmp_path, get_path_path_after_train_iters)


def create_empty_config_without_init_section():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'quantization'}
    return config


def create_config():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'quantization', 'initializer': {'range': {'num_init_steps': 1}}}
    return config


@pytest.mark.parametrize("wrap_dataloader",
                         (True, False),
                         ids=['wrapped_dataloader', 'standard_dataloader'])
class TestRangeInit:
    @staticmethod
    def create_algo_and_compressed_model(config):
        model = TwoConvTestModel()
        compressed_model, algo = create_compressed_model_and_algo_for_test(model, config)
        return algo, compressed_model

    @staticmethod
    def create_dataloader(wrap_dataloader, config, device='cpu') -> DataLoader:
        data_loader = create_mock_dataloader(config)
        if wrap_dataloader:
            data_loader = InitializingDataLoader(data_loader=data_loader,
                                                 device=device,
                                                 kwargs={})
        return data_loader

    @staticmethod
    def check_sign_and_scale(model, ref_table):
        model_conv = get_all_modules_by_type(model, 'SymmetricQuantizer')
        for scope, module in model_conv.items():
            for pattern, ref_values in ref_table.items():
                match = re.search(pattern, str(scope))
                if match:
                    assert isinstance(module, SymmetricQuantizer)
                    assert module.signed == ref_values[0], 'sign is not matched for {}'.format(str(scope))
                    assert module.scale == ref_values[1], 'scale is not matched for {}'.format(str(scope))

    @pytest.mark.parametrize("config_creator", (create_config, create_empty_config_without_init_section))
    def test_scale_and_sign_init_for_quant_algo__without_init_section(self, wrap_dataloader, config_creator):
        config = config_creator()
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (True, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scale_and_sign_init_for_quant_algo__with_zero_init_steps(self, wrap_dataloader):
        config = create_config()
        config['compression']['initializer']['range']['num_init_steps'] = 0

        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (False, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 1),
            '.*activation_quantizers.*Sequential\\[0\\].*': (False, 1),
            '.*activation_quantizers.*Sequential\\[1\\].*': (False, 1)
        })

    def test_scale_and_sign_init_for_quant_algo__after_load_state(self, wrap_dataloader):
        config = create_config()
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)
        load_state(compressed_model, {
            'module.features.0.0.pre_ops.0.op.signed_tensor': torch.tensor([0.]),  # quantizer of 1st conv's weights
            'module.features.1.0.pre_ops.0.op.scale': torch.tensor([100])  # quantizer of 2nd conv's weights
        })

        self.check_sign_and_scale(compressed_model, {
            '.*Sequential\\[0\\].*UpdateWeight.*': (False, 1),
            '.*Sequential\\[1\\].*UpdateWeight. *': (False, 100),
            '.*activation_quantizers.*Sequential\\[0\\].*': (True, 4),
            '.*activation_quantizers.*Sequential\\[1\\].*': (True, 24)
        })

    def test_scope_overrides(self, wrap_dataloader):
        config = create_config()
        config["compression"]["scope_overrides"] = {
            r"{re}NNCFConv2d\[[0-9]*\]$": {
                "bits": 7,
                "mode": "asymmetric",
            },
            r"{re}NNCFConv2d\[[0-9]*\]/conv2d_0": {
                "bits": 7,
                "signed": False,
            }
        }
        data_loader = self.create_dataloader(wrap_dataloader, config)
        config.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
        _, compressed_model = self.create_algo_and_compressed_model(config)

        quantizers = get_all_modules_by_type(compressed_model, ['SymmetricQuantizer',
                                                                'AsymmetricQuantizer'])
        quantizer_str_dict = {str(k): v for k, v in quantizers.items()}
        group_1 = [quantizer_str_dict["NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/"
                                      "Sequential[0]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/"
                                      "AsymmetricQuantizer[op]"],
                   quantizer_str_dict["NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/"
                                      "Sequential[0]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateInputs[1]/"
                                      "AsymmetricQuantizer[op]"],
                   quantizer_str_dict['NNCFNetwork/TwoConvTestModel[nncf_module]/Sequential[features]/'
                                      'Sequential[1]/NNCFConv2d[0]/ModuleDict[pre_ops]/UpdateWeight[0]/'
                                      'AsymmetricQuantizer[op]']
                   ]
        group_2 = [quantizer_str_dict['NNCFNetwork/ModuleDict[activation_quantizers]/'
                                      'SymmetricQuantizer[TwoConvTestModel/Sequential[features]'
                                      '/Sequential[0]/NNCFConv2d[0]/conv2d_0]']]

        for quantizer in group_1:
            assert isinstance(quantizer, AsymmetricQuantizer)
            assert quantizer.levels == 2 ** 7
        for quantizer in group_2:
            assert isinstance(quantizer, SymmetricQuantizer)
            assert not quantizer.signed


class SingleConv2dIdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, 1)
        self.conv2d.weight = torch.nn.Parameter(torch.ones_like(self.conv2d.weight))

    def forward(self, input_):
        return self.conv2d(input_)


class SingleConv2dSyntheticWeightModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, 100)

        for i in range(0, 100):
            for j in range(0, 100):
                self.conv2d.weight[0][0][i][j] = i * 100 + j

    def forward(self, input_):
        return self.conv2d(input_)


@pytest.mark.parametrize("quantization_mode",
                         ["symmetric", "asymmetric"])
def test_percentile_init(quantization_mode):
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self):
            self._length = 1

        def __getitem__(self, idx):
            if idx >= self._length:
                raise StopIteration
            test_input_sample = torch.zeros([1, 100, 100])
            for i in range(0, 100):
                for j in range(0, 100):
                    test_input_sample[0][i][j] = i * 100 + j
            return test_input_sample, test_input_sample

        def __len__(self):
            return self._length

    data_loader = torch.utils.data.DataLoader(SyntheticDataset(), batch_size=1)

    config_with_init = NNCFConfig()
    config_with_init.update(
        {
            "input_info": {
                "sample_size": [1, 1, 100, 100]
            },
            "compression": {
                "algorithm": "quantization",
                "activations": {
                    "mode": quantization_mode,
                },
                "weights": {
                    "mode": quantization_mode,
                },
                "initializer": {
                    "range": {
                        "num_init_steps": 1,
                        "type": "percentile",
                        "min_percentile": 32.10,
                        "max_percentile": 67.89
                    }
                }
            }
        }
    )

    # Activations init check
    id_model = SingleConv2dIdentityModel()
    config_with_init.register_extra_structs([QuantizationRangeInitArgs(data_loader)])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(id_model, config_with_init)

    act_quantizer = next(iter(compression_ctrl.non_weight_quantizers.values()))

    def assert_range(quantizer: BaseQuantizer):
        # Absolute tolerance is 1.0 due to percentile value interpolation
        if quantization_mode == 'symmetric':
            assert quantizer.scale.item() == approx(6789, abs=1.0)
        else:
            assert quantizer.input_low.item() == approx(3210, abs=1.0)
            assert quantizer.input_range.item() == approx(3578, abs=1.0)

    assert_range(act_quantizer)
    # Weight init check
    synth_weight_model = SingleConv2dSyntheticWeightModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(synth_weight_model,
                                                                    config_with_init)

    weight_quantizer = next(iter(compression_ctrl.non_weight_quantizers.values()))
    assert_range(weight_quantizer)


@pytest.mark.parametrize(("config_cutter", "range_init_call_count", "precision_init_call_count"),
                         [
                             (lambda x: x['initializer'].pop('range'), 1, 1),
                             (lambda x: x.pop('initializer'), 1, 0),
                             (lambda x: x['initializer'].pop('precision'), 1, 0),
                             (lambda x: x['initializer']['range'].update({'num_init_steps': 0}), 0, 1),
                         ], ids=['precision_init_only', 'no_init_params', 'range_init_only', 'skip_range_init'])
def test_range_init_is_called(config_cutter, range_init_call_count, precision_init_call_count, mocker):
    config = create_hawq_test_config()
    model = MockModel()
    config = register_default_init_args(config, mocker.stub(), mocker.stub())
    range_init_spy = mocker.patch('nncf.quantization.algo.QuantizationController._do_range_init')
    precision_init_spy = mocker.patch('nncf.quantization.init_precision.HAWQPrecisionInitializer.apply_init')

    config_cutter(config['compression'])
    create_compressed_model_and_algo_for_test(model, config)

    assert range_init_spy.call_count == range_init_call_count
    assert precision_init_spy.call_count == precision_init_call_count
