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
import json

import os
import pytest
import torch

from examples.torch.common.distributed import configure_distributed
from examples.torch.common.execution import ExecutionMode, prepare_model_for_execution, get_device
from examples.torch.common.model_loader import load_model
from examples.torch.common.sample_config import SampleConfig
from nncf.torch import register_default_init_args
from nncf.torch.checkpoint_loading import load_state
from nncf.common.graph import MODEL_INPUT_OP_NAME
from nncf.config import NNCFConfig
from nncf.torch.nncf_network import LEGACY_ACT_STORAGE_NAME
from nncf.torch.nncf_network import MODEL_WRAPPED_BY_NNCF_ATTR_NAME
from tests.common.helpers import TEST_ROOT
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.quantization.test_range_init import SingleConv2dIdentityModel
from tests.torch.test_compressed_graph import get_basic_quantization_config
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import Command
from tests.common.helpers import get_cli_dict_args
from tests.torch.test_sanity_sample import create_command_line

GLOBAL_CONFIG = {
    TEST_ROOT.joinpath("torch", "data", "configs", "squeezenet1_1_cifar10_rb_sparsity_int8.json"): [
        {
            'checkpoint_name': 'squeezenet1_1_custom_cifar10_rb_sparsity_int8_dp.pth',
            'dataset': "cifar10",
            'execution_mode': ExecutionMode.GPU_DATAPARALLEL,
        },
        {
            'checkpoint_name': 'squeezenet1_1_custom_cifar10_rb_sparsity_int8_ddp.pth',
            'dataset': "cifar10",
            'execution_mode': ExecutionMode.MULTIPROCESSING_DISTRIBUTED,
        },
    ],
}

CONFIG_PARAMS = []
for config_path_, cases_list_ in GLOBAL_CONFIG.items():
    for case_params_ in cases_list_:
        CONFIG_PARAMS.append((config_path_, case_params_,))


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([str(p[0]), p[1]['execution_mode']]) for p in CONFIG_PARAMS])
def _params(request, backward_compat_models_path):
    if backward_compat_models_path is None:
        pytest.skip('Path to models weights for backward compatibility testing is not set,'
                    ' use --backward-compat-models option.')
    config_path, case_params = request.param
    checkpoint_path = str(os.path.join(backward_compat_models_path, case_params['checkpoint_name']))
    return {
        'sample_config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'execution_mode': case_params['execution_mode'],
        'dataset': case_params['dataset']
    }


def test_model_can_be_loaded_with_resume(_params):
    p = _params
    sample_config_path = p['sample_config_path']
    checkpoint_path = p['checkpoint_path']

    config = SampleConfig.from_json(str(sample_config_path))
    nncf_config = NNCFConfig.from_json(str(sample_config_path))

    config.execution_mode = p['execution_mode']

    config.current_gpu = 0
    config.device = get_device(config)
    config.distributed = config.execution_mode in (ExecutionMode.DISTRIBUTED, ExecutionMode.MULTIPROCESSING_DISTRIBUTED)
    if config.distributed:
        config.dist_url = "tcp://127.0.0.1:9898"
        config.dist_backend = "nccl"
        config.rank = 0
        config.world_size = 1
        configure_distributed(config)

    model_name = config['model']
    model = load_model(model_name,
                       pretrained=False,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'))
    nncf_config = register_default_init_args(nncf_config, train_loader=create_ones_mock_dataloader(nncf_config))

    model.to(config.device)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    model, _ = prepare_model_for_execution(model, config)

    if config.distributed:
        compression_ctrl.distributed()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(model, checkpoint['state_dict'], is_resume=True)


def test_loaded_model_evals_according_to_saved_acc(_params, tmp_path, dataset_dir):
    p = _params
    config_path = p['sample_config_path']
    checkpoint_path = p['checkpoint_path']

    metrics_path = str(tmp_path.joinpath('metrics.json'))
    tmp_path = str(tmp_path)
    args = {}
    if not dataset_dir:
        dataset_dir = tmp_path
    args['data'] = dataset_dir
    args['dataset'] = p['dataset']
    args['config'] = str(config_path)
    args['mode'] = 'test'
    args['log-dir'] = tmp_path
    args['workers'] = 0  # Workaroundr the PyTorch MultiProcessingDataLoader issue
    args['seed'] = 1
    args['resume'] = checkpoint_path
    args['metrics-dump'] = metrics_path

    if p['execution_mode'] == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        args['multiprocessing-distributed'] = ''
    else:
        pytest.skip("DataParallel eval takes too long for this test to be run during pre-commit")

    runner = Command(create_command_line(get_cli_dict_args(args), "classification"))
    runner.run()

    with open(metrics_path) as metric_file:
        metrics = json.load(metric_file)
        # accuracy is rounded to hundredths
        assert torch.load(checkpoint_path)['best_acc1'] == pytest.approx(metrics['Accuracy'], abs=1e-2)


def test_renamed_activation_quantizer_storage_in_state_dict():
    old_style_sd = {
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.weight': torch.ones([3, 3, 1, 1]),
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.bias': torch.ones([3]),
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.pre_ops.0.op._num_bits': 8 * torch.ones([1], dtype=torch.int32),
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.pre_ops.0.op.signed_tensor': torch.ones([1], dtype=torch.int32),
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.pre_ops.0.op.enabled': torch.ones([1], dtype=torch.int32),
        f'{MODEL_WRAPPED_BY_NNCF_ATTR_NAME}.conv2d.pre_ops.0.op.scale': torch.ones([3, 1, 1, 1]),
        f'{LEGACY_ACT_STORAGE_NAME}./{MODEL_INPUT_OP_NAME}_0|OUTPUT._num_bits': 8 * torch.ones([1], dtype=torch.int32),
        f'{LEGACY_ACT_STORAGE_NAME}./{MODEL_INPUT_OP_NAME}_0|OUTPUT.signed_tensor': torch.zeros([1], dtype=torch.int32),
        f'{LEGACY_ACT_STORAGE_NAME}./{MODEL_INPUT_OP_NAME}_0|OUTPUT.enabled': torch.ones([1], dtype=torch.int32),
        f'{LEGACY_ACT_STORAGE_NAME}./{MODEL_INPUT_OP_NAME}_0|OUTPUT.scale': torch.ones([1]),
    }
    model = SingleConv2dIdentityModel()
    config = get_basic_quantization_config(input_info={
        "sample_size": [1, 3, 100, 100]
    })
    register_bn_adaptation_init_args(config)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

    with pytest.deprecated_call():
        _ = load_state(compressed_model, old_style_sd, is_resume=True)
