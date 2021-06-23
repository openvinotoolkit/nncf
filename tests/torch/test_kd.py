from copy import deepcopy
from functools import reduce

from tests.torch.test_models.synthetic import PartlyNonDifferentialOutputsModel, EmbeddingCatLinearModel
from tests.torch.test_models.synthetic import ContainersOutputsModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.torch.quantization.test_quantization_helpers import create_rank_dataloader, post_compression_test_distr_init
from tests.torch.quantization.test_quantization_helpers import distributed_init_test_default

import torch
from torch import nn
from torch.optim import SGD
import pytest

KEY_TO_KD_PARAMETERS = 'kd'


def get_kd_config(config):
    if isinstance(config['compression'], dict):
        config['compression'] = [config['compression']]
    config['compression'].append({
        'algorithm': 'knowledge_distillation',
        'type': 'mse'
    })
    return config


def get_sparsity_config_with_sparsity_init(config, sparsity_init=0.5):
    config['compression']['sparsity_init'] = sparsity_init
    config['compression']['params'] = {
        "schedule": "multistep",
        "multistep_steps": [
            2
        ],
        "multistep_sparsity_levels": [
            sparsity_init,
            sparsity_init + 0.5 * (1 - sparsity_init)
        ]
    }
    return config


def fill_params_of_model_by_normal(model, std=1.0):
    for param in model.parameters():
        param.data = torch.normal(0, std, size=param.data.size())


@pytest.mark.parametrize("inference_type", ['cpu', 'single_GPU', 'DP', 'DDP'])
def test_training_process(inference_type):
    if not torch.cuda.is_available() and not inference_type == 'cpu':
        return
    torch.manual_seed(1)
    input_size = [1, 1, 8, 8]
    sparsity_level = 0.3
    config = get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                                    sparsity_level)
    if inference_type == 'DDP':
        ngpus_per_node = torch.cuda.device_count()
        config.world_size = ngpus_per_node
        torch.multiprocessing.spawn(run_test_training,
                                    nprocs=ngpus_per_node,
                                    args=(config, inference_type, ngpus_per_node),
                                    join=True)
    else:
        run_test_training(None, config, inference_type, None)


def run_actual(model, config, inference_type, mock_dataloader, ddp_info=None):
    if inference_type == 'DDP':
        gpu, ngpus_per_node = ddp_info
    config = get_kd_config(config)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if inference_type == 'DDP':
        model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, model)
    elif inference_type == 'DP' or inference_type == 'single_GPU':
        model.to(torch.device('cuda:0'))
        if inference_type == 'DP':
            model = torch.nn.DataParallel(model)
    optimizer = SGD(model.parameters(), lr=1e-02, weight_decay=1e-02)
    model.train()
    output_storage = []
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        output = model(input_)
        output_storage.append(output)
        loss = compression_ctrl.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output_storage, model


def run_reference(model, config, inference_type, mock_dataloader, ddp_info=None):
    if inference_type == 'DDP':
        gpu, ngpus_per_node = ddp_info
    model = deepcopy(model)
    kd_model = deepcopy(model)
    mse = torch.nn.MSELoss().cuda()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if inference_type == 'DDP':
        model = post_compression_test_distr_init(compression_ctrl, config, ngpus_per_node, model)
        kd_model.to(torch.device(next(model.parameters()).device))
    elif inference_type == 'DP' or inference_type == 'single_GPU':
        model.to(torch.device('cuda:0'))
        kd_model.to(torch.device('cuda:0'))
        if inference_type == 'DP':
            model = torch.nn.DataParallel(model)
            kd_model = torch.nn.DataParallel(kd_model)
    optimizer = SGD(model.parameters(), lr=1e-02, weight_decay=1e-02)
    model.train()
    kd_model.train()
    output_storage = []
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        output = model(input_)
        kd_output = kd_model(input_)
        output_storage.append(output)
        loss = mse(output, kd_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output_storage


def run_test_training(gpu, config, inference_type, ngpus_per_node):
    torch.manual_seed(2)
    number_of_iters = 10
    batch_size = torch.cuda.device_count()
    config['input_info']['sample_size'] = [1, 1, 8, 8]
    if inference_type == 'DDP':
        distributed_init_test_default(gpu, ngpus_per_node, config)
        mock_dataloader = create_rank_dataloader(config, gpu, batch_size * number_of_iters, batch_size=batch_size)
    else:
        mock_dataloader = create_ones_mock_dataloader(config, num_samples=batch_size * number_of_iters,
                                                      batch_size=batch_size)

    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model, std=0.5)
    dumped_orig_model = deepcopy(model)

    actual_outputs, actual_model = run_actual(deepcopy(model), config, inference_type, mock_dataloader,
                                              (gpu, ngpus_per_node))
    reference_outputs = run_reference(model, config, inference_type, mock_dataloader, (gpu, ngpus_per_node))
    assert reduce(lambda a, b: a and torch.allclose(b[0], b[1]), zip(actual_outputs, reference_outputs), True), \
        "Outputs of model with actual KD implementation doesn't match outputs from model with reference " \
        "KD implementation"

    for param1, param2 in zip([param.to(torch.device('cpu')) for name, param in
                               filter(lambda x: KEY_TO_KD_PARAMETERS in x[0], actual_model.named_parameters())],
                              dumped_orig_model.parameters()):
        assert torch.allclose(param1, param2), "Weights of dumped original model doesn't match weights of original " \
                                               "model used for distillation (most likely weights of original model" \
                                               " are being corrupted due training)"


def test_loss_outputs_parsing():
    mse = torch.nn.MSELoss()
    input_size = [1, 1, 8, 8]
    model = PartlyNonDifferentialOutputsModel(input_size)
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(),
                                             batch_size=torch.cuda.device_count())
    compression_ctrl.scheduler.epoch_step()
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)

        loss_outputs = []
        for tensor1, tensor2 in zip(outputs, kd_outputs):
            if tensor1.requires_grad:
                loss_outputs.append((tensor1, tensor2))

        reference_kd_loss = sum([mse(item[0], item[1]) for item in loss_outputs])
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)


def test_kd_outputs_contrainers_parsing():
    mse = torch.nn.MSELoss()
    input_size = [1, 1, 8, 8]
    model = ContainersOutputsModel(input_size)
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(),
                                             batch_size=torch.cuda.device_count())
    compression_ctrl.scheduler.epoch_step()
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)

        reference_kd_loss = mse(outputs['xa'], kd_outputs['xa']) + \
                            mse(outputs['xb_and_xc'][0], kd_outputs['xb_and_xc'][0]) + \
                            mse(outputs['xb_and_xc'][1], kd_outputs['xb_and_xc'][1])
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)


@pytest.mark.parametrize('kd_loss_type', ['mse', 'softmax'])
def test_kd_loss_types(kd_loss_type):
    torch.manual_seed(2)
    if kd_loss_type == 'softmax':
        def kd_loss_fn(ref_outputs, compressed_model_outputs):
            return -(nn.functional.log_softmax(compressed_model_outputs, dim=1) *
                     nn.functional.softmax(ref_outputs, dim=1)).mean() * (compressed_model_outputs.shape[1])
    else:
        kd_loss_fn = torch.nn.MSELoss()
    input_size = [1, 100]

    model = nn.Sequential(nn.Linear(in_features=input_size[-1], out_features=10),
                          nn.Sigmoid())

    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.5
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    config['compression'][-1]['type'] = kd_loss_type
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_ones_mock_dataloader(config, num_samples=torch.cuda.device_count(),
                                             batch_size=torch.cuda.device_count())
    compression_ctrl.scheduler.epoch_step()
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        outputs = model(input_)
        kd_outputs = dumped_orig_model(input_)

        reference_kd_loss = kd_loss_fn(kd_outputs, outputs)
        actual_kd_loss = compression_ctrl.loss()
        assert torch.allclose(reference_kd_loss, actual_kd_loss)
