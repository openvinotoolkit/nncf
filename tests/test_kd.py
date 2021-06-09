from tests.helpers import BasicConvTestModel
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import create_mock_dataloader
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config

import torch
from torch.optim import SGD


def get_kd_config(config):
    if isinstance(config['compression'], dict):
        config['compression'] = [config['compression']]
    config['compression'].append({
        'algorithm': 'knowledge_distillation',
        'type': 'mse'
    })
    return config

def get_sparsity_config_with_sparsity_init(config):
    config['compression']['sparsity_init'] = 0.5
    config['compression']['params'] = {
        "schedule": "multistep",
        "multistep_steps": [
            2
        ],
        "multistep_sparsity_levels": [
            0.5,
            0.7
        ]
    }
    return config


def change_parameters_of_model(model, keyword='conv'):
    for name, module in model.named_modules():
        if keyword in name:
            module.weight = torch.nn.Parameter(module.weight / 2).to(module.weight.device)
            return


def test_dataparallel_convergence():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
    model_size = 4
    sparsity_level = 0.5
    model_kd_single_GPU = BasicConvTestModel()
    config_KD_single_GPU = get_kd_config(get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config()))
    model_kd_single_GPU, compression_ctrl = create_compressed_model_and_algo_for_test(model_kd_single_GPU, config_KD_single_GPU)

    change_parameters_of_model(model_kd_single_GPU)
    model_kd_single_GPU.to(torch.device('cuda:0'))

    model_kd_DP = BasicConvTestModel()
    config_KD_DP = get_kd_config(get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config()))
    model_kd_DP, compression_ctrl_with_kd = create_compressed_model_and_algo_for_test(model_kd_DP, config_KD_DP)

    change_parameters_of_model(model_kd_DP)
    model_kd_DP.to(torch.device('cuda:0'))
    model_kd_DP = torch.nn.DataParallel(model_kd_DP)

    number_of_iters = 100
    mock_dataloader = create_mock_dataloader(config_KD_single_GPU, num_samples=torch.cuda.device_count() * number_of_iters,
                                             batch_size=torch.cuda.device_count())
    model_kd_single_GPU.train()
    model_kd_DP.train()
    optimizer = SGD(model_kd_single_GPU.parameters(), lr=1e-04)
    optimizer_kd = SGD(model_kd_DP.parameters(), lr=1e-04)
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(torch.device('cuda:0'))
        output_single_GPU = model_kd_single_GPU(input_)
        output_DP = model_kd_DP(input_)

        print(f'outputs Single GPU {output_single_GPU}')
        print(f'outputs DP {output_DP}')

        comp_loss = compression_ctrl.loss()
        comp_loss_kd = compression_ctrl_with_kd.loss()

        optimizer.zero_grad()
        optimizer_kd.zero_grad()
        comp_loss.backward()
        comp_loss_kd.backward()
        print(f'Single GPU model parameters\n{[item.grad for item in model_kd_single_GPU.parameters()]}')
        print(f'DP model parameters\n{[item.grad for item in model_kd_DP.parameters()]}')

        optimizer.step()
        optimizer_kd.step()
