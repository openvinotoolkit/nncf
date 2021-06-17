from copy import deepcopy

from tests.helpers import TwoConvTestModel
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.helpers import create_mock_dataloader
from tests.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.quantization.test_quantization_helpers import create_rank_dataloader, post_compression_test_distr_init
from tests.quantization.test_quantization_helpers import distributed_init_test_default

import torch
from torch.optim import SGD
import pytest

KEY_TO_KD_PARAMETERS = 'kd'


class ComplexOutputsModel(torch.nn.Module):
    def __init__(self, input_size=[1, 1, 4, 4]):
        super().__init__()
        self.input_size = input_size
        self.Conv1 = torch.nn.Conv2d(in_channels=self.input_size[1], out_channels=1, kernel_size=3)
        self.Conv2_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.Conv2_2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, x):
        # first and seconds outputs with requires_grad=True
        # third output with requires_grad = False
        x = self.Conv1(x)
        output_1 = self.Conv2_1(x)
        with torch.no_grad():
            output_2 = self.Conv2_2(x)
        return x, output_1, output_2


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


def change_parameters_of_model(model, keyword='conv'):
    for name, module in model.named_modules():
        if keyword in name:
            module.weight = torch.nn.Parameter(module.weight / 2).to(module.weight.device)
            return


def fill_params_of_model_by_normal(model):
    for param in model.parameters():
        param.data = torch.normal(0, 1, size=param.data.size())


def create_sparsified_model_with_kd(model, input_size=[1, 1, 4, 4], sparsity_level=0.5):
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return model, compression_ctrl


@pytest.mark.parametrize("inference_type", ['cpu', 'single_GPU', 'DP'])
def test_kd_model_weights(inference_type):
    torch.manual_seed(1)
    input_size = [1, 1, 8, 8]
    if not torch.cuda.is_available() and not inference_type == 'cpu':
        return
    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if not inference_type == 'cpu':
        model.to(torch.device('cuda:0'))
        if inference_type == 'DP':
            model = torch.nn.DataParallel(model)
    model.train()
    optimizer = SGD(model.parameters(), lr=1e-02, weight_decay=1)
    number_of_iters = 100
    mock_dataloader = create_mock_dataloader(config, num_samples=torch.cuda.device_count() * number_of_iters,
                                             batch_size=torch.cuda.device_count())
    compression_ctrl.scheduler.epoch_step()
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(model.parameters()).device)
        model(input_)
        loss = compression_ctrl.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for param1, param2 in zip([param.to(torch.device('cpu')) for name, param in
                                   filter(lambda x: KEY_TO_KD_PARAMETERS in x[0], model.named_parameters())],
                                  dumped_orig_model.parameters()):
            assert torch.allclose(param1, param2)
            break


def test_loss_outputs_parsing():
    mse = torch.nn.MSELoss()
    input_size = [1, 1, 8, 8]
    model = ComplexOutputsModel(input_size)
    fill_params_of_model_by_normal(model)
    dumped_orig_model = deepcopy(model)
    sparsity_level = 0.3
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    model.train()
    mock_dataloader = create_mock_dataloader(config, num_samples=torch.cuda.device_count(),
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

        inner_kd_mse = sum([mse(item[0], item[1]) for item in loss_outputs])
        loss = compression_ctrl.loss()
        assert torch.allclose(inner_kd_mse, loss)


@pytest.mark.parametrize("inference_type", ['cpu', 'single_GPU', 'DP'])
def test_training_process(inference_type):
    if not torch.cuda.is_available() and not inference_type == 'cpu':
        return

    torch.manual_seed(1)
    input_size = [1, 1, 8, 8]
    sparsity_level = 0.3
    mse = torch.nn.MSELoss()
    actual_model = TwoConvTestModel()
    fill_params_of_model_by_normal(actual_model)
    dumped_orig_model = deepcopy(actual_model)
    config = get_kd_config(
        get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level))
    config_without_kd = get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                               sparsity_level)
    ref_model, ref_compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(actual_model), config_without_kd)
    actual_model, actual_compression_ctrl = create_compressed_model_and_algo_for_test(actual_model, config)
    act_optimizer = SGD(actual_model.parameters(), lr=1e-02, weight_decay=1)
    ref_optimizer = SGD(ref_model.parameters(), lr=1e-02, weight_decay=1)
    number_of_iters = 100
    mock_dataloader = create_mock_dataloader(config, num_samples=torch.cuda.device_count() * number_of_iters,
                                             batch_size=torch.cuda.device_count())

    if not inference_type == 'cpu':
        actual_model.to(torch.device('cuda:0'))
        ref_model.to(torch.device('cuda:0'))
        dumped_orig_model.to(torch.device('cuda:0'))
        if inference_type == 'DP':
            actual_model = torch.nn.DataParallel(actual_model)
            ref_model = torch.nn.DataParallel(ref_model)
            dumped_orig_model = torch.nn.DataParallel(dumped_orig_model)

    actual_model.train()
    ref_model.train()
    dumped_orig_model.train()
    actual_compression_ctrl.scheduler.epoch_step()
    ref_compression_ctrl.scheduler.epoch_step()
    for i, (input_, target) in enumerate(mock_dataloader):
        input_ = input_.to(next(actual_model.parameters()).device)
        act_output = actual_model(input_)

        actual_loss = actual_compression_ctrl.loss()
        act_optimizer.zero_grad()
        actual_loss.backward()
        act_optimizer.step()

        ref_output = ref_model(input_)
        kd_outputs = dumped_orig_model(input_)
        ref_loss = mse(ref_output, kd_outputs)
        ref_optimizer.zero_grad()
        ref_loss.backward()
        ref_optimizer.step()

        assert torch.allclose(ref_output, act_output)


@pytest.mark.skip(reason="WIP")
def test_training_process_ddp():

    torch.manual_seed(1)
    input_size = [1, 1, 8, 8]
    sparsity_level = 0.3
    config = get_sparsity_config_with_sparsity_init(get_basic_magnitude_sparsity_config(input_sample_size=input_size),
                                                    sparsity_level)
    ngpus_per_node = torch.cuda.device_count()
    config.world_size = ngpus_per_node
    torch.multiprocessing.spawn(run_distributed_training,
                                nprocs=ngpus_per_node,
                                args=(ngpus_per_node, config),
                                join=True)


def run_distributed_training(gpu, ngpus_per_node, config):
    distributed_init_test_default(gpu, ngpus_per_node, config)
    config_without_kd = deepcopy(config)
    config = get_kd_config(config)
    mse = torch.nn.MSELoss()
    act_model = TwoConvTestModel()
    fill_params_of_model_by_normal(act_model)
    act_model = deepcopy(act_model)
    ref_model = deepcopy(act_model)
    orig_model = deepcopy(act_model)

    act_model, act_compression_ctrl = create_compressed_model_and_algo_for_test(act_model, config)
    ref_model, ref_compression_ctrl = create_compressed_model_and_algo_for_test(ref_model, config_without_kd)
    act_model = post_compression_test_distr_init(act_compression_ctrl, config, ngpus_per_node, act_model)
    ref_model = post_compression_test_distr_init(ref_compression_ctrl, config, ngpus_per_node, act_model)
    act_optimizer = SGD(act_model.parameters(), lr=1e-02, weight_decay=1)
    ref_optimizer = SGD(ref_model.parameters(), lr=1e-02, weight_decay=1)
    number_of_iters = 10
    mock_dataloader = create_rank_dataloader(config, gpu)
    orig_model.to(next(act_model.parameters()).device)
    act_model.train()
    ref_model.train()
    orig_model.train()
    for i, (input_, target) in enumerate(mock_dataloader):
        print(f'iter {i}')
        act_output = act_model(input_)
        ref_output = ref_model(input_)
        # check stuff happening with device
        kd_output = orig_model(deepcopy(input_).to((next(orig_model.parameters()).device)))

        act_loss = act_compression_ctrl.loss()
        ref_loss = mse(ref_output, kd_output)

        act_optimizer.zero_grad()
        ref_optimizer.zero_grad()
        ref_loss.backward()
        act_loss.backward()
        act_optimizer.step()
        ref_optimizer.step()


@pytest.mark.skip(reason="WIP")
def test_kd_outputs_contrainers_parsing():
    pass


@pytest.mark.skip(reason="WIP")
@pytest.marl.pametrize('kdloss_type', ['mse', 'softmax'])
def test_kdloss_types(kdloss_type):
    pass
