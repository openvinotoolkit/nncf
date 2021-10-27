import pytest
from nncf.torch.tensor import PTNNCFTensorProcessor
import torch


@pytest.mark.parametrize('device', (torch.device('cpu'), 'cuda'))
def test_create_tensor(device):
    if not torch.cuda.is_available():
        if device == 'cuda':
            pytest.skip('There are no available CUDA devices')
    shape = [1, 3, 10, 100]
    tensor = PTNNCFTensorProcessor.ones(shape, device)
    assert torch.is_tensor(tensor.tensor)
    if device == 'cuda':
        device = torch.device('cuda:0')
    assert tensor.device == device
    assert list(tensor.tensor.shape) == shape
