import pytest
from nncf.torch.tensor import PTNNCFTensorProcessor
import torch


@pytest.mark.parametrize('device', (torch.device('cpu'), torch.device('cuda')))
def test_create_tensor(device):
    if not torch.cuda.is_available():
        if device == torch.device('cuda'):
            pytest.skip('There are no available CUDA devices')
    shape = [1, 3, 10, 100]
    tensor = PTNNCFTensorProcessor.ones(shape, device)
    assert torch.is_tensor(tensor.tensor)
    assert tensor.device.type == device.type
    assert list(tensor.tensor.shape) == shape
