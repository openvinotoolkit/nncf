import pytest
from nncf.tensorflow.pruning.tensor_processor import TFNNCFPruningTensorProcessor
import tensorflow as tf


@pytest.mark.parametrize('device', ("CPU", 'GPU'))
def test_create_tensor(device):
    if not tf.config.list_physical_devices('GPU'):
        if device == 'GPU':
            pytest.skip('There are no available CUDA devices')
    shape = [1, 3, 10, 100]
    tensor = TFNNCFPruningTensorProcessor.ones(shape, device)
    assert tf.is_tensor(tensor.tensor)
    assert tensor.tensor.device.split('/')[-1].split(':')[1] == device
    assert list(tensor.tensor.shape) == shape
