import pytest

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow import test_models


MODELS = [
    {'model': test_models.InceptionV3,
     'input_shape': (75, 75, 3)},
    {'model': test_models.InceptionResNetV2,
     'input_shape': (75, 75, 3)},
    {'model': test_models.NASNetMobile,
     'input_shape': (32, 32, 3)},
    {'model': test_models.DenseNet121,
     'input_shape': (32, 32, 3)},
]


@pytest.mark.parametrize('model,input_shape', [list(elem.values()) for elem in MODELS])
def test_concat(model, input_shape):
    config = get_basic_pruning_config(input_shape[1])
    model = model(list(input_shape))

    model, _ = create_compressed_model_and_algo_for_test(model, config)
