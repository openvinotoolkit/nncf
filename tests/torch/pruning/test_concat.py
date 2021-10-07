import numpy as np
import pytest

from functools import partial

from nncf.common.pruning.schedulers import BaselinePruningScheduler, ExponentialWithBiasPruningScheduler
from tests.torch.pruning.helpers import get_pruning_baseline_config, PruningTestModel, get_pruning_exponential_config
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch import test_models
from tests.torch.test_models.synthetic import EmbeddingCatLinearModel
from tests.torch.test_models.googlenet import GoogLeNet
from tests.torch.test_models.sr_small_model import SmallModel


MODELS = [
             {'model': EmbeddingCatLinearModel,
              'input_shape': (1, 10)},
             {'model': test_models.densenet_cifar,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.DPN26,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.DPN92,
              'input_shape': (1, 3, 32, 32)},
             {'model': GoogLeNet,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.inception_v3,
              'input_shape': (1, 3, 229, 229)},
             {'model': test_models.PNASNetA,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.PNASNetB,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.ShuffleNetG3,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.ShuffleNetG2,
              'input_shape': (1, 3, 32, 32)},
             {'model': partial(test_models.ShuffleNetV2, net_size=1),
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.squeezenet1_0,
              'input_shape': (1, 3, 32, 32)},
             {'model': test_models.squeezenet1_1,
              'input_shape': (1, 3, 32, 32)},
             {'model': SmallModel,
              'input_shape': ()},
             {'model': test_models.UNet,
              'input_shape': (1, 3, 360, 480)},
]

SKIP_LIST = [SmallModel, EmbeddingCatLinearModel]


@pytest.mark.parametrize('model,input_shape', [list(elem.values()) for elem in MODELS])
def test_models_with_concat(model, input_shape):
    if model in SKIP_LIST:
        pytest.skip()

    config = get_pruning_baseline_config(list(input_shape))
    config['compression']['algorithm'] = 'filter_pruning'
    model = model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
