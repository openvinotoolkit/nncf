import pytest
import torch
from torch import nn

from nncf.utils import training_mode_switcher
from nncf.utils import set_compression_parameters_requires_grad_true
from nncf.layer_utils import CompressionParameter
from nncf.initialization import DataLoaderBNAdaptationRunner

from tests.helpers import BasicConvTestModel, TwoConvTestModel, MockModel
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.quantization.test_saturation_issue_export import DepthWiseConvTestModel, EightConvTestModel
from tests.quantization.test_onnx_export import get_config_for_export_mode
# pylint:disable=unused-import
from tests.modules.test_rnn import _seed


def save_model_training_state(module, model_state):
    for ch in module.children():
        model_state[ch] = ch.training
        save_model_training_state(ch, model_state)


def compare_saved_model_state_and_current_model_state(module, model_state):
    for ch in module.children():
        assert model_state[ch] == ch.training
        compare_saved_model_state_and_current_model_state(ch, model_state)


def randomly_change_model_training_state(module):
    import random
    for ch in module.children():
        if random.uniform(0, 1) > 0.5:
            ch.training = False
        else:
            ch.training = True
        randomly_change_model_training_state(ch)


def randomly_change_NNCF_parameters_requires_grad(module):
    import random
    for p in module.parameters():
        if isinstance(p, CompressionParameter):
            if torch.is_floating_point(p):
                if random.uniform(0, 1) > 0.5:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(), MockModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_training_mode_switcher(_seed, model):
    randomly_change_model_training_state(model)

    saved_model_state = {}
    save_model_training_state(model, saved_model_state)

    with training_mode_switcher(model, True):
        # pylint: disable=unnecessary-pass
        pass

    compare_saved_model_state_and_current_model_state(model, saved_model_state)


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(), MockModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_bn_training_state_switcher(_seed, model):
    runner = DataLoaderBNAdaptationRunner(model, 'cuda', 0)
    saved_model_state = {}

    def check_were_only_bn_training_state_changed(module, saved_state):
        for ch in module.children():
            if isinstance(ch, (nn.BatchNorm1d,
                               nn.BatchNorm2d,
                               nn.BatchNorm3d)):
                assert ch.training
            else:
                assert ch.training == saved_state[ch]
            check_were_only_bn_training_state_changed(ch, saved_state)

    randomly_change_model_training_state(model)

    save_model_training_state(model, saved_model_state)

    # pylint: disable=protected-access
    with runner._bn_training_state_switcher():
        check_were_only_bn_training_state_changed(model, saved_model_state)

    compare_saved_model_state_and_current_model_state(model, saved_model_state)


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_set_compression_parameters_requires_grad_true(_seed, model):
    def check_NNCF_parameters_requires_grad(module):
        for p in module.parameters():
            if isinstance(p, CompressionParameter):
                if torch.is_floating_point(p):
                    assert p.requires_grad is True

    nncf_config = get_config_for_export_mode(True)
    nncf_config.update({"input_info": {
        "sample_size": [1, 1, 20, 20]
    }})
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    randomly_change_NNCF_parameters_requires_grad(compressed_model)

    set_compression_parameters_requires_grad_true(compressed_model)

    check_NNCF_parameters_requires_grad(compressed_model)
