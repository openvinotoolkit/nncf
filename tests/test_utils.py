import pytest
from collections import namedtuple
from typing import NamedTuple
import torch
from torch import nn

from nncf.utils import training_mode_switcher
from nncf.layer_utils import CompressionParameter
from nncf.initialization import DataLoaderBNAdaptationRunner

from tests.helpers import BasicConvTestModel, TwoConvTestModel, MockModel
from tests.quantization.test_saturation_issue_export import DepthWiseConvTestModel, EightConvTestModel
# pylint:disable=unused-import
from tests.modules.test_rnn import _seed


def save_model_state(module: nn.Module, model_state: NamedTuple):
    for ch in module.modules():
        model_state.training_state[ch] = ch.training

    for p in module.parameters():
        model_state.requires_grad_state[p] = p.requires_grad


def compare_saved_model_state_and_current_model_state(module: nn.Module, model_state: NamedTuple):
    for ch in module.modules():
        assert model_state.training_state[ch] == ch.training

    for p in module.parameters():
        assert p.requires_grad == model_state.requires_grad_state[p]


def randomly_change_model_training_state(module: nn.Module):
    import random
    for ch in module.modules():
        if random.uniform(0, 1) > 0.5:
            ch.training = False
        else:
            ch.training = True


def randomly_change_model_requires_grad_state(module: nn.Module, compression_params_only: bool = False):
    import random
    for p in module.parameters():
        if compression_params_only and not (isinstance(p, CompressionParameter) and torch.is_floating_point(p)):
            break
        if random.uniform(0, 1) > 0.5:
            p.requires_grad = False
        else:
            p.requires_grad = True


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(), MockModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_training_mode_switcher(_seed, model: nn.Module):
    State = namedtuple('State', ['training_state', 'requires_grad_state'])
    training_state, requires_grad_state = {}, {}
    saved_state = State(training_state, requires_grad_state)

    randomly_change_model_training_state(model)
    randomly_change_model_requires_grad_state(model)

    save_model_state(model, saved_state)
    with training_mode_switcher(model, True):
        # pylint: disable=unnecessary-pass
        pass

    compare_saved_model_state_and_current_model_state(model, saved_state)


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(), MockModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_bn_training_state_switcher(_seed, model: nn.Module):
    runner = DataLoaderBNAdaptationRunner(model, 'cuda', 0)
    State = namedtuple('State', ['training_state', 'requires_grad_state'])
    training_state, requires_grad_state = {}, {}
    saved_state = State(training_state, requires_grad_state)

    def check_were_only_bn_training_state_changed(module: nn.Module, saved_state: NamedTuple):
        for ch in module.modules():
            if isinstance(ch, (nn.BatchNorm1d,
                               nn.BatchNorm2d,
                               nn.BatchNorm3d)):
                assert ch.training
            else:
                assert ch.training == saved_state.training_state[ch]

    randomly_change_model_training_state(model)
    randomly_change_model_requires_grad_state(model)

    save_model_state(model, saved_state)

    # pylint: disable=protected-access
    with runner._bn_training_state_switcher():
        check_were_only_bn_training_state_changed(model, saved_state)

    compare_saved_model_state_and_current_model_state(model, saved_state)
