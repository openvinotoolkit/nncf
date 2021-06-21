import pytest
import torch
from torch import nn

from nncf.torch.utils import training_mode_switcher
from nncf.torch.utils import save_module_state
from nncf.torch.utils import _ModuleState
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.initialization import DataLoaderBNAdaptationRunner

from tests.torch.helpers import BasicConvTestModel, TwoConvTestModel, MockModel
from tests.torch.quantization.test_saturation_issue_export import DepthWiseConvTestModel, EightConvTestModel
# pylint:disable=unused-import
from tests.torch.modules.test_rnn import _seed


def compare_saved_model_state_and_current_model_state(module: nn.Module, model_state: _ModuleState):
    for ch in module.modules():
        assert model_state.training_state[ch] == ch.training

    for p in module.parameters():
        assert p.requires_grad == model_state.requires_grad_state[p]


def randomly_change_model_state(module: nn.Module, compression_params_only: bool = False):
    import random
    for ch in module.modules():
        if random.uniform(0, 1) > 0.5:
            ch.training = False
        else:
            ch.training = True

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
    randomly_change_model_state(model)
    saved_state = save_module_state(model)
    with training_mode_switcher(model, True):
        # pylint: disable=unnecessary-pass
        pass

    compare_saved_model_state_and_current_model_state(model, saved_state)


@pytest.mark.parametrize('model', [BasicConvTestModel(), TwoConvTestModel(), MockModel(),
                                   DepthWiseConvTestModel(), EightConvTestModel()])
def test_bn_training_state_switcher(_seed, model: nn.Module):

    def check_were_only_bn_training_state_changed(module: nn.Module, saved_state: _ModuleState):
        for ch in module.modules():
            if isinstance(ch, (nn.BatchNorm1d,
                               nn.BatchNorm2d,
                               nn.BatchNorm3d)):
                assert ch.training
            else:
                assert ch.training == saved_state.training_state[ch]

    runner = DataLoaderBNAdaptationRunner(model, 'cuda', 0)

    randomly_change_model_state(model)
    saved_state = save_module_state(model)

    # pylint: disable=protected-access
    with runner._bn_training_state_switcher():
        check_were_only_bn_training_state_changed(model, saved_state)

    compare_saved_model_state_and_current_model_state(model, saved_state)
