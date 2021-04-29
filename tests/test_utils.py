import pytest
from torch import nn

from nncf.utils import training_mode_switcher
from nncf.initialization import DataLoaderBNAdaptationRunner

from tests.helpers import BasicConvTestModel, TwoConvTestModel, MockModel
from tests.quantization.test_saturation_issue_export import DepthWiseConvTestModel, EightConvTestModel
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
