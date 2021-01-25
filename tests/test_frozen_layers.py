import pytest
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config
from tests.sparsity.rb.test_algo import get_basic_sparsity_config
from nncf.layers import NNCF_MODULES_MAP, NNCF_MODULES_DICT
from nncf.utils import get_module_by_node_name


def freeze_module(model, scope=None):
    if scope is not None:
        for nncf_module in NNCF_MODULES_MAP.keys():
            if nncf_module in scope:
                scope = scope.replace(nncf_module, NNCF_MODULES_MAP[nncf_module])
                break
        module = get_module_by_node_name(model, scope)
        module.weight.requires_grad = False
        return
    for module in model.modules():
        if module.__class__ in NNCF_MODULES_DICT.values():
            module.weight.requires_grad = False
            break


class FrozenLayersTestStruct:

    def __init__(self, name='No_name', config_creator=get_quantization_config_without_range_init, config_update=None,
                 model_creator=TwoConvTestModel, is_error=True):
        if config_update is None:
            config_update = {}
        self.name = name
        self.config_factory = config_creator
        self.config_update = config_update
        self.model_creator = model_creator
        self.is_error = is_error

    def create_config(self):
        config = self.config_factory()
        config.update(self.config_update)
        return config

    def create_model(self):
        model = self.model_creator()
        return model


TEST_PARAMS = [
    FrozenLayersTestStruct(name='8_bits_quantization_with_ignored_scope', config_update={
        "compression": {
            "algorithm": "quantization"
        },
        "ignored_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    }
                           , is_error=False),
    FrozenLayersTestStruct(name='mixed_precision_quantization_with_ignored_scope', config_update={
        "target_device": "VPU",
        "compression": {
            "algorithm": "quantization",
            "initializer": {
                "precision": {
                    "type": "manual"
                }
            }
        },
        "ignored_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    }
                           , is_error=False),
    FrozenLayersTestStruct(name='mixed_precision_quantization_with_target_scope', config_update={
        "target_device": "VPU",
        "compression": {
            "algorithm": "quantization",
            "initializer": {
                "precision": {
                    "type": "manual"
                }
            }
        },
        "target_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    }
                           , is_error=True),
    FrozenLayersTestStruct(name='8_bits_quantization',
                           is_error=False),
    FrozenLayersTestStruct(name='', config_update={
        "compression": {
            "algorithm": "quantization",
            "initializer": {
                "precision": {
                    "type": "manual"
                }
            }
        }
    }
                           , is_error=True),
    FrozenLayersTestStruct(name='4_bits_quantization', config_update={
        "target_device": "VPU",
        "compression": {
            "algorithm": "quantization",
            "weights": {
                "bits": 4
            },
            "activations": {
                "bits": 4
            }
        }
    },
                           is_error=True),
    FrozenLayersTestStruct(name='const_sparsity', config_update={
        "compression": {
            "algorithm": "const_sparsity"
        }
    }, is_error=False),
    FrozenLayersTestStruct(name='rb_sparsity_8_bits_quantization', config_update={
        "compression": [{
            "algorithm": "rb_sparsity"
        },
            {
                "algorithm": "quantization"
            }
        ]
    }, is_error=True),
    FrozenLayersTestStruct(name='const_sparsity_8_bits_quantization', config_update={
        "compression": [{
            "algorithm": "const_sparsity"
        },
            {
                "algorithm": "quantization"
            }
        ]
    }, is_error=False),
    FrozenLayersTestStruct(name='const_sparsity_4_bits_quantization', config_update={
        "target_device": "VPU",
        "compression": [{
            "algorithm": "const_sparsity"
        },
            {
                "algorithm": "quantization",
                "weights": {
                    "bits": 4
                },
                "activations": {
                    "bits": 4
                },
            }
        ]
    }, is_error=True),
    FrozenLayersTestStruct(name='rb_sparsity_4_bits_quantization', config_update={
        "target_device": "VPU",
        "compression": [{
            "algorithm": "rb_sparsity"
        },
            {
                "algorithm": "quantization",
                "weights": {
                    "bits": 4
                },
                "activations": {
                    "bits": 4
                },
            }
        ]
    }, is_error=True),
    FrozenLayersTestStruct(name='filter_pruning', config_creator=get_basic_pruning_config, config_update={
        "compression":
            {
                "algorithm": "filter_pruning",
                "params": {
                    "prune_first_conv": True,
                    "prune_last_conv": True
                }
            }
    }, is_error=True),
    FrozenLayersTestStruct(name='sparsity', config_creator=get_basic_sparsity_config,
                           is_error=True),
    FrozenLayersTestStruct(name='binarization', is_error=True, config_update={
        "compression": {
                "algorithm": "binarization"
            }
    })
]


@pytest.mark.parametrize('params', TEST_PARAMS, ids=[p.name + '_is_error_' + str(p.is_error) for p in TEST_PARAMS])
def test_frozen_layers(caplog, mocker, params):
    model = params.create_model()
    config = params.create_config()
    mocker.patch('nncf.quantization.algo.QuantizationBuilder._parse_init_params')
    ignored_scopes = config.get('ignored_scopes', [None])
    for scope in ignored_scopes:
        freeze_module(model, scope)

    if params.is_error:
        with pytest.raises(RuntimeError):
            __, _ = create_compressed_model_and_algo_for_test(model, config)
    else:
        __, _ = create_compressed_model_and_algo_for_test(model, config)
        if ignored_scopes[0] is None:
            assert 'Frozen layers' in caplog.text
