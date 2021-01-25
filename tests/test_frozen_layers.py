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


def update_config(config, update_dict):
    config.update(update_dict)
    return config


class FrozenLayersTestStruct:

    def __init__(self, id='No_name', config_creator=get_quantization_config_without_range_init, config_update=None,
                 model_creator=TwoConvTestModel, is_error=True):
        if config_update is None:
            config_update = {}
        self.id = id
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
    FrozenLayersTestStruct(id='8_bits_quantization_with_ignored_scope', config_update={
        "compression": {
            "algorithm": "quantization"
        },
        "ignored_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    }
                           , is_error=False),
    FrozenLayersTestStruct(id='mixed_precision_quantization_with_ignored_scope', config_update={
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
    FrozenLayersTestStruct(id='mixed_precision_quantization_with_target_scope', config_update={
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
    FrozenLayersTestStruct(id='8_bits_quantization',
                           is_error=False),
    FrozenLayersTestStruct(id='', config_update={
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
    FrozenLayersTestStruct(id='4_bits_quantization', config_update={
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
    FrozenLayersTestStruct(id='const_sparsity', config_update={
        "compression": {
            "algorithm": "const_sparsity"
        }
    }, is_error=False),
    FrozenLayersTestStruct(id='rb_sparsity_8_bits_quantization', config_update={
        "compression": [{
            "algorithm": "rb_sparsity"
        },
            {
                "algorithm": "quantization"
            }
        ]
    }, is_error=True),
    FrozenLayersTestStruct(id='const_sparsity_8_bits_quantization', config_update={
        "compression": [{
            "algorithm": "const_sparsity"
        },
            {
                "algorithm": "quantization"
            }
        ]
    }, is_error=False),
    FrozenLayersTestStruct(id='', config_update={
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
    FrozenLayersTestStruct(id='', config_update={
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
    FrozenLayersTestStruct(id='', config_creator=get_basic_pruning_config, config_update={
        "compression":
            {
                "algorithm": "filter_pruning",
                "params": {
                    "prune_first_conv": True,
                    "prune_last_conv": True
                }
            }
    }, is_error=True),
    FrozenLayersTestStruct(id='', config_creator=get_basic_sparsity_config,
                           is_error=True)
]


@pytest.mark.parametrize('params', TEST_PARAMS, ids=[p.id + '_is_error_' + str(p.is_error) for p in TEST_PARAMS])
def test_frozen_layers(mocker, params):
    model = params.create_model()
    config = params.create_config()
    mocker.patch('nncf.quantization.algo.QuantizationBuilder._parse_init_params')
    ignored_scopes = config.get('ignored_scopes', [None])

    for scope in ignored_scopes:
        freeze_module(model, scope)

    if params.is_error:
        with pytest.raises(RuntimeError):
            compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    else:
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
