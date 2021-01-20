from typing import Callable

import pytest
from torch import nn

from nncf import NNCFConfig
from tests.helpers import TwoConvTestModel, create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.pruning.helpers import get_basic_pruning_config
from tests.sparsity.rb.test_algo import get_basic_sparsity_config
from nncf.layers import NNCF_MODULES_MAP, NNCF_MODULES_DICT
from typing import NamedTuple
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


class FrozenLayersTestStruct(NamedTuple):
    config: NNCFConfig = get_quantization_config_without_range_init
    model_creator: Callable[[], nn.Module] = TwoConvTestModel
    is_error: bool = True


TEST_PARAMS = [
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": {
            "algorithm": "quantization"
        },
        "ignored_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    })
    , model_creator=TwoConvTestModel, is_error=False),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
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
    )
    , model_creator=TwoConvTestModel, is_error=False),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": {
            "algorithm": "quantization",
            "initializer": {
                "precision": {
                    "type": "manual"
                    }
                }
        },
        "target_scopes": ['TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]']
    })
    , model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=get_quantization_config_without_range_init(),
                           model_creator=TwoConvTestModel, is_error=False),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": {
        "algorithm": "quantization",
        "initializer": {
            "precision": {
                "type": "hawq",
                "bits": [4, 8],
                "compression_ratio": 1.5
                }
            }
        }
    })
    , model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
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
    }),
    model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": {
            "algorithm": "const_sparsity"
        }
    }), model_creator=TwoConvTestModel, is_error=False),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": [{
            "algorithm": "rb_sparsity"
        },
        {
            "algorithm": "quantization"
        }
        ]
    }), model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
        "compression": [{
            "algorithm": "const_sparsity"
        },
        {
            "algorithm": "quantization"
        }
        ]
    }), model_creator=TwoConvTestModel, is_error=False),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
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
    }), model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=update_config(get_quantization_config_without_range_init(), {
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
    }), model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=update_config(get_basic_pruning_config(), {
        "compression":
        {
            "algorithm": "filter_pruning",
            "params": {
                "prune_first_conv": True,
                "prune_last_conv": True
            }
        }
    }), model_creator=TwoConvTestModel, is_error=True),
    FrozenLayersTestStruct(config=get_basic_sparsity_config(),
                           model_creator=TwoConvTestModel, is_error=True)
]


@pytest.mark.parametrize('params', TEST_PARAMS, ids=[str(p.config['model']) for p in TEST_PARAMS])
def test_frozen_layers(capsys, params):
    model = params.model_creator()
    config = params.config

    ignored_scopes = config.get('ignored_scopes', [None])

    for scope in ignored_scopes:
        freeze_module(model, scope)

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
    if params.is_error:
        with pytest.raises(RuntimeError):
            compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)
