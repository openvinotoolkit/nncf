# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import nncf
from nncf import NNCFConfig
from nncf.common.logging import nncf_logger
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.utils import get_all_modules_by_type
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import get_empty_config
from tests.torch.helpers import register_bn_adaptation_init_args

FIRST_NNCF_CONV_SCOPE = "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0"
FIRST_CONV_SCOPE = "TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]/conv2d_0"


class AlgoBuilder:
    def __init__(self):
        self._config = {}

    def name(self, algo_name):
        self._config["algorithm"] = algo_name
        return self

    def int4(self):
        self._config.update({"weights": {"bits": 4}, "activations": {"bits": 4}})
        return self

    def mixed_precision(self):
        self._config.update({"initializer": {"precision": {"type": "manual"}}})
        return self

    def pruning(self):
        self._config.update(
            {
                "algorithm": "filter_pruning",
                "params": {
                    "prune_first_conv": True,
                },
            }
        )
        return self

    def ignore_first_conv(self):
        self._config["ignored_scopes"] = FIRST_NNCF_CONV_SCOPE
        return self

    def target_first_conv(self):
        self._config["target_scopes"] = FIRST_NNCF_CONV_SCOPE
        return self

    def get_config(self):
        return self._config


class FrozenLayersTestStruct:
    def __init__(self, name="No_name"):
        self.name = name
        self.config_update = {"target_device": "NPU", "compression": []}
        self.raising_error = False
        self.printing_warning = False
        self._freeze_all = False
        self._config_updaters = []

    def freeze_all(self):
        self._freeze_all = True
        return self

    def create_config(self):
        self.with_range_init()
        config = get_empty_config()
        config.update(self.config_update)
        for config_updater in self._config_updaters:
            config = config_updater(config)
        return config

    def create_frozen_model(self):
        """Freeze first conv by default and freeze all convs if _freeze_all is True"""
        model = TwoConvTestModel()
        num_convs_to_freeze = -1 if self._freeze_all else 1
        for i, module in enumerate(get_all_modules_by_type(model, "Conv2d").values()):
            if i < num_convs_to_freeze or num_convs_to_freeze == -1:
                module.weight.requires_grad = False
        return model

    def add_algo(self, algo_builder: AlgoBuilder):
        self.config_update["compression"].append(algo_builder.get_config())
        return self

    def with_range_init(self):
        def add_range_init(config):
            for compression in config["compression"]:
                if compression["algorithm"] == "quantization":
                    if "initializer" not in compression:
                        compression["initializer"] = {}
                    compression["initializer"].update({"range": {"num_init_samples": 1}})
                    data_loader = create_ones_mock_dataloader(config)
                    config = NNCFConfig.from_dict(config)
                    config.register_extra_structs([QuantizationRangeInitArgs(wrap_dataloader_for_init(data_loader))])
            return config

        self._config_updaters.append(add_range_init)
        return self

    def ignore_first_conv(self):
        self.config_update["ignored_scopes"] = FIRST_NNCF_CONV_SCOPE
        return self

    def target_first_conv(self):
        for algo_dict in self.config_update["compression"]:
            algo_dict["target_scopes"] = FIRST_NNCF_CONV_SCOPE
        return self

    def expects_error(self):
        self.raising_error = True
        return self

    def expects_warning(self):
        self.printing_warning = True
        return self


TEST_PARAMS = [
    FrozenLayersTestStruct(name="no_compression"),
    FrozenLayersTestStruct(name="8_bits_quantization").add_algo(AlgoBuilder().name("quantization")).expects_warning(),
    FrozenLayersTestStruct(name="8_bits_quantization_all_frozen")
    .add_algo(AlgoBuilder().name("quantization"))
    .freeze_all()
    .expects_warning(),
    FrozenLayersTestStruct(name="8_bits_quantization_with_frozen_not_wrapped")
    .add_algo(AlgoBuilder().name("quantization"))
    .ignore_first_conv(),
    FrozenLayersTestStruct(name="8_bits_quantization_with_frozen_in_ignored_scope").add_algo(
        AlgoBuilder().name("quantization").ignore_first_conv()
    ),
    FrozenLayersTestStruct(name="8_bits_quantization_with_frozen_in_ignored_nncf_scope").add_algo(
        AlgoBuilder().name("quantization").ignore_first_conv()
    ),
    FrozenLayersTestStruct(name="8_bits_quantization_with_not_all_frozen_in_ignored_scope")
    .add_algo(AlgoBuilder().name("quantization").ignore_first_conv())
    .freeze_all()
    .expects_warning(),
    FrozenLayersTestStruct(name="mixed_precision_quantization")
    .add_algo(AlgoBuilder().name("quantization").mixed_precision())
    .expects_error(),
    FrozenLayersTestStruct(name="mixed_precision_quantization_with_frozen_not_wrapped")
    .add_algo(AlgoBuilder().name("quantization").mixed_precision())
    .ignore_first_conv(),
    FrozenLayersTestStruct(name="mixed_precision_quantization_with_frozen_in_ignored_scope").add_algo(
        AlgoBuilder().name("quantization").mixed_precision().ignore_first_conv()
    ),
    FrozenLayersTestStruct(name="mixed_precision_quantization_with_not_all_frozen_in_ignored_scope")
    .add_algo(AlgoBuilder().name("quantization").mixed_precision())
    .ignore_first_conv()
    .freeze_all()
    .expects_error(),
    FrozenLayersTestStruct(name="mixed_precision_quantization_with_frozen_in_target_scope")
    .add_algo(AlgoBuilder().name("quantization").mixed_precision())
    .target_first_conv()
    .expects_error(),
    FrozenLayersTestStruct(name="4_bits_quantization")
    .add_algo(AlgoBuilder().name("quantization").int4())
    .expects_error(),
    FrozenLayersTestStruct(name="4_bits_quantization_with_frozen_in_ignored_scope").add_algo(
        AlgoBuilder().name("quantization").int4().ignore_first_conv()
    ),
    FrozenLayersTestStruct(name="4_bits_quantization_with_not_all_frozen_in_ignored_scope")
    .add_algo(AlgoBuilder().name("quantization").int4().ignore_first_conv())
    .freeze_all()
    .expects_error(),
    FrozenLayersTestStruct(name="magnitude_sparsity")
    .add_algo(AlgoBuilder().name("magnitude_sparsity"))
    .expects_error(),
    FrozenLayersTestStruct(name="rb_sparsity").add_algo(AlgoBuilder().name("rb_sparsity")).expects_error(),
    FrozenLayersTestStruct(name="rb_sparsity_8_bits_quantization_with_frozen")
    .add_algo(AlgoBuilder().name("rb_sparsity"))
    .add_algo(AlgoBuilder().name("quantization"))
    .expects_error(),
    FrozenLayersTestStruct(name="rb_sparsity_8_bits_quantization_with_frozen_sparsity_in_ignored_scope")
    .add_algo(AlgoBuilder().name("rb_sparsity").ignore_first_conv())
    .add_algo(AlgoBuilder().name("quantization"))
    .expects_warning(),
    FrozenLayersTestStruct(name="const_sparsity").add_algo(AlgoBuilder().name("const_sparsity")).expects_warning(),
    FrozenLayersTestStruct(name="const_sparsity_8_bits_quantization")
    .add_algo(AlgoBuilder().name("const_sparsity"))
    .add_algo(AlgoBuilder().name("quantization"))
    .expects_warning(),
    FrozenLayersTestStruct(name="const_sparsity_4_bits_quantization")
    .add_algo(AlgoBuilder().name("const_sparsity"))
    .add_algo(AlgoBuilder().name("quantization").int4())
    .expects_error()
    .expects_warning(),
    FrozenLayersTestStruct(name="const_sparsity_4_bits_quantization_with_frozen_int4_in_ignored_scope")
    .add_algo(AlgoBuilder().name("const_sparsity"))
    .add_algo(AlgoBuilder().name("quantization").int4().ignore_first_conv())
    .expects_warning(),
    FrozenLayersTestStruct(name="rb_sparsity_4_bits_quantization")
    .add_algo(AlgoBuilder().name("rb_sparsity"))
    .add_algo(AlgoBuilder().name("quantization").int4())
    .expects_error(),
    FrozenLayersTestStruct(name="rb_sparsity_4_bits_quantization_with_int4_ignored")
    .add_algo(AlgoBuilder().name("rb_sparsity"))
    .add_algo(AlgoBuilder().name("quantization").int4().ignore_first_conv())
    .expects_error(),
    FrozenLayersTestStruct(name="rb_sparsity_4_bits_quantization_with_frozen_in_ignored_scope")
    .add_algo(AlgoBuilder().name("rb_sparsity"))
    .add_algo(AlgoBuilder().name("quantization").int4())
    .ignore_first_conv(),
    FrozenLayersTestStruct(name="filter_pruning_with_frozen_in_ignored_scope").add_algo(
        AlgoBuilder().name("filter_pruning").ignore_first_conv()
    ),
    FrozenLayersTestStruct(name="filter_pruning_with_frozen_in_ignored_scope")
    .add_algo(AlgoBuilder().name("filter_pruning"))
    .expects_error(),
]


@pytest.fixture()
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


@pytest.mark.parametrize("params", TEST_PARAMS, ids=[p.name for p in TEST_PARAMS])
def test_frozen_layers(_nncf_caplog, params):
    model = params.create_frozen_model()
    config = params.create_config()
    register_bn_adaptation_init_args(config)

    if params.raising_error:
        with pytest.raises(nncf.InternalError):
            __, _ = create_compressed_model_and_algo_for_test(model, config)
    else:
        __, _ = create_compressed_model_and_algo_for_test(model, config)
    are_frozen_layers_mentioned = "Frozen layers" in _nncf_caplog.text
    if params.printing_warning:
        assert are_frozen_layers_mentioned
    else:
        assert not are_frozen_layers_mentioned
