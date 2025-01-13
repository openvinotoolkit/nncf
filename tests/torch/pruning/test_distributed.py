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

from torch import nn

from nncf.torch import register_default_init_args
from nncf.torch.structures import DistributedCallbacksArgs
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import get_basic_pruning_config


def test_default_distributed_init_struct():
    config = get_basic_pruning_config()
    init_loader = create_ones_mock_dataloader(config)
    register_default_init_args(config, init_loader)

    dist_callbacks = config.get_extra_struct(DistributedCallbacksArgs)
    assert callable(dist_callbacks.wrap_model)
    assert callable(dist_callbacks.unwrap_model)


def test_distributed_init_struct():
    class FakeModelClass:
        def __init__(self, model_: nn.Module):
            self.model = model_

        def unwrap(self):
            return self.model

    config = get_basic_pruning_config()
    init_loader = create_ones_mock_dataloader(config)
    wrapper_callback = FakeModelClass
    unwrapper_callback = lambda x: x.unwrap()
    nncf_config = register_default_init_args(
        config, init_loader, distributed_callbacks=(wrapper_callback, unwrapper_callback)
    )

    dist_callbacks = nncf_config.get_extra_struct(DistributedCallbacksArgs)
    model = PruningTestModel()
    wrapped_model = dist_callbacks.wrap_model(model)
    assert isinstance(wrapped_model, FakeModelClass)
    unwrapped_model = dist_callbacks.unwrap_model(wrapped_model)
    assert unwrapped_model == model
