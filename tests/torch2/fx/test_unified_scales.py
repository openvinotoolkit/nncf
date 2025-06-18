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

import torch

from tests.cross_fw.test_templates.test_unified_scales import TemplateTestUnifiedScales
from tests.torch2.fx.helpers import get_torch_fx_model_q_transformed


class TestUnifiedScales(TemplateTestUnifiedScales):
    def get_backend_specific_model(self, model: torch.nn.Module) -> torch.fx.GraphModule:
        q_input_shape = model.Q_INPUT_SHAPE
        kv_input_shape = model.KV_INPUT_SHAPE
        backend_model = get_torch_fx_model_q_transformed(
            model,
            (
                torch.ones(q_input_shape),
                torch.ones(q_input_shape),
                torch.ones(kv_input_shape),
                torch.ones(kv_input_shape),
            ),
        )

        return backend_model
