# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict

import torch

from nncf.tensor.definitions import TensorBackendType
from tests.cross_fw.test_templates.test_statistics_serializer import TemplateTestStatisticsSerializer


class TestTorchStatisticsSerializer(TemplateTestStatisticsSerializer):
    def _get_backend_statistics(self) -> Dict[str, Dict[str, torch.tensor]]:
        return {
            "layer/1/activation": {"mean": torch.tensor([0.1, 0.2, 0.3])},
            "layer/2/activation": {"variance": torch.tensor([0.05, 0.06, 0.07])},
        }

    def _get_tensor_backend(self) -> TensorBackendType:
        return TensorBackendType.TORCH

    def is_equal(self, a1: Dict[str, torch.tensor], a2: Dict[str, torch.tensor]) -> bool:
        for key in a1:
            if key not in a2:
                return False
            if not torch.allclose(a1[key], a2[key]):
                return False
        return True
