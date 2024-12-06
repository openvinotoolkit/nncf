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

import numpy as np

from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend
from tests.cross_fw.test_templates.test_statistics_serializer import TemplateTestStatisticsSerializer


class TestNPStatisticsSerializer(TemplateTestStatisticsSerializer):
    def _get_backend_statistics(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "layer/1/activation": {"mean": Tensor(np.array([0.1, 0.2, 0.3]))},
            "layer/2/activation": {"variance": Tensor(np.array([0.05, 0.06, 0.07]))},
        }

    def _get_tensor_backend(self) -> TensorBackend:
        return TensorBackend.numpy

    def is_equal(self, a1: Dict[str, Tensor], a2: Dict[str, Tensor]) -> bool:
        for key in a1:
            if key not in a2:
                return False
            if not np.array_equal(a1[key].data, a2[key].data):
                return False
        return True
