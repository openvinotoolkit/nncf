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

from typing import Type

from nncf.torch.quantization.precision_init.autoq_init import AutoQPrecisionInitializer
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.hawq_init import HAWQPrecisionInitializer
from nncf.torch.quantization.precision_init.manual_init import ManualPrecisionInitializer


class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str) -> Type[BasePrecisionInitializer]:
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        if init_type == "autoq":
            return AutoQPrecisionInitializer
        raise NotImplementedError
