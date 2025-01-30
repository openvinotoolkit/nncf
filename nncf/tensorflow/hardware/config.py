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

from typing import List, Type

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.hardware.config import HWConfig
from nncf.tensorflow.graph.metatypes.common import get_operator_metatypes


class TFHWConfig(HWConfig):
    def _get_available_operator_metatypes_for_matching(self) -> List[Type[OperatorMetatype]]:
        return get_operator_metatypes()
