# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import RawStatisticCollector
from nncf.onnx.tensor import ONNXNNCFTensor


class ONNXMinMaxStatisticCollector(MinMaxStatisticCollector):
    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)


class ONNXMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)


class ONNXMeanStatisticCollector(MeanStatisticCollector):
    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)


class ONNXRawStatisticCollector(RawStatisticCollector):
    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)
