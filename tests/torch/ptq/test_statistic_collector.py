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

import pytest

from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from tests.common.experimental.test_statistic_collector import TemplateTestStatisticCollector


class TestPTStatisticCollector(TemplateTestStatisticCollector):
    @pytest.mark.skip
    def test_raw_max_stat_building(self, raw_statistic_cls: RawTensorStatistic):
        pass
