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

from typing import Any

from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.extractors import TelemetryExtractor


class ModelProcessedWithStripApi(TelemetryExtractor):
    def extract(self, _: Any) -> CollectedEvent:
        return CollectedEvent(name="model_processed", data="strip_api")

class NNCFNetworkGeneratedFromWrapApi(TelemetryExtractor):
    def extract(self) -> CollectedEvent:
        return CollectedEvent(name="nncf_network_generated", data="wrap_api")

class DatasetGeneratedFromApi(TelemetryExtractor):
    def extract(self) -> CollectedEvent:
        return CollectedEvent(name="dataset_generated", data="generate_api")
