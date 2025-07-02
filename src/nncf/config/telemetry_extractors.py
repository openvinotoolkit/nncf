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

from nncf.config import NNCFConfig
from nncf.config.extractors import extract_algorithm_names
from nncf.telemetry.extractors import CollectedEvent
from nncf.telemetry.extractors import TelemetryExtractor


class CompressionStartedFromConfig(TelemetryExtractor):
    def extract(self, argvalue: NNCFConfig) -> CollectedEvent:
        algo_names = extract_algorithm_names(argvalue)
        return CollectedEvent(name="compression_started", data=",".join(algo_names))
