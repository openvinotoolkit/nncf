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
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

SerializableData = Union[str, Enum, bool]


@dataclass
class CollectedEvent:
    """
    name: The name for the event, will be sent as `event_action`
    data: Optional - the data associated with the event. Must be a string - serialize to string if it is not.
    int_data: Optional - the integer data associated with the event. Must be a positive integer.
    """

    name: str
    data: Optional[SerializableData] = None  # GA limitations
    int_data: Optional[int] = None


class TelemetryExtractor(ABC):
    """
    Interface for custom telemetry extractors, to be used with the `nncf.telemetry.tracked_function` decorator.
    """

    def __init__(self, argname: str = ""):
        self._argname = argname

    @property
    def argname(self) -> Optional[str]:
        return self._argname

    @abstractmethod
    def extract(self, argvalue: Any) -> CollectedEvent:
        """
        Implement this method to prepare the telemetry event data from the tracked function's argument value
        passed via `argvalue`.
        """


class VerbatimTelemetryExtractor(TelemetryExtractor):
    def extract(self, argvalue: SerializableData) -> CollectedEvent:
        if isinstance(argvalue, Enum):
            argvalue = str(argvalue.value)
        if isinstance(argvalue, bool):
            argvalue = "enabled" if argvalue else "disabled"
        return CollectedEvent(name=self._argname, data=argvalue)


class FunctionCallTelemetryExtractor(TelemetryExtractor):
    def __init__(self, argvalue: Any = None) -> None:
        super().__init__()
        self._argvalue = argvalue

    def extract(self, _: Any) -> CollectedEvent:
        return CollectedEvent(name="function_call", data=self._argvalue)
