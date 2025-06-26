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

from nncf.common.utils.api_marker import api


@api()
class Statistics(ABC):
    """
    Contains a collection of model- or compression-related data and provides a way for its human-readable
    representation.
    """

    @abstractmethod
    def to_str(self) -> str:
        """
        Returns a representation of the statistics as a human-readable string.
        """
