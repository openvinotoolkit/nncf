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


import json
import re
from pathlib import Path
from typing import Any, Union

from nncf.common.utils.os import safe_open


class JSONWithComments(json.JSONDecoder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def decode(self, s: str) -> Any:  # type: ignore
        pattern = r'(?:"(?:\\.|[^"\\])*")|//.*'
        s = re.sub(pattern, lambda m: m.group(0) if m.group(0).startswith('"') else "", s)
        return super().decode(s)


def read_config(path: Union[str, Path]) -> Any:
    file_path = Path(path)
    with safe_open(file_path) as f:
        data = json.load(f, cls=JSONWithComments)
    return data
