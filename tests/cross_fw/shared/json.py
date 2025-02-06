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
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np


def load_json(stats_path: Path):
    with open(stats_path, "r", encoding="utf8") as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def dump_to_json(local_path: Path, data: Dict[str, np.ndarray]):
    with open(local_path, "w", encoding="utf8") as file:
        json.dump(deepcopy(data), file, indent=4, cls=NumpyEncoder)
