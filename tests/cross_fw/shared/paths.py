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

import os
from pathlib import Path

TEST_ROOT = Path(__file__).parent.absolute().parents[1]
PROJECT_ROOT = TEST_ROOT.parent.absolute()
EXAMPLES_DIR = PROJECT_ROOT / "examples"
GITHUB_REPO_URL = "https://github.com/openvinotoolkit/nncf/"


DATASET_DEFINITIONS_PATH = TEST_ROOT / "cross_fw" / "shared" / "data" / "dataset_definitions.yml"

ROOT_PYTHONPATH_ENV = os.environ.copy()
ROOT_PYTHONPATH_ENV["PYTHONPATH"] = f"{PROJECT_ROOT}:{ROOT_PYTHONPATH_ENV.get('PYTHONPATH', '')}".strip(":")


def get_accuracy_aware_checkpoint_dir_path(model_specific_run_dir: Path) -> Path:
    model_time_dirs = [x for x in model_specific_run_dir.glob("*/") if x.is_dir()]
    aa_subdir = model_time_dirs[0] / "accuracy_aware_training"
    aa_time_dirs = [x for x in aa_subdir.glob("*/") if x.is_dir()]
    assert len(aa_time_dirs) == 1
    return aa_time_dirs[0]
