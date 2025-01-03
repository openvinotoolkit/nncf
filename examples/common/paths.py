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
import datetime
from pathlib import Path

from examples.common.sample_config import SampleConfig


def configure_paths(config: SampleConfig, run_name: str):
    config.name = run_name
    d = datetime.datetime.now()
    run_id = "{:%Y-%m-%d__%H-%M-%S}".format(d)
    log_dir = Path(config.log_dir) / run_name / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir = str(log_dir)

    if config.nncf_config is not None:
        config.nncf_config["log_dir"] = config.log_dir

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir
    checkpoint_save_dir = Path(config.checkpoint_save_dir)
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)

    # create aux dirs
    intermediate_checkpoints_path = log_dir / "intermediate_checkpoints"
    intermediate_checkpoints_path.mkdir(parents=True, exist_ok=True)
    config.intermediate_checkpoints_path = str(intermediate_checkpoints_path)
