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


# NOTE 1: This module generates the dynamic version for the package during the build process.
# It provides the version attribute for setuptools as specified in pyproject.toml:
#   [tool.setuptools.dynamic]
#   version = { attr = "_version_helper.version" }

# NOTE 2: To generate a release package without including the commit hash in the version,
# set the NNCF_RELEASE_BUILD environment variable.

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

NNCF_VERSION_FILE = "nncf/version.py"


def get_custom_version() -> str:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", Path(NNCF_VERSION_FILE).read_text(), re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string.")
    version_value = version_match.group(1)
    is_building_release = "NNCF_RELEASE_BUILD" in os.environ

    if not is_building_release:

        import subprocess  # nosec

        dev_version_id = "unknown_version"
        repo_root = os.path.dirname(os.path.realpath(__file__))

        # Get commit hash
        with contextlib.suppress(subprocess.CalledProcessError):
            dev_version_id = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)  # nosec
                .strip()
                .decode()
            )

        # Detect modified files
        with contextlib.suppress(subprocess.CalledProcessError):
            repo_root = os.path.dirname(os.path.realpath(__file__))
            run = subprocess.run(["git", "diff-index", "--quiet", "HEAD"], cwd=repo_root)  # nosec
            if run.returncode == 1:
                dev_version_id += "dirty"

        return version_value + f".dev0+{dev_version_id}"

    return version_value


version: str


def __getattr__(name: str) -> str:
    if name == "version":
        global version
        version = get_custom_version()
        return version
    raise AttributeError(name)
