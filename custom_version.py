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

"""
Dynamic Version Generation for Package Builds

This module is responsible for generating the dynamic version for the package during the build process.
It provides the `version` attribute for setuptools as specified in the `pyproject.toml` configuration file:

    [tool.setuptools.dynamic]
    version = { attr = "_version_helper.version" }

Environment Variable Instructions:
-----------------------------------
1. **NNCF_RELEASE_BUILD**:
    - Set this environment variable to generate a release package using `python -m build` without including
      the commit hash in the version.
    - If this variable is not set, the file `nncf/version.py` will be overridden with a custom version
      that includes the commit hash. Example usage:

        NNCF_RELEASE_BUILD=1 python -m build

2. **NNCF_BUILD_SUFFIX**:
    - Use this environment variable to specify a particular suffix for the build. Example usage:

        NNCF_BUILD_SUFFIX="rc1" python -m build

Post-Build Recommendation:
---------------------------
After generating the package, it is recommended to revert any changes to `nncf/version.py` to avoid potential conflicts.
This can be done using the following command:

    git checkout nncf/version.py

This ensures that `nncf/version.py` remains in its original state after the dynamic versioning process.
"""


from __future__ import annotations

import contextlib
import os
import re
import subprocess
from pathlib import Path

NNCF_VERSION_FILE = "nncf/version.py"


def get_custom_version() -> str:
    version_match = re.search(
        r"^__version__ = ['\"]((\d+\.\d+\.\d+)([^'\"]*))['\"]", Path(NNCF_VERSION_FILE).read_text(), re.M
    )
    if not version_match:
        raise RuntimeError("Unable to find version string.")

    version_full = version_match.group(1)
    version_value = version_match.group(2)
    version_suffix = version_match.group(3)

    nncf_build_suffix = os.environ.get("NNCF_BUILD_SUFFIX")
    if nncf_build_suffix:
        # Suffix expected on build package
        return f"{version_value}.{nncf_build_suffix}"

    is_building_release = "NNCF_RELEASE_BUILD" in os.environ
    if is_building_release or version_suffix:
        return version_full

    dev_version_id = "unknown_version"
    repo_root = os.path.dirname(os.path.realpath(__file__))

    # Get commit hash
    with contextlib.suppress(subprocess.CalledProcessError):
        dev_version_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root).strip().decode()  # nosec
        )

    # Detect modified files
    with contextlib.suppress(subprocess.CalledProcessError):
        run = subprocess.run(["git", "diff-index", "--quiet", "HEAD"], cwd=repo_root)  # nosec
        if run.returncode == 1:
            dev_version_id += "dirty"

    return version_value + f".dev0+{dev_version_id}"


version: str


def __getattr__(name: str) -> str:

    if name == "version":
        global version
        version = get_custom_version()

        # Rewrite version.py to pass custom version to package
        if os.environ.get("_PYPROJECT_HOOKS_BUILD_BACKEND"):
            content = Path(NNCF_VERSION_FILE).read_text()
            version_str = re.search(r"^__version__ = ['\"][^'\"]*['\"]", content, re.M).group(0)
            content = content.replace(version_str, f'__version__ = "{version}"')
            Path(NNCF_VERSION_FILE).write_text(content)

        return version

    raise AttributeError(name)
