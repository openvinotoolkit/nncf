# Copyright (c) 2026 Intel Corporation
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
import tempfile
from contextlib import contextmanager

import nbformat
import pytest
from nbclient import NotebookClient

from tests.cross_fw.shared.paths import PROJECT_ROOT

TOOL_DIR = PROJECT_ROOT / "tools" / "activation_profiler"
NOTEBOOK_FILE = TOOL_DIR / "nncf_profiler_example.ipynb"


@contextmanager
def isolated_ipython_cwd(cwd):
    original_cwd = os.getcwd()
    original_ipythondir = os.environ.get("IPYTHONDIR")

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["IPYTHONDIR"] = tmpdir
        try:
            os.chdir(cwd)
            yield
        finally:
            os.chdir(original_cwd)
            if original_ipythondir is None:
                os.environ.pop("IPYTHONDIR", None)
            else:
                os.environ["IPYTHONDIR"] = original_ipythondir


@pytest.mark.xfail(reason="Ticket: 180038")
def test_nncf_profiling_notebook():
    """Test that the nncf_profiler_example.ipynb notebook runs successfully."""

    # Check if notebook exists
    assert NOTEBOOK_FILE.exists(), f"Notebook not found at {NOTEBOOK_FILE}"

    # Read the notebook
    with NOTEBOOK_FILE.open() as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook using nbclient with a temporary IPython config directory
    with isolated_ipython_cwd(TOOL_DIR):
        client = NotebookClient(nb, timeout=600)
        client.execute()
