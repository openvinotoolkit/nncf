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
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_nncf_profiling_notebook():
    """Test that the nncf_profiler_example.ipynb notebook runs successfully."""
    notebook_path = Path(__file__).parent.parent.parent / "tools" / "profiler" / "nncf_profiler_example.ipynb"
    profiler_dir = Path(__file__).parent.parent.parent / "tools" / "profiler"

    # Check if notebook exists
    assert notebook_path.exists(), f"Notebook not found at {notebook_path}"

    # Read the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook using nbclient with a temporary IPython config directory
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["IPYTHONDIR"] = tmpdir
        # Save original working directory
        original_cwd = os.getcwd()
        try:
            # Change to profiler directory for notebook execution
            os.chdir(str(profiler_dir))
            client = NotebookClient(nb, timeout=600)
            client.execute()
        finally:
            # Clean up: restore working directory and environment variable
            os.chdir(original_cwd)
            if "IPYTHONDIR" in os.environ:
                del os.environ["IPYTHONDIR"]
