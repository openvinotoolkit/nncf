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

from tests.cross_fw.shared.paths import TEST_ROOT
from tools.visualize_compression_results import visualize


def test_visualization_of_compression_results(tmp_path):
    in_file = TEST_ROOT / "tools" / "data" / "phi3_asym.csv"
    ref_md_file = TEST_ROOT / "tools" / "data" / "phi3_asym.md"

    visualize(in_file, tmp_path)

    md_file = tmp_path / (in_file.stem + ".md")
    assert md_file.exists()
    assert md_file.with_suffix(".png").exists()
    assert ref_md_file.read_text()[:-1] == md_file.read_text()  # ref file ends with a newline character by code style
