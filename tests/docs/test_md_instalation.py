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

import re
from pathlib import Path

import yaml

from tests.cross_fw.shared.paths import PROJECT_ROOT

MD_INSTALLATION = PROJECT_ROOT / "docs" / "Installation.md"
CONSTRAINTS = PROJECT_ROOT / "constraints.txt"
WORKFLOW_PRECOMMIT = PROJECT_ROOT / ".github" / "workflows" / "precommit.yml"
WORKFLOW_CALL_PRECOMMIT = PROJECT_ROOT / ".github" / "workflows" / "call_precommit.yml"


def _extract_section(content: str, header: str) -> str:
    """
    Extracts content under a specific markdown header until the next any header.
    """
    lines = content.splitlines()
    section_lines = []
    in_section = False
    for line in lines:
        if line.strip().startswith("#") and line == header:
            in_section = True
            continue
        if in_section and line.strip().startswith("#"):
            break
        if in_section:
            section_lines.append(line)
    return "\n".join(section_lines).strip()


def _parse_last_versions_from_md() -> dict[str, str]:
    """
    Parse from the last corresponded version in Installation.md file.
    https://github.com/openvinotoolkit/nncf/blob/develop/docs/Installation.md#corresponding-versions

    | NNCF      | OpenVINO   | PyTorch  | ONNX     | TensorFlow | Python |
    |-----------|------------|----------|----------|------------|--------|
    | `develop` | `2025.2.0` | `2.7.1`  | `1.17.0` | `2.15.1`   | `3.10` |

    :return: dict with versions for frameworks and python
    """
    content = MD_INSTALLATION.read_text(encoding="utf-8")
    section = _extract_section(content, "## Corresponding versions")
    table = [s for s in section.splitlines() if s.strip().startswith("|")]
    headers = [h.strip().lower() for h in table[0].strip("|").split("|")]
    first_data_row = [cell.strip(" `") for cell in table[2].strip("|").split("|")]
    version_dict = dict(zip(headers[1:], first_data_row[1:]))
    version_dict.pop("tensorflow", None)
    return version_dict


def _parse_versions_from_constraints_file() -> dict[str, str]:
    constraints = CONSTRAINTS.read_text(encoding="utf-8")

    def _parse(pattern: str) -> str:
        match = re.search(pattern, constraints)
        return match.group(1) if match else "not found"

    return {
        "openvino": _parse(r"openvino==(\d+\.\d+.\d+)"),
        "pytorch": _parse(r"torch==(\d+\.\d+.\d+)"),
        "onnx": _parse(r"onnx==(\d+\.\d+.\d+)"),
    }


def _parse_python_version_from_precommit_workflow(with_patch: bool = False) -> str:
    workflow = WORKFLOW_PRECOMMIT.read_text(encoding="utf-8")
    ret = re.search(r"python_version: \"(\d+\.\d+\.\d+)", workflow).group(1)
    if not with_patch:
        ret = ".".join(ret.split(".")[:2])
    return ret


def _get_cuda_version_from_workflow(workflow_file: Path, job_name: str) -> str:
    """"""
    with workflow_file.open() as f:
        data = yaml.safe_load(f)
    steps = data["jobs"][job_name]["steps"]
    for step in steps:
        run_command = step.get("run", "")
        match = re.search(r"cuda[_-](\d+\.\d+)", run_command)
        if match:
            return match.group(1)
    return "not found"


def test_corresponded_versions():
    md_versions = _parse_last_versions_from_md()
    actual_versions = _parse_versions_from_constraints_file()
    actual_versions["python"] = _parse_python_version_from_precommit_workflow()
    assert actual_versions == md_versions, "Update last corresponded versions in ./docs/Installation.md"


def test_test_environment():
    """Ensure that the test environment is set up correctly."""
    md_content = MD_INSTALLATION.read_text(encoding="utf-8")
    pattern = (
        r"This repository is tested on Python\* (\d+\.\d+\.\d+), "
        r"PyTorch\* (\d+\.\d+\.\d+) \(NVidia CUDA\\\* Toolkit (\d+\.\d+)\)"
    )
    match = re.search(pattern, md_content)

    md_versions = {
        "python": match.group(1),
        "pytorch": match.group(2),
        "cuda_torch": match.group(3),
    }

    actual_versions = _parse_versions_from_constraints_file()

    ref = {
        "python": _parse_python_version_from_precommit_workflow(True),
        "pytorch": actual_versions["pytorch"],
        "cuda_torch": _get_cuda_version_from_workflow(WORKFLOW_CALL_PRECOMMIT, "pytorch-cuda"),
    }
    print(md_versions, ref)
    assert md_versions == ref, (
        f"Need to change line `{pattern[:30]}...` line in {MD_INSTALLATION.relative_to(PROJECT_ROOT)}."
    )
