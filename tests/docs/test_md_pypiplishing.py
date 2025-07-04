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

import pytest

from tests.cross_fw.shared.paths import PROJECT_ROOT

GITHUB_BASE_URL = "https://github.com/openvinotoolkit/nncf/blob/develop"
MD_PYPI_PUBLISHING = PROJECT_ROOT / "docs" / "PyPiPublishing.md"
MD_README = PROJECT_ROOT / "README.md"

# Headers to skip in comparison
HEADER_TO_SKIP = [
    "## Table of contents",
]


def _read_markdown_headers(path: Path) -> list[str]:
    """Reads all markdown headers from a file after removing ID tags."""
    return [
        _remove_id_tags(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("#") and _remove_id_tags(line.strip()) not in HEADER_TO_SKIP
    ]


def _extract_section(content: str, header: str) -> str:
    """Extracts markdown content under a given header."""
    lines = content.splitlines()
    section = []
    in_section = False

    for line in lines:
        if line.strip().startswith("#") and _remove_id_tags(line.strip()) == header:
            in_section = True
            continue
        if in_section and line.strip().startswith("#"):
            break
        if in_section:
            section.append(line)

    return "\n".join(section).strip()


def _replace_relative_links(text: str) -> str:
    """Convert relative markdown links to absolute GitHub URLs."""

    def replacer(match):
        label, link = match.groups()
        if link.startswith("http"):
            return match.group(0)
        normalized_link = re.sub(r"^(\./|/)", "", link)
        return f"[{label}]({GITHUB_BASE_URL}/{normalized_link})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replacer, text)


def _remove_id_tags(text: str) -> str:
    """Remove  HTML anchor ID tags from markdown headers."""
    return re.sub(r'<a id="[^"]*"></a>', "", text)


# Load headers
HEADERS_TO_CHECK = _read_markdown_headers(MD_PYPI_PUBLISHING)
assert len(HEADERS_TO_CHECK) > 5, "No headers found in PyPiPublishing.md"


@pytest.mark.parametrize("header", HEADERS_TO_CHECK)
def test_alignment_pypi_publishing(header: str):
    content_src = MD_README.read_text(encoding="utf-8")
    content_dst = MD_PYPI_PUBLISHING.read_text(encoding="utf-8")

    section_src = _extract_section(content_src, header)
    section_dst = _extract_section(content_dst, header)

    if header == "# Neural Network Compression Framework (NNCF)":
        # This paragraph  in publishing markdown contains a special text in the begin before <div> and in the end
        match = re.search(r"</div>(.*)", section_src, re.S)
        section_src = match.group(1).strip()

        match = re.search(r"(.*)For more information about NNCF, see:", section_dst, re.S)
        section_dst = match.group(1).strip()

    # Transform section from README.md to be published
    section_src = _replace_relative_links(section_src)
    section_src = _remove_id_tags(section_src)
    section_src = section_src.strip()

    assert section_src == section_dst, f"Mismatch in section: {header}"
