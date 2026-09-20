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

import re

from tests.cross_fw.shared.paths import PROJECT_ROOT

MAKEFILE = PROJECT_ROOT / "Makefile"

# A literal Python path written out in a recipe, e.g. `python tests/.../fuzz_target.py`.
# Deliberately literal-only: a path assembled from a Make variable, reached after a `cd`,
# or run as `python -m package.module` is not matched and not claimed to be.
SCRIPT_RE = re.compile(r"(?<![\w/.-])((?:tests|tools|examples|src|docs|scripts)/[\w/.-]+\.py)")


def _referenced_scripts() -> list[tuple[int, str]]:
    """Literal Python script paths written out in Makefile recipes, with line numbers."""
    found = []
    for lineno, line in enumerate(MAKEFILE.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("\t") or line.lstrip().startswith("#"):
            continue  # not a recipe command
        for match in SCRIPT_RE.finditer(line):
            found.append((lineno, match.group(1)))
    return found


def test_makefile_scripts_exist():
    """Every Python script written out literally in a Makefile recipe must exist.

    This is narrow on purpose. It does not check requirements files, paths built from
    Make variables, or `python -m` invocations, so a green run here is not a claim that
    every Makefile target works. It catches one recurring mistake: a file is moved or
    renamed and the recipe that runs it keeps the old path, which makes the target fail
    the moment anyone invokes it.
    """
    missing = [
        f"Makefile:{lineno}: {path}" for lineno, path in _referenced_scripts() if not (PROJECT_ROOT / path).is_file()
    ]
    assert not missing, "Makefile invokes scripts that do not exist:\n" + "\n".join(missing)
