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
import argparse
import re
import subprocess
from pathlib import Path

import tomllib


def _package_name(requirement_line: str) -> str | None:
    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*", requirement_line.strip())
    return match.group(0).lower() if match else None


def collect_requirements_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "**/requirements*.txt", "requirements.txt"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = {Path(line) for line in result.stdout.splitlines() if line.strip()}
    print(f"Found {len(paths)} requirements.txt files")
    return sorted(paths)


def get_unique_packages_from_requirements_files() -> list[str]:
    unique_packages = set()
    for req_file in collect_requirements_files():
        text = req_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            package_name = _package_name(line)
            if package_name:
                unique_packages.add(package_name)
    return sorted(unique_packages)


def get_package_requirements() -> list[str]:
    project_requirements = []
    toml_text = Path("pyproject.toml").read_text(encoding="utf-8")
    toml_data = tomllib.loads(toml_text)

    project_requirements.extend(toml_data["project"]["dependencies"])
    for optional_deps in toml_data["project"].get("optional-dependencies", {}).values():
        project_requirements.extend(optional_deps)

    names = set()
    for package in project_requirements:
        name = _package_name(package)
        if name:
            names.add(name)

    return sorted(names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dependencies from all requirements.txt files and pyproject.toml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If omitted, collected package names are printed to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requirements = get_unique_packages_from_requirements_files()
    package_requirements = get_package_requirements()
    total_requirements = sorted(set(requirements) | set(package_requirements))
    output_text = "\n".join(total_requirements) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
