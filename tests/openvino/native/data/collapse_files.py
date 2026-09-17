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
"""
Collapse the reference files stored per version in this directory.

Usage:
    python collapse_files.py 2026.1
    python collapse_files.py 2026.1 --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

from packaging import version
from packaging.version import Version

DATA_ROOT = Path(__file__).parent


def parse_version_dirs(root: Path) -> dict[Version, Path]:
    """
    Collect version subdirectories of ``root``.
    """
    version_dirs: dict[Version, Path] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            version_dirs[version.parse(entry.name)] = entry
        except version.InvalidVersion:
            continue
    return version_dirs


def collect_rel_paths(version_dir: Path) -> set[Path]:
    """
    Collect reference file paths of a version folder relative to that folder.
    """
    return {path.relative_to(version_dir) for path in version_dir.rglob("*") if path.is_file()}


def collapse(root: Path, cutoff_name: str, dry_run: bool) -> None:
    """
    Collapse reference folders older than ``cutoff_name`` into a single base folder.
    """
    cutoff = version.parse(cutoff_name)
    version_dirs = parse_version_dirs(root)

    if not version_dirs:
        msg = f"No version folders found in {root}"
        raise RuntimeError(msg)

    below = {ver: path for ver, path in version_dirs.items() if ver < cutoff}
    if not below:
        print(f"Nothing to collapse: no reference folders older than {cutoff_name}.")
        return

    # Reference files already available for version `cutoff` or any newer version.
    # These must not be copied into the base folder (and nothing is created in newer folders).
    covered: set[Path] = set()
    for ver, version_dir in version_dirs.items():
        if ver >= cutoff:
            covered.update(collect_rel_paths(version_dir))

    # For every reference file present only in older folders, pick the content that
    # resolves for version `cutoff` (the highest version folder < cutoff that contains it).
    older = sorted((ver for ver in below), key=lambda v: v)
    resolved: dict[Path, Path] = {}
    for ver in older:
        version_dir = version_dirs[ver]
        for rel_path in collect_rel_paths(version_dir):
            if rel_path in covered:
                continue
            resolved[rel_path] = version_dir / rel_path

    base_dir = root / cutoff_name

    print(f"Collapsing reference files older than {cutoff_name} into '{base_dir.name}/':")
    for rel_path in sorted(resolved, key=str):
        source = resolved[rel_path]
        target = base_dir / rel_path
        if source == target:
            continue
        print(f"  {source.relative_to(root)} -> {target.relative_to(root)}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    print("Removing obsolete version folders:")
    for ver in sorted(below, key=lambda v: v):
        version_dir = below[ver]
        if version_dir == base_dir:
            continue
        print(f"  {version_dir.relative_to(root)}")
        if not dry_run:
            shutil.rmtree(version_dir)

    if dry_run:
        print("Dry run: no files were modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "version",
        help="Cutoff version (e.g. 2026.1). Reference files for older versions are collapsed.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory that holds the per-version reference folders (default: this script's folder).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without modifying any files.",
    )
    args = parser.parse_args()

    try:
        version.parse(args.version)
    except version.InvalidVersion:
        parser.error(f"Invalid version: {args.version!r}")

    collapse(args.data_root, args.version, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
