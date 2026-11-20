#!/bin/bash
# Set the pinned OpenVINO version in constraints.txt, all requirements*.txt tracked by git
# and pyproject.toml. For pre-release versions (rcN / devN) the nightly wheel index is added,
# for final versions it is removed.
#
# Usage: .github/skills/prepare_release/set_openvino_version.sh <openvino_version>
#   e.g. .github/skills/prepare_release/set_openvino_version.sh 2026.5.0rc1
#        .github/skills/prepare_release/set_openvino_version.sh 2026.5.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <openvino_version>" >&2
    exit 1
fi

VERSION="$1"
NIGHTLY_URL="https://storage.openvinotoolkit.org/simple/wheels/nightly"
EXTRA_INDEX_LINE="--extra-index-url ${NIGHTLY_URL} --pre"

if [[ "$VERSION" =~ (rc|dev) ]]; then
    PRERELEASE=1
else
    PRERELEASE=0
fi

cd "$(git rev-parse --show-toplevel)"

FILES=$(git ls-files 'constraints.txt' '**/requirements*.txt' | xargs grep -l '^openvino==' || true)

for f in $FILES; do
    # Drop any existing nightly index line, then rewrite the pin.
    sed -i "\#^--extra-index-url ${NIGHTLY_URL}#d" "$f"
    sed -i -E "s#^openvino==.*#openvino==${VERSION}#" "$f"
    if [[ $PRERELEASE -eq 1 ]]; then
        sed -i "/^openvino==${VERSION}\$/a ${EXTRA_INDEX_LINE}" "$f"
    fi
    echo "updated $f"
done

# pyproject.toml: [tool.uv] block is needed so `uv pip install` can resolve pre-release wheels.
if [[ $PRERELEASE -eq 1 ]]; then
    if ! grep -q '^\[tool.uv\]' pyproject.toml; then
        sed -i '/^\[tool.setuptools.dynamic\]/i [tool.uv]\
extra-index-url = ["'"${NIGHTLY_URL}"'"]\
prerelease = "allow"\
index-strategy = "unsafe-best-match"\
' pyproject.toml
        echo "added [tool.uv] to pyproject.toml"
    fi
else
    if grep -q '^\[tool.uv\]' pyproject.toml; then
        sed -i '/^\[tool.uv\]/,/^$/d' pyproject.toml
        echo "removed [tool.uv] from pyproject.toml"
    fi
fi

echo "OpenVINO version set to ${VERSION} (prerelease=${PRERELEASE})"
