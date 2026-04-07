---
name: bump-version
description: Increment the version of NNCF in src/nncf/version.py.
---

# Bump version of NNCF

## Context

- **File Path:** `src/nncf/version.py`
- **Variable Name:** `__version__`
- **Format:** Standard Semantic Versioning (`"Major.Minor.Patch"`)

## Execution Logic

1. **Read** the current version from `src/nncf/version.py` (pattern: `__version__ = "X.Y.Z"`)
2. **Parse** Major, Minor, and Patch components
3. **Increment** based on user intent:
    - **Major**: increment Major, reset Minor and Patch to 0
    - **Minor** (default): increment Minor, reset Patch to 0
    - **Patch**: increment Patch only
4. **Update** `__version__` with new version string in double quotes
5. **Preserve** all other content in the file
