---
name: generate-release-notes
description: Generate release notes for the new NNCF release.
---

# Generate release notes for NNCF

## Context

**File Paths:**

- `ReleaseNotes.md`
- `src/nncf/version.py`
- `.github/skills/generate_release_notes/parse_commits.sh`

## Instructions

## Execution Logic

1. **Execute**: Run the local file `.github/skills/generate_release_notes/parse_commits.sh`, collect stdout that contains the commits in format `AUTHOR;COMMIT_HASH;COMMIT_MESSAGE`.
2. **Read**: Read the current version of NNCF from `src/nncf/version.py` to determine the new version number for the release notes.
3. **Updater**: Update `ReleaseNotes.md` with the new release notes based on the commits collected from the script execution. The release notes should be generated according to the following guidelines:

    - The release notes should be formatted in a clear and concise manner, using formats from the previous releases as a reference.
    - Each change should be categorized based on the backend (e.g., OpenVINO, PyTorch, ONNX) by adding in the beginning of the line the backend name in  brackets. For example, `(OpenVINO) Fix bug in NNCF for OpenVINO`.
    - If it cannot be categorized by the backend, it should not add any backend name in the beginning of the line. For example, `Fix bug in NNCF`.
    - Each sub category should be sorted by the backend, start with general changes (without backend name) and then sorted in order `OpenVINO`, `PyTorch`, `ONNX`.
    - Add a link to the pull request for each change in the end of line in the format `(#PR_NUMBER)`. For example, if the commit message is "Fix bug in NNCF (#123)", the release notes should include "Fix bug in NNCF (#123)".
    - If not enough information in the commit message, to generate the change description, read full commit message from the commit hash using `git log --format=%B -n 1 COMMIT_HASH` and use it to generate the change description.
    - Template for release notes:

    ```txt
    ## New in Release X.Y.Z

    Breaking changes:
    - ...
    General:
    - ...
    Features:
    - ...
    Fixes:
    - ...
    Improvements:
    - ...
    Tutorials:
    - ...
    Known issues:
    - ...
    Deprecations/Removals:
    - ...
    Requirements:
    - ...
    ```

    - Keep `Tutorials` and `Known issues` sections empty if there are no relevant changes with `- ...` in these sections.

    ```txt
    Tutorials:
    - ...
    Known issues:
    - ...
    ```

## Constraints

- The skill should only update the `ReleaseNotes.md` file as its final output.
- `ReleaseNotes.md` is a large file, should read the first 100 lines of the file to find the last release section and insert the new release notes before it.
- To collect commits available use only `parse_commits.sh` script, dont run any other shell commands to read commits. To read the full commit message for generating change description, use only `git log --format=%B -n 1 COMMIT_HASH` command. Do not run any other shell commands to read commit messages.
