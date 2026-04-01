---
name: generate-release-notes
description: Generate release notes for the new NNCF release.
---

# Generate release notes for NNCF

## Contexts

**Files:**

- `ReleaseNotes.md`
- `src/nncf/version.py`
- `.github/skills/generate_release_notes/parse_commits.sh`

## Execution Logic

1. **Extract Commits:** Run `.github/skills/generate_release_notes/parse_commits.sh`. Format of stdout: `AUTHOR;COMMIT_HASH;COMMIT_MESSAGE`.
2. **Read Version:** Get new version from `src/nncf/version.py`.
3. **Categorize by Backend:** Prepend backend name in brackets based on keywords:
    - OpenVINO: `openvino`, `ov`, `npu`.
    - PyTorch: `pytorch`, `torch`, `pt`, `cuda`, `fx`, `torch.fx`.
    - ONNX: `onnx`, `onnxruntime`, `ort`.
    - If no backend keywords, no prefix.
    - Do not add backend prefixes to the `Requirements` and `General` sections.
4. **Formatting:**
    - Each line: `(Backend) Description (#PR_NUMBER)`
    - Sort subcategories by backend: General first, then OpenVINO, PyTorch, ONNX.
    - If message is unclear, use: `git log --format=%B -n 1 COMMIT_HASH`.
5. **Template:** Update `ReleaseNotes.md` using this structure (use `- ...` for empty sections):

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

## Constraints

- Update only `ReleaseNotes.md`.
- Use only parse_commits.sh for commit lists.
- Use only `git log --format=%B -n 1 COMMIT_HASH` to get commit messages if needed.
