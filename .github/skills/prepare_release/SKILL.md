---
name: prepare-release
description: Run NNCF pre-release activities - create the release branch, switch the release branch to the OpenVINO release candidate, trigger conformance and examples workflows, open the release notes PR, and bump the version in develop.
---

# Prepare NNCF release

## Context

**Files:**

- `src/nncf/version.py` - `__version__ = "X.Y.Z"`, the version being released.
- `constraints.txt`, `**/requirements*.txt`, `pyproject.toml` - OpenVINO pins.
- `docs/Installation.md` - "Corresponding versions" table.
- `.github/skills/prepare_release/set_openvino_version.sh` - helper that rewrites all OpenVINO pins.
- `.github/skills/generate_release_notes/SKILL.md` - release notes skill used in step 4.
- `.github/pull_request_template.md` - body template for every PR.

**Conventions:**

- `origin` is `openvinotoolkit/nncf`; PRs are opened from the user's fork (`FORK` remote, find it with
  `git remote -v`, ask the user if ambiguous).
- `VERSION` is the value of `__version__` on `develop`, e.g. `3.5.0`.
  `SHORT` is `VERSION` without dots, e.g. `350`.
- Release branch: `release_v<SHORT>`, e.g. `release_v350`.
- Branches created from the release branch: `r<SHORT>/<short_desc>`, e.g. `r350/ov_2026_5_0rc1`,
  `r350/release_notes`.
- PR titles targeting a release branch must start with `[release_v<SHORT>]`.
  The OpenVINO PR title format is `[release_v${SHORT}] OpenVINO ${OV_VERSION}`.
- Nightly OpenVINO wheel index: `https://storage.openvinotoolkit.org/simple/wheels/nightly`.
- Previous release PRs to mirror: `Bump Version to 3.5.0` (#4213), `[release_v340] Openvino 2026.4.0rc2` (#4211),
  `Release notes for NNCF v3.4.0` (#4214).

## Execution Logic

Confirm with the user before every `git push` and `gh pr create`. Stop and report if any step fails.

### Step 0 - Sync develop

```bash
git fetch origin develop
git switch develop
git reset --keep origin/develop
VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' src/nncf/version.py)
SHORT=${VERSION//./}
```

Report `VERSION` and `SHORT` to the user and ask for confirmation. Also ask for the Jira ticket number of
the release; `TICKET` is `CVS-<num>` (e.g. `CVS-12345`) and goes into the `### Related tickets` section of
every PR body in this skill.

### Step 1 - Create the release branch

```bash
git switch -c release_v${SHORT} origin/develop
git push origin release_v${SHORT}
```

The branch must exist on `origin` before step 4: the release notes skill compares the two newest
`origin/release_v*` branches.

### Step 2 - Switch the release branch to the OpenVINO release candidate

1. Find the latest RC on the nightly index; `OV_VERSION` is the last line:

    ```bash
    curl -sL https://storage.openvinotoolkit.org/simple/wheels/nightly/openvino/ \
      | grep -oE 'openvino-[0-9]+\.[0-9]+\.[0-9]+rc[0-9]+' | sort -uV | sed 's/openvino-//' | tail -n 1
    ```

    If no RC is published yet, report it and ask the user which version to use.

2. Apply the pin everywhere:

    ```bash
    git switch -c r${SHORT}/ov_${OV_VERSION//./_} release_v${SHORT}
    .github/skills/prepare_release/set_openvino_version.sh ${OV_VERSION}
    ```

    The script rewrites `openvino==` in `constraints.txt` and every tracked `requirements*.txt`, adds
    `--extra-index-url <nightly> --pre` after each pin and adds a `[tool.uv]` block to `pyproject.toml`.
    Do not edit `.github/workflows/schedule_weekly_ov_dev.yml`.

3. In `docs/Installation.md` replace the `develop` row of the "Corresponding versions" table with the
    release row. Use the final OpenVINO version (RC suffix stripped) and keep the other columns:

    ```txt
    | `develop` | `2026.4.1` | ...  ->  | `3.5.0`   | `2026.5.0` | ...
    ```

4. Review `git --no-pager diff`, then commit and open the PR against the release branch:

    ```bash
    git commit -am "[release_v${SHORT}] OpenVINO ${OV_VERSION}"
    git push FORK r${SHORT}/ov_${OV_VERSION//./_}
    gh pr create -R openvinotoolkit/nncf -B release_v${SHORT} -H <fork_owner>:r${SHORT}/ov_${OV_VERSION//./_} \
      -t "[release_v${SHORT}] OpenVINO ${OV_VERSION}" -F <body_file>
    ```

    PR body (follow `.github/pull_request_template.md`): Changes - `openvino==OV_VERSION`, Reason - `Release`,
    Related tickets - `TICKET`. Tests - empty, workflow links are added automatically; do not add them manually.

### Step 3 - Run conformance and examples workflows on the PR

`PR_NUM` is the number returned by `gh pr create` in step 2.

```bash
gh workflow run conformance_ptq.yml -R openvinotoolkit/nncf -f pr_num=${PR_NUM}
gh workflow run conformance_weight_compression.yml -R openvinotoolkit/nncf -f pr_num=${PR_NUM}
gh workflow run examples.yml -R openvinotoolkit/nncf -f pr_num=${PR_NUM}
gh run list -R openvinotoolkit/nncf --limit 5 --json databaseId,name,url,status
```

Do not wait for the runs to finish.

### Step 4 - Release notes PR

```bash
git switch -c r${SHORT}/release_notes release_v${SHORT}
```

Follow `.github/skills/generate_release_notes/SKILL.md` to update `ReleaseNotes.md`, then:

```bash
git commit -am "Release notes for NNCF v${VERSION}"
git push FORK r${SHORT}/release_notes
gh pr create -R openvinotoolkit/nncf -B release_v${SHORT} -H <fork_owner>:r${SHORT}/release_notes \
  -t "[release_v${SHORT}] Release notes for NNCF v${VERSION}" -F <body_file>
```

PR body: Changes - `Release notes for NNCF vVERSION`, Reason - `Release`, Related tickets - `TICKET`,
Tests - `N/A`.

### Step 5 - Bump version in develop

Default next version: increase the minor part and reset patch (`3.5.0` -> `3.6.0`). Ask the user if a
different next version is required.

```bash
git switch -c bump_nncf_version_${SHORT} origin/develop
sed -i "s/^__version__ = \".*\"/__version__ = \"${NEXT_VERSION}\"/" src/nncf/version.py
git commit -am "Bump Version to ${NEXT_VERSION}"
git push FORK bump_nncf_version_${SHORT}
gh pr create -R openvinotoolkit/nncf -B develop -H <fork_owner>:bump_nncf_version_${SHORT} \
  -t "Bump Version to ${NEXT_VERSION}" -F <body_file>
```

PR body: Changes - `Update NNCF version to NEXT_VERSION`, Reason - `Release`,
Related tickets - `TICKET`, Tests - `N/A`.

## Constraints

- Never push to `origin/develop`; only the release branch is pushed to `origin` directly (step 1).
- Never force-push.
- Ask before each push and PR creation; ask before resetting a local `develop` that has unpushed commits.
- Only edit the files listed in Context; do not touch unrelated pins.
- Requires `gh` authenticated with write access to `openvinotoolkit/nncf` and network access to
  `storage.openvinotoolkit.org`.
