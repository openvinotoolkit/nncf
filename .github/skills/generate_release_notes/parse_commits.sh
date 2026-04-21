#!/bin/bash
set -euxo pipefail

# 1. Get the two most recent release branches from origin
git fetch origin 'refs/heads/release_v*:refs/remotes/origin/release_v*'

# 2. Assign them to variables (Current is the 1st, Previous is the 2nd)
LAST=$(git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/remotes/origin/release_v* | head -n 1)
PREV=$(git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/remotes/origin/release_v* | head -n 2 | tail -n 1)

echo "Comparing $PREV to $LAST" >&2

# 3. Generate the log and filter out dependabot in one pipeline
echo "Collected commits for $LAST:" >&2
git --no-pager log "$PREV..$LAST" --pretty=format:"%an;%h;%s" --grep="dependabot" --invert-grep || true
