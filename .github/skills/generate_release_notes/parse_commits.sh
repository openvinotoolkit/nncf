#!/bin/bash
set -euxo pipefail

TMP_BRANCHES_FILE="/tmp/branches_tmp.txt"
TMP_COMMITS="/tmp/commits_tmp.txt"

# 1. Get the two most recent release branches from origin
git fetch origin 'refs/heads/release_v*:refs/remotes/origin/release_v*'
git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/**/release_v* | head -n 2 > "$TMP_BRANCHES_FILE"

# 2. Assign them to variables (Current is the 1st, Previous is the 2nd)
LAST=$(sed -n '1p' "$TMP_BRANCHES_FILE")
PREV=$(sed -n '2p' "$TMP_BRANCHES_FILE")
rm "$TMP_BRANCHES_FILE"

echo "Comparing $PREV to $LAST..."

# 3. Generate the log and filter out dependabot in one pipeline
git log "$PREV..$LAST" --pretty=format:"%an;%h;%s" | grep -v "dependabot" > "$TMP_COMMITS"

# 4. Output the collected commits
echo "Output saved to $TMP_COMMITS"

echo "Collected commits for $LAST:"
cat "$TMP_COMMITS"
