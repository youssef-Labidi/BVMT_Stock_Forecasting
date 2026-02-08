#!/usr/bin/env bash
# push_subfolder.sh <remote.git.url> <subfolder> <branch>
# Example: ./push_subfolder.sh https://github.com/Homssalomssa/bvmt-sentiment-analysis.git bvmt-forecasting add-bvmt-forecasting

set -e
REMOTE="$1"
SUBFOLDER="$2"
BRANCH="${3:-add-bvmt-forecasting}"

if [ -z "$REMOTE" ] || [ -z "$SUBFOLDER" ]; then
  echo "Usage: $0 <remote.git.url> <subfolder> [branch]"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree not clean. Please commit or stash changes first."
  exit 1
fi

git checkout -b "$BRANCH"
mkdir -p "$SUBFOLDER"

# Move tracked files into subfolder
while IFS= read -r file; do
  if [ "$file" = "$0" ]; then
    continue
  fi
  mkdir -p "$(dirname "$SUBFOLDER/$file")"
  git mv "$file" "$SUBFOLDER/$file" || echo "Skipped $file"
done < <(git ls-files)

git add .
git commit -m "Move project into subfolder $SUBFOLDER"

git remote add target "$REMOTE" 2>/dev/null || true
git push target "$BRANCH"

echo "Done. Open a PR from $BRANCH on the target repo."
