#!/bin/bash

# Fetch all tags from remote

# Delete all local tags
git tag -l | xargs git tag -d

# Delete all remote tags
git fetch --tags

git tag -l | xargs -n 1 -I {} git push --delete origin {}
# Verify deletion
echo "Local tags:"
git tag -l

echo "Remote tags:"
git ls-remote --tags origin
