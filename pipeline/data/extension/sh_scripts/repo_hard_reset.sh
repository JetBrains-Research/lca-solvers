#!/bin/bash

cd /Users/Evgeniy.Glukhov/Datasets/lca/kotlin/permissive_repos

# Get the current directory
parent_dir=$(pwd)

# Loop through all directories
for repo in */; do
    # Check if the directory contains a .git folder (confirming it's a git repo)
    if [ -d "$repo/.git" ]; then
        echo "Processing $repo"

        # Navigate into the repository
        cd "$repo"

        # Check if index.lock exists and delete it
        if [ -f ".git/index.lock" ]; then
            echo "Deleting .git/index.lock in $repo"
            rm .git/index.lock
        fi

        # Perform git reset --hard
        echo "Resetting $repo"
        git reset --hard

        # Clean untracked files and directories
        echo "Cleaning untracked files and directories in $repo"
        git clean -fdx

        # Go back to the parent directory
        cd "$parent_dir"
    else
        echo "$repo is not a git repository, skipping..."
    fi
done

echo "All repositories processed."
