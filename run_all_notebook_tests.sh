#!/bin/bash

# This script runs the notebook tests for all subdirectories of docs/notebooks.

NOTEBOOK_ROOT="docs/notebooks"
EXCLUDED_DIR="workshops"
FAILED_DIRS=()

# Find all immediate subdirectories of the notebook root
for dir in "$NOTEBOOK_ROOT"/*/; do
    # Remove trailing slash
    dir=${dir%/}
    # Get the base directory name
    base_dir=$(basename "$dir")

    # Skip the excluded directory
    if [ "$base_dir" == "$EXCLUDED_DIR" ]; then
        echo "Skipping directory: $dir"
        continue
    fi

    echo "--------------------------------------------------"
    echo "Testing notebooks in: $dir"
    echo "--------------------------------------------------"

    # Run the python test script on the directory
    python test_notebooks.py "$dir"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Tests failed in directory: $dir"
        FAILED_DIRS+=("$dir")
    else
        echo "Tests passed in directory: $dir"
    fi
done

echo "--------------------------------------------------"
echo "Summary"
echo "--------------------------------------------------"

if [ ${#FAILED_DIRS[@]} -eq 0 ]; then
    echo "All notebook tests passed!"
else
    echo "The following directories have failing notebook tests:"
    for failed_dir in "${FAILED_DIRS[@]}"; do
        echo "  - $failed_dir"
    done
    exit 1
fi
