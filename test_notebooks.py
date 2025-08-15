#!/usr/bin/env python

import os
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("Please install nbconvert and nbformat: pip install nbconvert nbformat")
    sys.exit(1)

def run_notebook(notebook_path):
    """Executes a notebook and returns True if it runs without errors, False otherwise."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Set the picaso_refdata environment variable
    os.environ['picaso_refdata'] = '/reference/'

    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        return True
    except Exception as e:
        print(f"Error executing notebook {notebook_path}:")
        print(e)
        return False

def main():
    notebook_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'notebooks')
    failed_notebooks = []

    for root, _, files in os.walk(notebook_dir):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                # Exclude WIP notebooks
                if 'WIP' in notebook_path:
                    print(f"Skipping WIP notebook: {notebook_path}")
                    continue
                print(f"Running notebook: {notebook_path}")
                if not run_notebook(notebook_path):
                    failed_notebooks.append(notebook_path)

    if failed_notebooks:
        print("\nFailed notebooks:")
        for notebook in failed_notebooks:
            print(notebook)
        sys.exit(1)
    else:
        print("\nAll notebooks ran successfully!")
        sys.exit(0)

if __name__ == '__main__':
    main()
