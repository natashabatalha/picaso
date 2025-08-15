#!/usr/bin/env python

import os
import subprocess
import sys
sys.path.append(os.path.join('docs', 'notebooks'))

local_github = True

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("Please install nbconvert and nbformat: pip install nbconvert nbformat")
    sys.exit(1)

def run_notebook(notebook_path, github=False):
    """Executes a notebook and returns True if it runs without errors, False otherwise."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

        # Insert a code cell at the beginning to append the path if using a local github picaso installation
        if github == True:
            path_cell = nbformat.v4.new_code_cell('import sys; sys.path.append("/Users/jjm6243/dev_picaso/")')
            virga_cell = nbformat.v4.new_code_cell('import sys; sys.path.append("/Users/jjm6243/Documents/virga/")')
            nb.cells.insert(0, virga_cell)
            nb.cells.insert(0, path_cell)

            path_cell = nbformat.v4.new_code_cell('import os; os.environ["picaso_refdata"] = "/Users/jjm6243/dev_picaso/reference"')
            cdbs_cell = nbformat.v4.new_code_cell('import os; os.environ["PYSYN_CDBS"] = "/Users/jjm6243/dev_picaso/reference/stellar_grids"')
            nb.cells.insert(0, cdbs_cell)
            nb.cells.insert(0, path_cell)

    # # Set the picaso_refdata environment variable
    # os.environ['picaso_refdata'] = '/reference/'

    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        return True
    except Exception as e:
        print(f"Error executing notebook {notebook_path}:")
        print(e)
        return False

def main():
    notebook_dir = os.path.join('docs', 'notebooks')
    failed_notebooks = []

    for root, _, files in os.walk(notebook_dir):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                # Exclude WIP notebooks
                if 'WIP' in notebook_path:
                    print(f"Skipping WIP notebook: {notebook_path}")
                    continue
                elif 'Quickstart' in notebook_path:
                    print(f"Skipping Quickstart notebook: {notebook_path}")
                    continue
                elif 'Reference' in notebook_path:
                    print(f"Skipping Reference notebook: {notebook_path}")
                    continue
                print(f"Running notebook: {notebook_path}")
                if not run_notebook(notebook_path, github=local_github):
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
