#!/usr/bin/env python

import os
import subprocess
import sys
#sys.path.append(os.path.join('docs', 'notebooks'))


"""
Primarily used for internal integration testing of the notebooks
"""

try:
    import nbformat
    import jupytext
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print("Please install jupytext, nbconvert and nbformat: pip install jupytext nbconvert nbformat")
    sys.exit(1)

def run_notebook(notebook_path,
                 github=False,picaso_refdata=None,PYSYN_CDBS=None, picaso_code = None, virga_code=None):
    """Executes a notebook and returns True if it runs without errors, False otherwise."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        if notebook_path.endswith('.py'):
            nb = jupytext.read(f)
        else:
            nb = nbformat.read(f, as_version=4)

        # Insert a code cell at the beginning to append the path if using a local github picaso installation
        if github == True:
            path_cell = nbformat.v4.new_code_cell(f'import sys; sys.path.append("{picaso_code}")')
            virga_cell = nbformat.v4.new_code_cell(f'import sys; sys.path.append("{virga_code}")')
            nb.cells.insert(0, virga_cell)
            nb.cells.insert(0, path_cell)

            path_cell = nbformat.v4.new_code_cell(f'import os; os.environ["picaso_refdata"] = "{picaso_refdata}"')
            cdbs_cell = nbformat.v4.new_code_cell(f'import os; os.environ["PYSYN_CDBS"] = "{PYSYN_CDBS}"')
            nb.cells.insert(0, cdbs_cell)
            nb.cells.insert(0, path_cell)

    # # Set the picaso_refdata environment variable
    # os.environ['picaso_refdata'] = '/reference/'

    ep = ExecutePreprocessor(timeout=3000, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        return True
    except Exception as e:
        print(f"Error executing notebook {notebook_path}:")
        print(e)
        return False
    
import argparse
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("notebook_dir", type=str, help="Directory path to the notebooks you want to test")
    parser.add_argument("--local", type=bool, default=False, help="Fresh github isntall witout env variables")
    parser.add_argument("--picaso_refdata", type=str, default=os.getenv('picaso_refdata'), help="PICASO refdata path")
    parser.add_argument("--PYSYN_CDBS", type=str, default=os.getenv('PYSYN_CDBS'), help="PYSYN_CDBS refdata path") 
    parser.add_argument("--picaso_code", type=str, default=None, help="PICASO Code path") 
    parser.add_argument("--virga_code", type=str, default=None, help="Virga Code path") 

    args = parser.parse_args()

    notebook_dir = args.notebook_dir
    gitlocal = args.local
    picaso_refdata = args.picaso_refdata
    PYSYN_CDBS = args.PYSYN_CDBS
    picaso_code = args.picaso_code
    virga_code = args.virga_code

    failed_notebooks = []

    for root, _, files in os.walk(notebook_dir):
        for file in files:
            if file.endswith('.ipynb') or file.endswith('.py'):
                notebook_path = os.path.join(root, file)
                # Exclude WIP notebooks
                if 'WIP' in notebook_path:
                    print(f"Skipping WIP notebook: {notebook_path}")
                    continue
                elif 'Quickstart' in notebook_path:
                    print(f"Skipping Quickstart notebook: {notebook_path}")
                    continue
                # option to exclude workshop notebooks 
                # elif 'workshop' in notebook_path:
                #     print(f"Skipping workshop notebook: {notebook_path}")
                #     continue
                elif '.ipynb_checkpoint' in notebook_path: 
                    continue
                print(f"Running notebook: {notebook_path}")
                if not run_notebook(notebook_path, github=gitlocal,picaso_refdata=picaso_refdata,PYSYN_CDBS=PYSYN_CDBS,picaso_code=picaso_code,virga_code=virga_code):
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
