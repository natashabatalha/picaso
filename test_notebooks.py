import os
import subprocess
import sys
import logging

def test_single_notebook(notebook_path, timeout_seconds=30):
    """
    Tests a single Jupyter notebook.
    """
    logging.info(f'Testing notebook: {notebook_path}')
    try:
        result = subprocess.run(
            [
                'jupyter',
                'nbconvert',
                '--to',
                'notebook',
                '--execute',
                '--inplace',
                notebook_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        logging.info(f'SUCCESS: {notebook_path} executed successfully.')
        return True
    except subprocess.TimeoutExpired as e:
        logging.error(f'FAILURE: {notebook_path} timed out after {timeout_seconds} seconds.')
        if e.stdout:
            logging.error(f"stdout:\n{e.stdout}")
        if e.stderr:
            logging.error(f"stderr:\n{e.stderr}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f'FAILURE: {notebook_path} failed to execute.')
        logging.error(f"stdout:\n{e.stdout}")
        logging.error(f"stderr:\n{e.stderr}")
        return False

def test_notebooks_in_directory(target_dir):
    """
    Finds and tests all Jupyter notebooks in the specified directory.
    """
    logging.info(f"Starting to test notebooks in: {target_dir}")
    excluded_dir = os.path.join('docs', 'notebooks', 'workshops')
    has_failures = False

    for root, _, files in os.walk(target_dir):
        if root.startswith(excluded_dir):
            continue

        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                if not test_single_notebook(notebook_path):
                    has_failures = True

    if has_failures:
        logging.error(f'Some notebooks in {target_dir} failed to execute.')
        return False
    else:
        logging.info(f'All notebooks in {target_dir} executed successfully.')
        return True

if __name__ == '__main__':
    logging.basicConfig(
        filename='notebook_test.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',  # Overwrite the log file on each run
    )

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        logging.info(f"Target path from command line: {target_path}")
        if os.path.isfile(target_path):
            if not test_single_notebook(target_path):
                sys.exit(1)
        elif os.path.isdir(target_path):
            if not test_notebooks_in_directory(target_path):
                sys.exit(1)
        else:
            logging.error(f"Invalid path: {target_path}")
            sys.exit(1)
    else:
        logging.error("Usage: python test_notebooks.py <path_to_notebook_or_directory>")
        sys.exit(1)
