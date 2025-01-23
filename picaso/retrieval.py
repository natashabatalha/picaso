import os 

__refdata__ = os.environ.get('picaso_refdata')

def setup_retrieval(rtype, output_file):
    """
    Creates a Python script template with xarray and a function.

    Parameters
    ----------
    type: str
        will create different kinds of templates for you 
        options: line, grid, free, grid+cloud
    
    output_path : str 
        Path to save the script to. 

    Returns:
        The generated script as a string, or None if saved to a file.
    """
    if rtype=='line':
        input_file = os.path.join(__refdata__, 'scripts','line_retrieval.py')


    with open(input_file, 'r') as f:
        content = f.read()

    # Modify the content by replacing the assignment to var1
    modified_content = content#.replace("var1=3", f"var1={new_value}")

    with open(output_file, 'w') as f:
        f.write(modified_content)

