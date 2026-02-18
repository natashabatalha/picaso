
.. _notebook_workflow:

Tutorial & Notebook Workflow
============================

PICASO has transitioned its tutorial notebooks from the standard `.ipynb` format to Jupytext-compatible `.py` files. This change helps us maintain a cleaner codebase, reduce repository size, and improve version control tracking of changes in the tutorials.

While you can view these tutorials directly in our online documentation, you may want to run them locally on your own machine. This page explains how to get the tutorial files and how to work with them using Jupytext.

Getting the Tutorials
---------------------

There are two main ways to get the tutorial `.py` files from our GitHub repository:

1. **Clone the Repository (Recommended for Contributors):**
   If you plan to contribute to PICASO or want to stay up-to-date with the latest changes, you can clone the entire repository:

   .. code-block:: bash

      git clone https://github.com/natashabatalha/picaso.git

   The tutorials are located in the ``docs/notebooks/`` directory.

2. **Download from GitHub:**
   If you just want a few tutorials, you can navigate to the ``docs/notebooks/`` directory on `GitHub <https://github.com/natashabatalha/picaso/tree/master/docs/notebooks>`_, click on the desired ``.py`` file, and use the "Download raw file" button.

Using Jupytext to Run Tutorials
-------------------------------

`Jupytext <https://jupytext.readthedocs.io/>`_ is a Jupyter extension that allows you to open and run Python scripts as if they were Jupyter notebooks.

Installing Jupytext
^^^^^^^^^^^^^^^^^^^

To use Jupytext, you first need to install it in your Python environment:

.. code-block:: bash

   pip install jupytext

Opening `.py` files in Jupyter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once Jupytext is installed, you can open the tutorial ``.py`` files directly in Jupyter:

* **JupyterLab:** Right-click the ``.py`` file in the file browser and select **Open With** -> **Notebook**.
* **Jupyter Notebook (Classic):** The ``.py`` files will appear with a notebook icon. Simply clicking them will open them as notebooks.

When you save the notebook, Jupytext will automatically update the ``.py`` file with your changes, including the output if configured, while keeping the clean Python script format.

Using Jupytext in VS Code
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer using **VS Code**, you can also work with Jupytext files seamlessly:

1. Install the **Jupyter** extension in VS Code.
2. Install the **Jupytext** extension from the VS Code Marketplace (published by *Don Jayamanne*).
3. Once installed, you can right-click any ``.py`` file in the file explorer and select **Open With...** -> **Jupytext Notebook**.

This allows you to enjoy the full VS Code notebook experience (including IntelliSense and debugging) while working directly with the Jupytext ``.py`` files.

Converting to `.ipynb`
^^^^^^^^^^^^^^^^^^^^^^

If you prefer working with standard `.ipynb` files, you can easily convert the tutorial scripts using the Jupytext command-line tool:

.. code-block:: bash

   jupytext --to notebook your_tutorial.py

This will create a ``your_tutorial.ipynb`` file that you can use as a normal notebook.

Why `.py` instead of `.ipynb`?
------------------------------

* **Better Version Control:** Changes in code are easier to track with ``git diff``.
* **Smaller Repository:** No bulky cell outputs or metadata are stored in the repository.
* **Ease of Use:** You can easily edit the tutorials in any text editor or IDE while still having the interactive experience of a notebook.
