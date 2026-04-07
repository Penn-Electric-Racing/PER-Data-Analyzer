Getting Started
===================

Installation
------------------

1. Install the following tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* `Git <https://git-scm.com/downloads>`_
* `Python <https://www.python.org/downloads/>`_ (at least version 3.9)
* `Visual Studio Code <https://code.visualstudio.com/download>`_

  Then, install the following extension packs (Navigate to the Extensions marketplace using the left sidebar in
  VS Code or by pressing ``Ctrl + Shift + X``):

  * Jupyter (Microsoft)
  * Python (Microsoft)


2. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

	# Navigate to a folder of your choice, then run the following in your terminal
	git clone https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git


3. Setup Code Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following commands to setup the development environment:

Note: If you get an error about the python version, you can change the ``python`` to ``python3``
in the commands below.

.. code-block:: bash

	python -m venv .venv

	# For Windows:
	.venv\Scripts\activate

	# For Mac/Linux:
	source .venv/bin/activate

	pip install -r requirements.txt
	pip install -e .
	pre-commit install
	nbstripout --install


4. Run a Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Select the Virtual Environment to use:

   * Press ``Ctrl + Shift + P``
   * Type "Python: Select Interpreter"
   * Choose the ``.venv`` Python interpreter located in the project folder (this is the one you created above).

2. Configure Jupyter Notebooks:

   * Open whichever notebook you wish to run
   * Click the "Kernel" icon (Computer Symbol) in the top right corner
   * Select the ``.venv`` Python interpreter (same one as above)


5. [Optional] Switching Branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Figure out the name of the branch you want to switch to. You can see all branches on the repository's Github page.

2. Open the terminal (in your project folder) and run the following commands:

.. code-block:: bash

	git fetch
	git checkout [name_of_branch]

3. If you have uncommitted changes, Git may force you to deal with them first. Run the following command to stash them
before switching branches:

.. code-block:: bash

	git add .
	git stash

	# If needed, you can return to this branch later and restore your stashed changes with
	git stash pop	# Reach out for help if any issues arise here


6. [Optional] Contributing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a new branch before you begin

.. code-block:: bash

	git checkout -b [name_your_new_branch]

2. Make your changes (remember to run a couple different notebooks to verify it works!)

3. Push your changes to Github

.. code-block:: bash

	# Create a commit. You may do this multiple times as you work on your changes
	git add .
	git commit -m  "[descriptive_commit_message]"

	git push  # If this is your first push, you may need to follow some instructions in the terminal

4. Go to the `PER-Data-Analyzer GitHub <https://github.com/Penn-Electric-Racing/PER-Data-Analyzer>`_ and use the interface
to open a Pull Request for your branch. Add a Software lead as a reviewer.


.. note::

   If you change any files **outside** of the notebooks (.py files), you need to **restart** the runtime
   of the notebook.


Tutorial Notebook
------------------

**This notebook is included in the project repository. You should be able to run it after following the
installation instructions above.**

.. toctree::
   :maxdepth: 1

   notebook
