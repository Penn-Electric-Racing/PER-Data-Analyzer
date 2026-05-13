Getting Started
===================

Installation
------------------


Option 1: Developer Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the installation method for anyone who wants to contribute to the project or has special use cases. We also recommend
this method if you prefer to work locally on your machine instead of using Google Colab.


1. Ensure you have the following tools
""""""""""""""""""""""""""""""""""""""""""""""""""
* `Git <https://git-scm.com/downloads>`_
* `Python <https://www.python.org/downloads/>`_ (at least version 3.9)
* `Visual Studio Code <https://code.visualstudio.com/download>`_

  Then, install the following extension packs (Navigate to the Extensions marketplace using the left sidebar in
  VS Code or by pressing ``Ctrl + Shift + X``):

  * Jupyter (Microsoft)
  * Python (Microsoft)


2. Clone the repository
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

	# Navigate to a folder of your choice, then run the following in your terminal
	git clone https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git


3. Setup Code Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Run the following commands to setup the development environment:

Note: If you get an error about the python version, you can change the ``python`` to ``python3``
in the commands below.

.. code-block:: bash

	cd PER-Data-Analyzer
	python -m venv .venv

	# For Windows:
	.venv\Scripts\activate

	# For Mac/Linux:
	source .venv/bin/activate

	pip install -r requirements-dev.txt

	pre-commit install
	nbstripout --install


4. Run a Notebook
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Select the Virtual Environment to use:

   * Press ``Ctrl + Shift + P``
   * Type "Python: Select Interpreter"
   * Choose the ``.venv`` Python interpreter located in the project folder (this is the one you created above).

2. Configure Jupyter Notebooks:

   * Open whichever notebook you wish to run
   * Click the "Kernel" icon (Computer Symbol) in the top right corner
   * Select the ``.venv`` Python interpreter (same one as above)


.. note::

   If you change any files **outside** of the notebooks (.py files), you need to **restart** the runtime
   of the notebook.

.. tip::

   Two tutorial notebooks are included in the repository. You should be able to run both after following
   the installation instructions above.


.. toctree::
   :maxdepth: 1

   notebook.nblink
   notebook_advanced.nblink
   


Option 2: Standard Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not as easy to modify and iterate on the source code, but useful for quick setup and utilizing the
library in external environments like Google Colab.

1. Install via pip
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

	pip install "perda[notebook] @ git+https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git@main"


.. note::
	You can embed the above command in a Google Colab notebook to automatically install when you run the notebook.
	Append an exclamation mark and leave it in a normal code cell like below:

	.. code-block:: python

		!pip install "perda[notebook] @ git+https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git@main"


2. Optional: Enable semantic search
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The default installation supports keyword-based search. To also enable semantic (AI-powered) search,
install the ``full`` extra instead. This pulls in ``sentence-transformers`` and its dependencies
(including PyTorch), so expect a larger download.

.. code-block:: python

	!pip install "perda[full] @ git+https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git@main"

When the semantic model is available, :func:`perda.utils.search.search` automatically uses it.
If it is not installed, search falls back to keyword-only scoring with no error.