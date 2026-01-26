# PER-Data-Analyzer

PER's in-house library for log parsing and data analysis.


## Installation

**Option 1**: Developer Installation:
    
Also useful if you would like to directly use the Tutorial notebook.

1. Clone the repository:

    ```bash
    git clone https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git
    ```

2. Setup the development environment:

    ```bash
    cd PER-Data-Analyzer
    python -m venv .venv
    .\.venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux

    pip install -r requirements.txt
    pip install -e .
    ```

**Option 2**: Direct Installation

Not easy to modify and iterate on the source code, but useful for quick setup and utilizing the library in external environments like Google Colab.

1. Install via pip:

`pip install git+https://github.com/Penn-Electric-Racing/PER-Data-Analyzer.git@main`


## Code Demo
See [Tutorial.ipynb](notebooks/Tutorial.ipynb)
