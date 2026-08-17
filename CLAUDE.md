### Codebase Description

PERDA (PER-Data-Analyzer) is Penn Electric Racing's in-house general purpose Python library for parsing and analyzing log files produced by an FSAE-style electric racecar.

- `perda/analyzer/` contains the Analyzer and RunCollection classes. These are the primary API users interact with to analyze data.

- `perda/core_data_structures` contains SingleRunData and DataInstance Pydantic models, and functions to work with them. These are the fundamental data structures that the `analyzer` module uses to store and manipulate data.

- `perda/utils` contains helper functions that reduce complexity in the main Analyzer by black-boxing commonly data analysis tasks.

- `perda/plotting/` contains functions to generate graphs and plots for data analysis. These functions are designed to be generic and reusable, not tied to specific analysis use cases.

- `notebooks` contains two Jupyter notebooks that demonstrate how to use the library. There should NOT be any implementation code in these notebooks beyond standard API calls.

- `tests` contains unit tests for the library.


### Code Cleanliness Rules

- Always type annotate function return types and parameters. Use __future__ annotations instead of string annotations.

- Never use `TYPE_CHECKING`. If you struggle with circular imports, this is a problem with your code structure. Refactor your code to avoid circular imports.

- AVOID scoped imports. The only justification for scoped imports is to conditionally import a module that is not always available.

- AVOID unconditionally printing to the console.

- If printing for logging purposes, make sure to expose a `verbose` argument to control the printing.

- If printing a summary and representation, prefer Pydantic models with __str__ methods. Let the user decide whether to print the string representation.

- Use `numpy.typing` for array types, e.g. `NDArray[np.float64]` instead of `np.ndarray`.

- Use the Python typing module for generic types, e.g. `list[float]` instead of `List[float]`.

- Do NOT use `Any` type.

- Avoid using `None` as a default value, unless it is truly a meaningful default. If you have a real default, use that instead, and make the typing non-optional.

- Document functions with numpy-style docstrings. Be concise and use understandable language. Avoid unnecessary technical jargon. Avoid long descriptions when none is necessary.

- Do NOT leave module level docstrings. If an explanation is very necessary, leave it in the 'Notes' section of relevant function docstrings.

- Do NOT have long comments in the code. If such comments are necessary, they should be in the 'Notes' section of the relevant function docstring.

- For short and simple helper functions, especially internal or underscored functions, you may omit docstrings if you can name the function and its parameters clearly enough to be self-documenting.


### Codebase Organization Rules

- Whenever defining a fixed collection of data, use a Pydantic V2 model, with Field functions. Do NOT use dictionaries with fixed keys, dataclasses, or named tuples.

- Prefer constructors to have fixed, mandatory arguments. If it is necessary to have alternate ways to construct an object, use separate factory functions in a dedicated file, e.g. `run_collection_factory.py`.

- AVOID making Pydantic fields optional. Strict typing and validation is preferred.

- Do NOT use @property decorators unless you want easy access to fields that are meant to be fully dependent on other fields. Do NOT write verbose getters and setters. Directly access parameters through the Pydantic models.

- Prefer longer and more descriptive variable names over short and cryptic ones that use abbreviations or acronyms.

- Prefer small, single-purpose functions over large, multi-purpose functions. If a function is getting too long, break it up into smaller functions.

- Keep tests and test YAMLs up to date with the codebase.

- Whenever there is a sensible default value for an argument, provide it in the function signature instead of assigning it later in the function body.

- Separate graphs and logic. Try to make graphing utilities generic, reusable, and modularized in `plotting`. `plotting` should be free of any application-specific logic. `analyzer` should be responsible for wrapping the generic plotting functions in an application-specific layer.

- No magic numbers in code. Define constants as named, uppercase module-level variables. Constants specific to a module should be defined at the top of that module. Constants that are shared across modules should be defined in `constants.py`. `units.py` reserved for specific constants fitting that description.
