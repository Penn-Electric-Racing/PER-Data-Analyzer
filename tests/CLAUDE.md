- Organize tests by files. Each test_* file should correspond to a single file in the codebase.

- When possible, parameterize tests. Increase the number of test cases without increasing the number of functions.

- Use fixtures to set up common test data or state. Use fixture.py file per test folder to keep files smaller and more maintainable. Only put widely used fixtures in the top-level conftest.py.

- Do not make section header comments in test files