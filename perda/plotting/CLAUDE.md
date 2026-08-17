`plotting` is a submodule focused on offering a generic and consistent API for generating plots and visualizations.

- NO SPECIFIC FUNTIONALITY OR LOGIC in this module. Callers are responsible for wrapping the generics in an application-specific layer.

- The inputs to the plotting functions should be generic objects, such as numpy arrays, dictionaries, and lists. The one exception is for DataInstance, which we consider to be an atomic data type representing one time-series variable for plotting purposes.

- Graphs should always be generated using Plotly

- All plotting functions should NOT render plots directly. Rather, they should return a Plotly figure object. Downstream users of the plotting library can then make custom modifications and control image rendering.

- Graphing APIs should expose arguments that allow customization of important parts of the graph's appearance

- Generic formatting and styling of the plots should be controlled with configuration objects.

- These configuration objects should be strongly typed Pydantic V2 models. Keep the configuration objects in `plotting_constants.py`.

- All config objects should have sensible default values for all fields. Keep the defaults as module-level constants in `plotting_constants.py`.

- Always try to reuse the fields in existing configuration objects. It is okay to have the same field in a config object be used slightly differently in different contexts, however be consistent whenever possible. Only create new config objects if there if your requirements have no overlap with existing config objects.

- Each type of plot should live in its own descriptively-named file. In general, each file should contain one plotting function and its specific helpers.
