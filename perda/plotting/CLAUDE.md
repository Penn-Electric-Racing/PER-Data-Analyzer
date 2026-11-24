`plotting` is a submodule focused on offering a generic and consistent API for generating plots and visualizations.

- NO SPECIFIC FUNTIONALITY OR LOGIC in this module. Callers are responsible for wrapping the generics in an application-specific layer.

- If some new analysis feature requires a plot, create the plot with existing functions and make custom modifications to the returned figure. Only modify the existing functions if the new feature requirements seems important and reusable. Only write new plotting functions if there is really no suitable existing function to use.

- The inputs to the plotting functions should be generic objects, such as numpy arrays, dictionaries, and lists. The one exception is for DataInstance, which we consider to be an atomic data type representing one time-series variable for plotting purposes.

- Graphs should always be generated using Plotly

- All plotting functions should NOT render plots directly. Rather, they should return a Plotly figure object. Downstream users of the plotting library can then make custom modifications and control image rendering.

- Only expose options that are important, commonly changed, or tricky to change after the figure is built. Examples: titles, axis labels, log scaling (`log_x`, `log_y`), figure size, subplot spacing, outlier thresholds, bucket/bin sizes. Everything else (fonts, colors, margins) is hardcoded from `defaults.py`; users change it by modifying the returned figure.

- `defaults.py` holds all default values as uppercase module-level constants. Use these constants as the default values in function signatures and for hardcoded styling. Reuse existing constants and argument names across plotting functions whenever possible.

- Write docstrings for both human and agentic users: describe every argument in plain language, including its units, accepted values, and default. Longer docstrings are fine.

- Each type of plot should live in its own descriptively-named file. In general, each file should contain one plotting function and its specific helpers.
