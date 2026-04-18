# perda.plotting.data_instance_plotter

### perda.plotting.data_instance_plotter.plot_dual_axis(left_data_instances, right_data_instances, title=None, left_y_axis_title=None, right_y_axis_title=None, show_legend=True, font_config=FontConfig(large=20, medium=14, small=10), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), timestamp_unit=Timescale.MS, vlines=None, vline_config=VLineConfig(color='gray', width=2, dash='dash', opacity=0.7))

Plot DataInstances on dual y-axes using Plotly.

* **Parameters:**
  * **left_data_instances** (*List* *[*[*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *]*) – List of DataInstance objects to plot on the left y-axis
  * **right_data_instances** (*List* *[*[*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *]*) – List of DataInstance objects to plot on the right y-axis
  * **title** (*str* *|* *None* *,* *optional*) – Title for the entire plot.
  * **left_y_axis_title** (*str* *|* *None* *,* *optional*) – Label for the left y-axis.
  * **right_y_axis_title** (*str* *|* *None* *,* *optional*) – Label for the right y-axis.
  * **show_legend** (*bool* *,* *optional*) – Whether to show plot legends. Default is True
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*) – Font configuration for plot elements. Default is DEFAULT_FONT_CONFIG
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*) – Layout configuration for plot dimensions. Default is DEFAULT_LAYOUT_CONFIG
  * **timestamp_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Timestamp unit in the underlying data. Converted to seconds for x-axis display.
  * **vlines** (*List* *[**float* *]*  *|* *None* *,* *optional*) – X-axis positions (in seconds) where vertical lines are drawn. Default is None.
  * **vline_config** ([*VLineConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.VLineConfig) *,* *optional*) – Visual configuration for the vertical lines. Default is DEFAULT_VLINE_CONFIG.
* **Return type:**
  go.Figure

### Examples

```pycon
>>> fig = plot_dual_axis([speed_di], [torque_di], left_y_axis_title="Speed (mph)", right_y_axis_title="Torque (Nm)")
>>> fig.show()
```

### perda.plotting.data_instance_plotter.plot_single_axis(data_instances, title=None, y_axis_title=None, show_legend=True, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10), timestamp_unit=Timescale.MS, vlines=None, vline_config=VLineConfig(color='gray', width=2, dash='dash', opacity=0.7))

Plot one or more DataInstances on a single y-axis using Plotly.

* **Parameters:**
  * **data_instances** (*List* *[*[*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *]*) – List of DataInstance objects to plot
  * **title** (*str* *|* *None* *,* *optional*) – Title for the entire plot.
  * **y_axis_title** (*str* *|* *None* *,* *optional*) – Label for the y-axis.
  * **show_legend** (*bool* *,* *optional*) – Whether to show plot legends. Default is True
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
  * **timestamp_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Timestamp unit in the underlying data. Converted to seconds for x-axis display.
  * **vlines** (*List* *[**float* *]*  *|* *None* *,* *optional*) – X-axis positions (in seconds) where vertical lines are drawn. Default is None.
  * **vline_config** ([*VLineConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.VLineConfig) *,* *optional*) – Visual configuration for the vertical lines. Default is DEFAULT_VLINE_CONFIG.
* **Return type:**
  go.Figure

### Examples

```pycon
>>> fig = plot_single_axis([speed_di, torque_di], title="Speed & Torque", y_axis_title="Value")
>>> fig.show()
```
