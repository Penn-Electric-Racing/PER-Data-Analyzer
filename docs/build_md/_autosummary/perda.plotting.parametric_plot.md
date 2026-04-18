# perda.plotting.parametric_plot

### perda.plotting.parametric_plot.plot_parametric_curve(x, y, x_label='X', y_label='Y', title=None, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10))

Plot a 2D parametric curve as a static Plotly figure.

* **Parameters:**
  * **x** (*NDArray* *[**float64* *]*) – X-axis values.
  * **y** (*NDArray* *[**float64* *]*) – Y-axis values. Must have the same length as x.
  * **x_label** (*str* *,* *optional*) – X-axis label.
  * **y_label** (*str* *,* *optional*) – Y-axis label.
  * **title** (*str* *|* *None* *,* *optional*) – Plot title.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
* **Return type:**
  go.Figure

### Examples

```pycon
>>> fig = plot_parametric_curve(lon_arr, lat_arr, x_label="Longitude", y_label="Latitude")
>>> fig.show()
```

### perda.plotting.parametric_plot.plot_parametric_curve_square(x, y, x_label='X', y_label='Y', title=None, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10))

Plot a 2D parametric curve with equal axes and a square aspect ratio.

Automatically computes axis ranges so both axes span the same interval,
centered on the data midpoint. Useful for curves where the spatial
relationship between x and y must be preserved (e.g. GPS tracks).

* **Parameters:**
  * **x** (*NDArray* *[**float64* *]*) – X-axis values.
  * **y** (*NDArray* *[**float64* *]*) – Y-axis values. Must have the same length as x.
  * **x_label** (*str* *,* *optional*) – X-axis label.
  * **y_label** (*str* *,* *optional*) – Y-axis label.
  * **title** (*str* *|* *None* *,* *optional*) – Plot title.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
* **Return type:**
  go.Figure

### Examples

```pycon
>>> fig = plot_parametric_curve_square(lon_arr, lat_arr, x_label="Longitude", y_label="Latitude")
>>> fig.show()
```

### perda.plotting.parametric_plot.plot_parametric_trimmer(x, y, timestamps=None, x_label='X', y_label='Y', title=None, timestamp_unit=Timescale.MS, layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10))

Interactive parametric curve trimmer with a dual-handle range slider.

Displays a 2D parametric scatter plot alongside an ipywidgets IntRangeSlider
that trims which points are shown. Designed for use in Jupyter notebooks.

* **Parameters:**
  * **x** (*NDArray* *[**float64* *]*) – X-axis values.
  * **y** (*NDArray* *[**float64* *]*) – Y-axis values. Must have the same length as x.
  * **timestamps** (*NDArray* *[**float64* *]*  *|* *None* *,* *optional*) – Timestamps corresponding to each point, shown in the slider labels.
    If None, slider labels show index only.
  * **x_label** (*str* *,* *optional*) – X-axis label.
  * **y_label** (*str* *,* *optional*) – Y-axis label.
  * **title** (*str* *|* *None* *,* *optional*) – Plot title.
  * **timestamp_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Unit for timestamp labels in the slider. Ignored if timestamps is None.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*)
* **Return type:**
  ipywidgets.VBox

### Examples

```pycon
>>> widget = plot_parametric_trimmer(lon_arr, lat_arr, timestamps=ts_arr, x_label="Longitude", y_label="Latitude")
>>> display(widget)
```
