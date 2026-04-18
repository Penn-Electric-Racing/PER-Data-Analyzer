# perda.plotting.diff_plotter

### perda.plotting.diff_plotter.plot_diff_bars(base_extra_ts, incom_extra_ts, value_mismatch_ts, total_present_ts, title='Diff Counts Over Time (bucketed)', y_axis_title='Event Count', show_legend=True, diff_plot_config=DiffPlotConfig(bucket_size_ms=1000, color_base_extra='blue', color_incom_extra='darkorange', color_value_mismatch='crimson', color_total='gray'), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), font_config=FontConfig(large=20, medium=14, small=10))

Plot diff results as bucketed bar charts.

Renders four overlaid bar traces — total datapoints present (background),
base-only extras, incoming-only extras, and value mismatches — bucketed
into fixed-width time windows.

* **Parameters:**
  * **base_extra_ts** (`ndarray`[`tuple`[`Any`, `...`], `dtype`[`int64`]]) – Timestamps (ms) of datapoints present only in the base run.
  * **incom_extra_ts** (`ndarray`[`tuple`[`Any`, `...`], `dtype`[`int64`]]) – Timestamps (ms) of datapoints present only in the incoming run.
  * **value_mismatch_ts** (`ndarray`[`tuple`[`Any`, `...`], `dtype`[`int64`]]) – Timestamps (ms) of matched points whose values differ.
  * **total_present_ts** (`ndarray`[`tuple`[`Any`, `...`], `dtype`[`int64`]]) – Timestamps (ms) of all datapoints seen in either run.
  * **title** (`str` | `None`) – Plot title.
  * **y_axis_title** (`str` | `None`) – Label for the y-axis.
  * **show_legend** (`bool`) – Whether to show the legend.
  * **diff_plot_config** ([`DiffPlotConfig`](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.DiffPlotConfig)) – Colors and bucket size configuration.
  * **layout_config** ([`LayoutConfig`](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig)) – Plot dimensions and margin configuration.
  * **font_config** ([`FontConfig`](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig)) – Font sizes for titles, labels, and ticks.
* **Return type:**
  go.Figure
