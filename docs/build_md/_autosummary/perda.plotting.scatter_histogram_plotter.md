# perda.plotting.scatter_histogram_plotter

### perda.plotting.scatter_histogram_plotter.plot_scatter_and_histogram(x, y, title, scatter_title, histogram_title, x_label, y_label, scatter_name, histogram_name, hline=None, hline_label=None, vline=None, vline_label=None, font_config=FontConfig(large=20, medium=14, small=10), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), plot_config=ScatterHistogramPlotConfig(color_scatter='blue', color_line='crimson', color_histogram='blue', histogram_bins=80))

Build a figure with a scatter plot over time and a histogram of the same values.

* **Parameters:**
  * **x** (*NDArray* *[**float64* *]*) – X-axis values for the scatter plot (e.g. time).
  * **y** (*NDArray* *[**float64* *]*) – Y-axis values shared between the scatter plot and histogram.
  * **title** (*str*) – Overall figure title.
  * **scatter_title** (*str*) – Subplot title for the scatter plot.
  * **histogram_title** (*str*) – Subplot title for the histogram.
  * **x_label** (*str*) – X-axis label for the scatter plot.
  * **y_label** (*str*) – Y-axis label for the scatter plot and X-axis label for the histogram.
  * **scatter_name** (*str*) – Legend name for the scatter trace.
  * **histogram_name** (*str*) – Legend name for the histogram trace.
  * **hline** (*float* *|* *None* *,* *optional*) – Y value at which to draw a horizontal reference line on the scatter plot.
    Default is None.
  * **hline_label** (*str* *|* *None* *,* *optional*) – Annotation text for the horizontal reference line. Default is None.
  * **vline** (*float* *|* *None* *,* *optional*) – X value at which to draw a vertical reference line on the histogram.
    Default is None.
  * **vline_label** (*str* *|* *None* *,* *optional*) – Annotation text for the vertical reference line. Default is None.
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*) – Font sizes for plot elements. Default is DEFAULT_FONT_CONFIG.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*) – Plot dimensions and margins. Default is DEFAULT_LAYOUT_CONFIG.
  * **plot_config** ([*ScatterHistogramPlotConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.ScatterHistogramPlotConfig) *,* *optional*) – Colors and histogram bin count. Default is DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG.
* **Returns:**
  Plotly figure with scatter and histogram subplots.
* **Return type:**
  go.Figure
