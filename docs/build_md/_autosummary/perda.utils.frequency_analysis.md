# perda.utils.frequency_analysis

### perda.utils.frequency_analysis.analyze_frequency(data_instance, expected_frequency_hz=None, source_time_unit=Timescale.MS, gap_threshold_multiplier=2.0, font_config=FontConfig(large=20, medium=14, small=10), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), plot_config=ScatterHistogramPlotConfig(color_scatter='blue', color_line='crimson', color_histogram='blue', histogram_bins=80))

Analyse the sampling frequency of a DataInstance and return a diagnostic figure.

Prints a summary of frequency statistics and gap detection, then returns a
figure with two subplots: instantaneous frequency over time and a frequency
histogram.

* **Parameters:**
  * **data_instance** ([*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)) – The DataInstance whose logging cadence to analyse.
  * **expected_frequency_hz** (*float* *|* *None* *,* *optional*) – Nominal expected sampling frequency in Hz. When provided, additional
    diagnostics (frequency error, missed-message estimate, reference lines)
    are included. Default is None.
  * **source_time_unit** ([*Timescale*](perda.units.md#perda.units.Timescale) *,* *optional*) – Timestamp unit used in `data_instance.timestamp_np`. Default is ms.
  * **gap_threshold_multiplier** (*float* *,* *optional*) – An interval is flagged as a gap when it exceeds this multiple of the
    expected interval (if `expected_frequency_hz` is given) or the median
    interval. Default is 2.0.
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig) *,* *optional*) – Font sizes for plot elements. Default is DEFAULT_FONT_CONFIG.
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig) *,* *optional*) – Plot dimensions and margins. Default is DEFAULT_LAYOUT_CONFIG.
  * **plot_config** ([*ScatterHistogramPlotConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.ScatterHistogramPlotConfig) *|* *None* *,* *optional*) – Colors and histogram bin count. Default is DEFAULT_SCATTER_HISTOGRAM_PLOT_CONFIG.
* **Returns:**
  Plotly figure with frequency time-series and frequency histogram subplots.
* **Return type:**
  go.Figure

### Examples

```pycon
>>> fig = analyze_frequency(di, expected_frequency_hz=100)
>>> fig.show()
```
