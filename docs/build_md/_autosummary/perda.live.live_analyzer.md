# perda.live.live_analyzer

## perda.live.live_analyzer

Live data analysis using the CDP IPC server, mirroring the Analyzer interface.

### *class* perda.live.live_analyzer.LiveAnalyzer(host, port, timeout, range_timeout)

Bases: `object`

Live data analyzer that pulls time-series data from the CDP IPC server
and exposes a plotting interface analogous to `Analyzer`.

* **Parameters:**
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

#### analyze_frequency(var, time_secs=30, expected_frequency_hz=None, gap_threshold_multiplier=2.0, font_config=FontConfig(large=20, medium=14, small=10), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'), plot_config=ScatterHistogramPlotConfig(color_scatter='blue', color_line='crimson', color_histogram='blue', histogram_bins=80))

* **Return type:**
  `Figure`
* **Parameters:**
  * **var** (*str*)
  * **time_secs** (*int*)
  * **expected_frequency_hz** (*float* *|* *None*)
  * **gap_threshold_multiplier** (*float*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig))
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig))
  * **plot_config** ([*ScatterHistogramPlotConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.ScatterHistogramPlotConfig))

#### *classmethod* dataserver(port=5001, timeout=5.0, range_timeout=2.0)

* **Return type:**
  [`LiveAnalyzer`](#perda.live.live_analyzer.LiveAnalyzer)
* **Parameters:**
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

#### ensure_connected()

* **Return type:**
  `None`

#### fetch(var, time_secs=30)

* **Return type:**
  [`DataInstance`](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance)
* **Parameters:**
  * **var** (*str*)
  * **time_secs** (*int*)

#### get(access_string, value_type)

* **Return type:**
  `Union`[`float`, `bool`, `int`]
* **Parameters:**
  * **access_string** (*str*)
  * **value_type** ([*ValueType*](perda.live.cdp_client.md#perda.live.cdp_client.ValueType))

#### is_connected(refresh=False)

* **Return type:**
  `bool`
* **Parameters:**
  **refresh** (*bool*)

#### *classmethod* local(port=5001, timeout=5.0, range_timeout=2.0)

* **Return type:**
  [`LiveAnalyzer`](#perda.live.live_analyzer.LiveAnalyzer)
* **Parameters:**
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

#### plot(var_1, var_2=None, time_secs=30, title=None, y_label_1=None, y_label_2=None, show_legend=True, font_config=FontConfig(large=20, medium=14, small=10), layout_config=LayoutConfig(width=1200, height=700, margin={'l': 70, 'r': 50, 't': 90, 'b': 70}, plot_bgcolor='white', title_x=0.5, title_xanchor='center', title_yanchor='top'))

* **Return type:**
  `Figure`
* **Parameters:**
  * **var_1** (*str* *|* [*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *|* *List* *[**str* *|* [*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *]*)
  * **var_2** (*str* *|* [*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *|* *List* *[**str* *|* [*DataInstance*](perda.core_data_structures.data_instance.md#perda.core_data_structures.data_instance.DataInstance) *]*  *|* *None*)
  * **time_secs** (*int*)
  * **title** (*str* *|* *None*)
  * **y_label_1** (*str* *|* *None*)
  * **y_label_2** (*str* *|* *None*)
  * **show_legend** (*bool*)
  * **font_config** ([*FontConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.FontConfig))
  * **layout_config** ([*LayoutConfig*](perda.plotting.plotting_constants.md#perda.plotting.plotting_constants.LayoutConfig))

#### *classmethod* remote(host, port=5001, timeout=5.0, range_timeout=2.0)

* **Return type:**
  [`LiveAnalyzer`](#perda.live.live_analyzer.LiveAnalyzer)
* **Parameters:**
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

#### set(access_string, value, value_type)

* **Return type:**
  `None`
* **Parameters:**
  * **access_string** (*str*)
  * **value** (*float* *|* *bool* *|* *int*)
  * **value_type** ([*ValueType*](perda.live.cdp_client.md#perda.live.cdp_client.ValueType))
