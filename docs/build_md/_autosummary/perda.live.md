# perda.live

## perda.live

Live data acquisition from the CDP IPC server.

## Quick start

> from perda.live import LiveAnalyzer, ValueType

> live = LiveAnalyzer.local()               # running on this machine
> live = LiveAnalyzer.dataserver()          # Penn Electric Racing dataserver
> live = LiveAnalyzer.remote(“10.0.0.5”)   # arbitrary IP

> fig = live.plot(“bms.board.glvTemp”)
> fig.show()

### *class* perda.live.CDPClient(timeout=5.0, range_timeout=2.0)

Bases: `object`

Python client for communicating with the Rust CDP IPC server.

Example usage:
: client = CDPClient()
  client.connect(“127.0.0.1”, 5001)
  <br/>
  # Get a single value
  value = client.get(“vehicle.speed”, ValueType.FLOAT)
  <br/>
  # Get a time-series range (last 10 seconds)
  di = client.get_range(“vehicle.speed”, ValueType.FLOAT, time_secs=10)
  <br/>
  # Set a value
  client.set(“vehicle.target_speed”, 65.0, ValueType.FLOAT)
  <br/>
  client.disconnect()

* **Parameters:**
  * **timeout** (*float*)
  * **range_timeout** (*float*)

#### ACCESS_STRING_SIZE *= 56*

#### PACKET_SIZE *= 65*

#### connect(host='127.0.0.1', port=5001)

Connect to the CDP server.

* **Return type:**
  `None`
* **Parameters:**
  * **host** (*str*)
  * **port** (*int*)

#### disconnect()

Disconnect from the CDP server.

* **Return type:**
  `None`

#### get(access_string, value_type)

Get the latest value for a signal from the CDP server.

* **Parameters:**
  * **access_string** (`str`) – The access path (e.g. “vehicle.speed”)
  * **value_type** ([`ValueType`](perda.live.cdp_client.md#perda.live.cdp_client.ValueType)) – Expected type of the value
* **Return type:**
  `Union`[`float`, `bool`, `int`]
* **Returns:**
  The requested value as float, bool, or int

#### get_range(access_string, time_secs)

Request a time-series range of data from the CDP server.

The server will chunk the response into multiple UDP packets. This
method collects all packets until either the expected sample count is
reached or range_timeout elapses, then assembles them into a single
DataInstance.

* **Parameters:**
  * **access_string** (`str`) – The access path (e.g. “vehicle.speed”)
  * **value_type** – Expected type of the value
  * **time_secs** (`int`) – How many seconds of history to retrieve (u32)
* **Returns:**
  A DataInstance with the assembled time-series data

#### set(access_string, value, value_type)

Set a value on the CDP server.

* **Parameters:**
  * **access_string** (`str`) – The access path (e.g. “vehicle.target_speed”)
  * **value** (`Union`[`float`, `bool`, `int`]) – The value to set
  * **value_type** ([`ValueType`](perda.live.cdp_client.md#perda.live.cdp_client.ValueType)) – Type of the value
* **Return type:**
  `None`

### *exception* perda.live.CDPException

Bases: `Exception`

Base exception for CDP client errors.

### *exception* perda.live.CDPProtocolError

Bases: [`CDPException`](perda.live.cdp_client.md#perda.live.cdp_client.CDPException)

Raised when there’s a protocol-level error.

### *exception* perda.live.CDPServerError(status, message)

Bases: [`CDPException`](perda.live.cdp_client.md#perda.live.cdp_client.CDPException)

Raised when the server returns an error response.

* **Parameters:**
  * **status** ([*ResponseStatus*](perda.live.cdp_client.md#perda.live.cdp_client.ResponseStatus))
  * **message** (*str*)

### *class* perda.live.LiveAnalyzer(host, port, timeout, range_timeout)

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
  [`LiveAnalyzer`](perda.live.live_analyzer.md#perda.live.live_analyzer.LiveAnalyzer)
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
  [`LiveAnalyzer`](perda.live.live_analyzer.md#perda.live.live_analyzer.LiveAnalyzer)
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
  [`LiveAnalyzer`](perda.live.live_analyzer.md#perda.live.live_analyzer.LiveAnalyzer)
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

### *class* perda.live.ResponseStatus(\*values)

Bases: `Enum`

Response status codes from the CDP server.

#### BADLY_FORMED *= 6*

#### COMM_ERROR *= 2*

#### DISCONNECTED *= 5*

#### GET_RANGED_SUCCESS *= 7*

#### GET_SUCCESS *= 0*

#### INCORRECT_TYPE *= 4*

#### INVALID_ACCESS *= 3*

#### SET_SUCCESS *= 1*

### *class* perda.live.ValueType(\*values)

Bases: `Enum`

Value types supported by the CDP IPC protocol.

#### BOOL *= 2*

#### FLOAT *= 1*

#### NUMERIC *= 0*

### perda.live.get_range_value(access_string, time_secs, host='127.0.0.1', port=5001, timeout=5.0, range_timeout=2.0)

Convenience function to fetch a time-series range as a DataInstance.

* **Parameters:**
  * **access_string** (*str*)
  * **time_secs** (*int*)
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

### perda.live.get_value(access_string, value_type, host='127.0.0.1', port=5001, timeout=5.0)

Convenience function to get a single value.

* **Return type:**
  `Union`[`float`, `bool`, `int`]
* **Parameters:**
  * **access_string** (*str*)
  * **value_type** ([*ValueType*](perda.live.cdp_client.md#perda.live.cdp_client.ValueType))
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)

### perda.live.set_value(access_string, value, value_type, host='127.0.0.1', port=5001, timeout=5.0)

Convenience function to set a single value.

* **Return type:**
  `None`
* **Parameters:**
  * **access_string** (*str*)
  * **value** (*float* *|* *bool* *|* *int*)
  * **value_type** ([*ValueType*](perda.live.cdp_client.md#perda.live.cdp_client.ValueType))
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)

### Modules

| [`cdp_client`](perda.live.cdp_client.md#module-perda.live.cdp_client)          | Python client library for communicating with the Rust CDP IPC server.                                            |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`live_analyzer`](perda.live.live_analyzer.md#module-perda.live.live_analyzer) | perda.live.live_analyzer<br/><br/>Live data analysis using the CDP IPC server, mirroring the Analyzer interface. |
