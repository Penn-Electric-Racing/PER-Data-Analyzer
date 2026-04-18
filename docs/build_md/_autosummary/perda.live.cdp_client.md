# perda.live.cdp_client

Python client library for communicating with the Rust CDP IPC server.
Supports getting and setting values, as well as fetching time-series ranges.

### *class* perda.live.cdp_client.CDPClient(timeout=5.0, range_timeout=2.0)

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
  * **value_type** ([`ValueType`](#perda.live.cdp_client.ValueType)) – Expected type of the value
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

#### server_addr *: `Optional`[`Tuple`[`str`, `int`]]*

#### set(access_string, value, value_type)

Set a value on the CDP server.

* **Parameters:**
  * **access_string** (`str`) – The access path (e.g. “vehicle.target_speed”)
  * **value** (`Union`[`float`, `bool`, `int`]) – The value to set
  * **value_type** ([`ValueType`](#perda.live.cdp_client.ValueType)) – Type of the value
* **Return type:**
  `None`

#### socket *: `Optional`[`socket`]*

### *exception* perda.live.cdp_client.CDPException

Bases: `Exception`

Base exception for CDP client errors.

### *exception* perda.live.cdp_client.CDPProtocolError

Bases: [`CDPException`](#perda.live.cdp_client.CDPException)

Raised when there’s a protocol-level error.

### *exception* perda.live.cdp_client.CDPServerError(status, message)

Bases: [`CDPException`](#perda.live.cdp_client.CDPException)

Raised when the server returns an error response.

* **Parameters:**
  * **status** ([*ResponseStatus*](#perda.live.cdp_client.ResponseStatus))
  * **message** (*str*)

### *class* perda.live.cdp_client.CanType(\*values)

Bases: `Enum`

CAN data types as defined by the server.

#### Bool *= 0*

#### Double *= 10*

#### Float *= 9*

#### Int16 *= 6*

#### Int32 *= 7*

#### Int64 *= 8*

#### Int8 *= 5*

#### UInt16 *= 2*

#### UInt32 *= 3*

#### UInt64 *= 4*

#### UInt8 *= 1*

### *class* perda.live.cdp_client.ResponseStatus(\*values)

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

### *class* perda.live.cdp_client.ValueType(\*values)

Bases: `Enum`

Value types supported by the CDP IPC protocol.

#### BOOL *= 2*

#### FLOAT *= 1*

#### NUMERIC *= 0*

### perda.live.cdp_client.get_range_value(access_string, time_secs, host='127.0.0.1', port=5001, timeout=5.0, range_timeout=2.0)

Convenience function to fetch a time-series range as a DataInstance.

* **Parameters:**
  * **access_string** (*str*)
  * **time_secs** (*int*)
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)
  * **range_timeout** (*float*)

### perda.live.cdp_client.get_value(access_string, value_type, host='127.0.0.1', port=5001, timeout=5.0)

Convenience function to get a single value.

* **Return type:**
  `Union`[`float`, `bool`, `int`]
* **Parameters:**
  * **access_string** (*str*)
  * **value_type** ([*ValueType*](#perda.live.cdp_client.ValueType))
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)

### perda.live.cdp_client.set_value(access_string, value, value_type, host='127.0.0.1', port=5001, timeout=5.0)

Convenience function to set a single value.

* **Return type:**
  `None`
* **Parameters:**
  * **access_string** (*str*)
  * **value** (*float* *|* *bool* *|* *int*)
  * **value_type** ([*ValueType*](#perda.live.cdp_client.ValueType))
  * **host** (*str*)
  * **port** (*int*)
  * **timeout** (*float*)
