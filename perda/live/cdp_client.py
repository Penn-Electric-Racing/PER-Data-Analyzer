"""
Python client library for communicating with the Rust CDP IPC server.
Supports getting and setting values, as well as fetching time-series ranges.
"""

import logging
import socket
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ValueType(Enum):
    """Value types supported by the CDP IPC protocol."""

    NUMERIC = 0x00
    FLOAT = 0x01
    BOOL = 0x02
    _RANGE = 0x03  # Internal use only — signals a GetRanged request


class CanType(Enum):
    """CAN data types as defined by the server."""

    Bool = 0
    UInt8 = 1
    UInt16 = 2
    UInt32 = 3
    UInt64 = 4
    Int8 = 5
    Int16 = 6
    Int32 = 7
    Int64 = 8
    Float = 9
    Double = 10


# Maps CanType to (struct format char, byte size)
_CAN_TYPE_FMT: dict[CanType, Tuple[str, int]] = {
    CanType.Bool: ("B", 1),
    CanType.UInt8: ("B", 1),
    CanType.UInt16: ("H", 2),
    CanType.UInt32: ("I", 4),
    CanType.UInt64: ("Q", 8),
    CanType.Int8: ("b", 1),
    CanType.Int16: ("h", 2),
    CanType.Int32: ("i", 4),
    CanType.Int64: ("q", 8),
    CanType.Float: ("f", 4),
    CanType.Double: ("d", 8),
}


class ResponseStatus(Enum):
    """Response status codes from the CDP server."""

    GET_SUCCESS = 0x0
    SET_SUCCESS = 0x1
    COMM_ERROR = 0x2
    INVALID_ACCESS = 0x3
    INCORRECT_TYPE = 0x4
    DISCONNECTED = 0x5
    BADLY_FORMED = 0x6
    GET_RANGED_SUCCESS = 0x7


class CDPException(Exception):
    """Base exception for CDP client errors."""

    pass


class CDPProtocolError(CDPException):
    """Raised when there's a protocol-level error."""

    pass


class CDPServerError(CDPException):
    """Raised when the server returns an error response."""

    def __init__(self, status: ResponseStatus, message: str):
        self.status = status
        super().__init__(message)


@dataclass
class _RangedPacket:
    """Parsed contents of a single GetRangedSuccess UDP packet."""

    var_id: int
    begin: int
    end: int
    can_type: CanType
    storage_interval: int  # milliseconds
    values: np.ndarray
    timestamps: np.ndarray  # milliseconds


def _parse_ranged_packet(payload: bytes) -> _RangedPacket:
    """
    Parse the payload of a GetRangedSuccess packet (everything after the 0x7 sig byte).

    Layout (all little-endian):
        4B  id.id           u32
        4B  id.index        u32  (ignored)
        4B  df.begin        u32
        4B  df.end          u32
        1B  df.ty           u8   (CanType discriminant)
        4B  storage_interval u32  (milliseconds)
       NxS  packed samples
    """
    if len(payload) < 17:
        raise CDPProtocolError(f"GetRanged payload too short: {len(payload)} bytes")

    var_id, _index, begin, end, can_type_raw = struct.unpack_from("<IIIIB", payload, 0)
    # Header is 4+4+4+4+1 = 17 bytes, then storage_interval u32 = 4 more bytes
    storage_interval: int = struct.unpack_from("<I", payload, 17)[0]

    try:
        can_type = CanType(can_type_raw)
    except ValueError:
        raise CDPProtocolError(f"Unknown CanType discriminant: {can_type_raw}")

    fmt_char, sample_size = _CAN_TYPE_FMT[can_type]
    n_samples = end - begin + 1
    expected_data_size = n_samples * sample_size
    data_offset = 21  # 17 (fixed header) + 4 (storage_interval)

    actual_data_size = len(payload) - data_offset
    if actual_data_size != expected_data_size:
        raise CDPProtocolError(
            f"Data size mismatch: expected {expected_data_size}B, got {actual_data_size}B"
        )

    raw = struct.unpack_from(f"<{n_samples}{fmt_char}", payload, data_offset)
    values = np.array(raw, dtype=float)

    # Reconstruct timestamps in milliseconds
    timestamps = np.array(
        [(begin + i) * storage_interval for i in range(n_samples)],
        dtype=np.int64,
    )

    return _RangedPacket(
        var_id=var_id,
        begin=begin,
        end=end,
        can_type=can_type,
        storage_interval=storage_interval,
        values=values,
        timestamps=timestamps,
    )


class CDPClient:
    """
    Python client for communicating with the Rust CDP IPC server.

    Example usage:
        client = CDPClient()
        client.connect("127.0.0.1", 5001)

        # Get a single value
        value = client.get("vehicle.speed", ValueType.FLOAT)

        # Get a time-series range (last 10 seconds)
        di = client.get_range("vehicle.speed", ValueType.FLOAT, time_secs=10)

        # Set a value
        client.set("vehicle.target_speed", 65.0, ValueType.FLOAT)

        client.disconnect()
    """

    ACCESS_STRING_SIZE = 56
    PACKET_SIZE = 1 + ACCESS_STRING_SIZE + 8  # 65 bytes total

    def __init__(self, timeout: float = 5.0, range_timeout: float = 2.0):
        """
        Initialize the CDP client.

        Args:
            timeout:       Socket timeout for get/set requests (seconds)
            range_timeout: Maximum time to wait for all ranged response
                           packets before giving up (seconds)
        """
        self.socket: Optional[socket.socket] = None
        self.server_addr: Optional[Tuple[str, int]] = None
        self.timeout = timeout
        self.range_timeout = range_timeout

    def connect(self, host: str = "127.0.0.1", port: int = 5001) -> None:
        """Connect to the CDP server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)
            self.server_addr = (host, port)
            logger.info(f"Connected to CDP server at {host}:{port}")
        except Exception as e:
            raise CDPException(f"Failed to connect to server: {e}")

    def disconnect(self) -> None:
        """Disconnect from the CDP server."""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.server_addr = None
            logger.info("Disconnected from CDP server")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, access_string: str, value_type: ValueType) -> Union[float, bool, int]:
        """
        Get the latest value for a signal from the CDP server.

        Args:
            access_string: The access path (e.g. "vehicle.speed")
            value_type:    Expected type of the value

        Returns:
            The requested value as float, bool, or int
        """
        self._check_connected()
        packet = self._build_request_packet(access_string, value_type, mode="get")
        try:
            self.socket.sendto(packet, self.server_addr)
            response, _ = self.socket.recvfrom(1024)
            return self._parse_get_response(response, value_type)
        except socket.timeout:
            raise CDPException("Request timed out")
        except Exception as e:
            raise CDPException(f"Communication error: {e}")

    def get_range(
        self,
        access_string: str,
        time_secs: int,
    ):
        """
        Request a time-series range of data from the CDP server.

        The server will chunk the response into multiple UDP packets. This
        method collects all packets until either the expected sample count is
        reached or range_timeout elapses, then assembles them into a single
        DataInstance.

        Args:
            access_string: The access path (e.g. "vehicle.speed")
            value_type:    Expected type of the value
            time_secs:     How many seconds of history to retrieve (u32)

        Returns:
            A DataInstance with the assembled time-series data
        """
        from perda.analyzer import DataInstance

        self._check_connected()

        if not (0 < time_secs < 2**32):
            raise CDPException("time_secs must be a positive u32 value")

        packet = self._build_request_packet(
            access_string, ValueType._RANGE, mode="get_range", time_secs=time_secs
        )

        try:
            self.socket.sendto(packet, self.server_addr)
        except Exception as e:
            raise CDPException(f"Failed to send ranged request: {e}")

        # Switch to range_timeout for collection phase
        self.socket.settimeout(self.range_timeout)

        packets: list[_RangedPacket] = []
        storage_interval: Optional[int] = None
        var_id: Optional[int] = None
        total_samples_received = 0
        expected_samples: Optional[int] = None

        try:
            while True:
                try:
                    response, _ = self.socket.recvfrom(65535)
                except socket.timeout:
                    logger.warning(
                        "get_range timed out waiting for packets; "
                        f"received {total_samples_received} of "
                        f"{expected_samples or '?'} expected samples"
                    )
                    break

                if len(response) < 1:
                    raise CDPProtocolError("Empty response packet")

                status_byte = response[0]

                # Any error status during a ranged request is fatal
                if status_byte != ResponseStatus.GET_RANGED_SUCCESS.value:
                    self._raise_server_error(ResponseStatus(status_byte))

                pkt = _parse_ranged_packet(response[1:])
                packets.append(pkt)
                total_samples_received += len(pkt.values)

                # Derive expected total samples from first packet
                if storage_interval is None:
                    storage_interval = pkt.storage_interval
                    var_id = pkt.var_id
                    if storage_interval > 0:
                        expected_samples = (time_secs * 1000) // storage_interval
                    else:
                        logger.warning(
                            "storage_interval is zero; terminating on timeout"
                        )

                # Terminate early if we have all expected samples
                if (
                    expected_samples is not None
                    and total_samples_received >= expected_samples
                ):
                    logger.debug(
                        f"get_range complete: {total_samples_received} samples received"
                    )
                    break

        finally:
            # Restore normal timeout
            self.socket.settimeout(self.timeout)

        if not packets:
            raise CDPException("No ranged response packets received")

        # Sort packets by begin index to ensure chronological order
        packets.sort(key=lambda p: p.begin)

        all_timestamps = np.concatenate([p.timestamps for p in packets])
        all_values = np.concatenate([p.values for p in packets])

        return DataInstance(
            timestamp_np=all_timestamps,
            value_np=all_values,
            label=access_string,
            var_id=var_id,
        )

    def set(
        self,
        access_string: str,
        value: Union[float, bool, int],
        value_type: ValueType,
    ) -> None:
        """
        Set a value on the CDP server.

        Args:
            access_string: The access path (e.g. "vehicle.target_speed")
            value:         The value to set
            value_type:    Type of the value
        """
        self._check_connected()
        packet = self._build_request_packet(
            access_string, value_type, mode="set", value=float(value)
        )
        try:
            self.socket.sendto(packet, self.server_addr)
            response, _ = self.socket.recvfrom(1024)
            self._parse_set_response(response)
        except socket.timeout:
            raise CDPException("Request timed out")
        except Exception as e:
            raise CDPException(f"Communication error: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_connected(self) -> None:
        if not self.socket or not self.server_addr:
            raise CDPException("Not connected to server")

    def _build_request_packet(
        self,
        access_string: str,
        value_type: ValueType,
        mode: str,  # "get" | "get_range" | "set"
        value: float = 0.0,
        time_secs: int = 0,
    ) -> bytes:
        """Build a request packet according to the protocol.

        Sig byte layout:
            bits 0-1 : ValueType (0b11 = Range, triggers GetRanged when bit 7 set)
            bit  7   : 1 = Get/GetRanged, 0 = Set
        """
        if len(access_string.encode("utf-8")) > self.ACCESS_STRING_SIZE:
            raise CDPException(
                f"Access string too long (max {self.ACCESS_STRING_SIZE} bytes)"
            )

        sig = value_type.value & 0b11

        if mode in ("get", "get_range"):
            sig |= 1 << 7
        # mode == "set": no additional bits

        packet = bytearray(self.PACKET_SIZE)
        packet[0] = sig

        access_bytes = access_string.encode("utf-8")
        packet[1 : 1 + len(access_bytes)] = access_bytes

        if mode == "get_range":
            # time_secs packed as u32 LE into the 8-byte value field
            packet[1 + self.ACCESS_STRING_SIZE :] = (
                struct.pack("<I", time_secs) + b"\x00" * 4
            )
        else:
            packet[1 + self.ACCESS_STRING_SIZE :] = struct.pack("<d", value)

        return bytes(packet)

    def _parse_get_response(
        self, response: bytes, value_type: ValueType
    ) -> Union[float, bool, int]:
        if len(response) < 1:
            raise CDPProtocolError("Response too short")

        status = ResponseStatus(response[0])
        if status != ResponseStatus.GET_SUCCESS:
            self._raise_server_error(status)

        if len(response) != 9:
            raise CDPProtocolError(
                f"Invalid GET response length: expected 9, got {len(response)}"
            )

        float_value = struct.unpack("<d", response[1:9])[0]

        if value_type == ValueType.BOOL:
            return bool(float_value)
        elif value_type == ValueType.NUMERIC:
            return int(float_value) if float_value.is_integer() else float_value
        else:
            return float_value

    def _parse_set_response(self, response: bytes) -> None:
        if len(response) < 1:
            raise CDPProtocolError("Response too short")

        status = ResponseStatus(response[0])
        if status != ResponseStatus.SET_SUCCESS:
            self._raise_server_error(status)

    def _raise_server_error(self, status: ResponseStatus) -> None:
        error_messages = {
            ResponseStatus.COMM_ERROR: "Communication error with backend",
            ResponseStatus.INVALID_ACCESS: "Invalid access string",
            ResponseStatus.INCORRECT_TYPE: "Incorrect value type",
            ResponseStatus.DISCONNECTED: "Server disconnected from backend",
            ResponseStatus.BADLY_FORMED: "Badly formed request",
        }
        message = error_messages.get(status, f"Unknown server error: {status}")
        raise CDPServerError(status, message)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_value(
    access_string: str,
    value_type: ValueType,
    host: str = "127.0.0.1",
    port: int = 5001,
    timeout: float = 5.0,
) -> Union[float, bool, int]:
    """Convenience function to get a single value."""
    with CDPClient(timeout=timeout) as client:
        client.connect(host, port)
        return client.get(access_string, value_type)


def set_value(
    access_string: str,
    value: Union[float, bool, int],
    value_type: ValueType,
    host: str = "127.0.0.1",
    port: int = 5001,
    timeout: float = 5.0,
) -> None:
    """Convenience function to set a single value."""
    with CDPClient(timeout=timeout) as client:
        client.connect(host, port)
        client.set(access_string, value, value_type)


def get_range_value(
    access_string: str,
    time_secs: int,
    host: str = "127.0.0.1",
    port: int = 5001,
    timeout: float = 5.0,
    range_timeout: float = 2.0,
):
    """Convenience function to fetch a time-series range as a DataInstance."""
    with CDPClient(timeout=timeout, range_timeout=range_timeout) as client:
        client.connect(host, port)
        return client.get_range(access_string, time_secs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with CDPClient() as client:
        client.connect(host="127.0.0.1")

        # Single value example
        try:
            temp = client.get("bms.board.glvTemp", ValueType.FLOAT)
            print(f"Current temp: {temp}")
        except CDPServerError as e:
            print(f"Server error: {e} (status: {e.status})")

        # Ranged example — last 30 seconds of motor temperature
        try:
            di = client.get_range("motor.temp", ValueType.FLOAT, time_secs=30)
            print(f"Got {len(di.value_np)} samples for '{di.label}'")
        except CDPServerError as e:
            print(f"Server error: {e} (status: {e.status})")
