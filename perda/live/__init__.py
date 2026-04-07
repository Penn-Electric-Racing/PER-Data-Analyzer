"""
perda.live
----------
Live data acquisition from the CDP IPC server.

Quick start
-----------
    from perda.live import LiveAnalyzer, ValueType

    live = LiveAnalyzer.local()               # running on this machine
    live = LiveAnalyzer.dataserver()          # Penn Electric Racing dataserver
    live = LiveAnalyzer.remote("10.0.0.5")   # arbitrary IP

    fig = live.plot("bms.board.glvTemp")
    fig.show()
"""

from .cdp_client import (
    CDPClient,
    CDPException,
    CDPProtocolError,
    CDPServerError,
    ResponseStatus,
    ValueType,
    get_range_value,
    get_value,
    set_value,
)
from .live_analyzer import LiveAnalyzer

__all__ = [
    "LiveAnalyzer",
    "CDPClient",
    "CDPException",
    "CDPProtocolError",
    "CDPServerError",
    "ResponseStatus",
    "ValueType",
    "get_value",
    "set_value",
    "get_range_value",
]
