import socket
import threading

import pytest

from perda.live.cdp_client import CDPClient, CDPServerError, ResponseStatus, ValueType


def test_get_raises_server_error_with_status() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))

    def reply_disconnected() -> None:
        _, address = server.recvfrom(1024)
        server.sendto(bytes([ResponseStatus.DISCONNECTED.value]), address)

    thread = threading.Thread(target=reply_disconnected)
    thread.start()
    with CDPClient(timeout=1.0) as client:
        client.connect("127.0.0.1", server.getsockname()[1])
        with pytest.raises(CDPServerError) as error:
            client.get("bms.pack.voltage", ValueType.FLOAT)
    thread.join()
    server.close()
    assert error.value.status == ResponseStatus.DISCONNECTED
