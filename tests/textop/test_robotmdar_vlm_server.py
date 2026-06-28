from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from mjlab_textop.robotmdar.vlm_server import FixedPromptService, make_handler


def test_vlm_server_fixed_prompt_endpoint() -> None:
    service = FixedPromptService("walk forward")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        request = urllib.request.Request(
            f"http://{host}:{port}/choose_prompt",
            data=json.dumps({"current_prompt": "stand still"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload == {"prompt": "walk forward"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
