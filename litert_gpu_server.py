from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import litert_lm


def decode_data_url_to_file(data_url: str) -> str:
    # Expected: data:image/jpeg;base64,...
    header, b64 = data_url.split(",", 1)
    mime = header.removeprefix("data:").split(";")[0] or "image/jpeg"
    suffix = mimetypes.guess_extension(mime) or ".jpg"

    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(base64.b64decode(b64))
    f.close()
    return f.name


def openai_content_to_litert_content(content: Any) -> Any:
    if isinstance(content, str):
        return content

    out: list[dict[str, Any]] = []
    for item in content:
        typ = item.get("type")

        if typ == "text":
            out.append({"type": "text", "text": item.get("text", "")})

        elif typ == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if not url.startswith("data:"):
                raise ValueError("Only data: image URLs are supported")
            image_path = decode_data_url_to_file(url)
            out.append({"type": "image", "path": image_path})

        else:
            raise ValueError(f"Unsupported content item type: {typ!r}")

    return out


class LiteRTOpenAIHandler(BaseHTTPRequestHandler):
    engine: litert_lm.Engine
    model_id: str

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return

        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))

            messages = req.get("messages", [])
            system_messages: list[dict[str, Any]] = []
            user_message: dict[str, Any] | None = None

            for msg in messages:
                role = msg.get("role")
                if role == "system":
                    system_messages.append(
                        {
                            "role": "system",
                            "content": openai_content_to_litert_content(
                                msg.get("content", "")
                            ),
                        }
                    )
                elif role == "user":
                    user_message = msg

            if user_message is None:
                raise ValueError("No user message found")

            sampler_config = litert_lm.SamplerConfig(
                temperature=req.get("temperature", 0.0),
                top_k=req.get("top_k"),
                top_p=req.get("top_p"),
                seed=req.get("seed"),
            )

            with self.engine.create_conversation(
                messages=system_messages or None,
                sampler_config=sampler_config,
            ) as conv:
                litert_msg = {
                    "role": "user",
                    "content": openai_content_to_litert_content(
                        user_message.get("content", "")
                    ),
                }

                text_parts: list[str] = []
                for chunk in conv.send_message_async(litert_msg):
                    for item in chunk.get("content", []):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))

                text = "".join(text_parts).strip()

            self._json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.get("model", self.model_id),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        except Exception as exc:
            self._json(
                500,
                {
                    "error": {
                        "message": str(exc),
                        "type": type(exc).__name__,
                    }
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Path to .litertlm file, e.g. /tmp/litert-models/gemma-4-E2B-it.litertlm",
    )
    parser.add_argument("--model-id", default="gemma-4-E2B-it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9379)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser().resolve())

    engine_cm = litert_lm.Engine(
        model_path,
        backend=litert_lm.Backend.GPU(),
        vision_backend=litert_lm.Backend.GPU(),
        max_num_tokens=args.max_num_tokens,
    )

    with engine_cm as engine:
        LiteRTOpenAIHandler.engine = engine
        LiteRTOpenAIHandler.model_id = args.model_id

        server = ThreadingHTTPServer((args.host, args.port), LiteRTOpenAIHandler)
        print(f"Serving {args.model_id} on http://{args.host}:{args.port}")
        print("Text backend: GPU")
        print("Vision backend: GPU")
        server.serve_forever()


if __name__ == "__main__":
    main()
