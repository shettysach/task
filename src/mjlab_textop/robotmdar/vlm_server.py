from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class LiteRtLmPromptService:
    def __init__(
        self,
        *,
        model_file: Path,
        max_num_tokens: int,
    ) -> None:
        try:
            import litert_lm
        except ImportError as exc:
            raise ImportError(
                "Install litert-lm-api in the VLM server environment."
            ) from exc

        self._litert_lm = litert_lm
        self._engine = litert_lm.Engine(
            str(model_file),
            max_num_tokens=max_num_tokens,
        )
        self._engine.__enter__()

    def close(self) -> None:
        self._engine.__exit__(None, None, None)

    def choose_prompt(self, payload: dict[str, Any]) -> str:
        with self._engine.create_conversation(
            messages=[
                self._litert_lm.Message.system(
                    "You choose RobotMDAR/TextOp motion prompts. "
                    "Return only one short motion prompt."
                )
            ]
        ) as conversation:
            response = conversation.send_message(_make_user_message(payload))
        return _extract_text(response)


class FixedPromptService:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def close(self) -> None:
        return

    def choose_prompt(self, payload: dict[str, Any]) -> str:
        return self.prompt


def make_handler(prompt_service):
    class VlmHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/choose_prompt":
                self.send_error(404)
                return
            try:
                payload = self._read_json()
                prompt = prompt_service.choose_prompt(payload)
            except Exception as exc:
                self._write_json(500, {"error": type(exc).__name__})
                return
            self._write_json(200, {"prompt": prompt})

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request payload must be a JSON object")
            return payload

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return VlmHandler


def run_server(args: argparse.Namespace) -> None:
    if args.fixed_prompt is not None:
        prompt_service = FixedPromptService(args.fixed_prompt)
    else:
        prompt_service = LiteRtLmPromptService(
            model_file=Path(args.model_file).expanduser().resolve(),
            max_num_tokens=args.max_num_tokens,
        )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(prompt_service))
    print(f"Serving VLM prompt endpoint on http://{args.host}:{args.port}/choose_prompt")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        prompt_service.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local RobotMDAR prompt selector endpoint.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--max-num-tokens", type=int, default=256)
    parser.add_argument("--fixed-prompt", default=None)
    args = parser.parse_args()
    if args.port <= 0:
        raise ValueError(f"--port must be positive, got {args.port}")
    if args.max_num_tokens <= 0:
        raise ValueError(
            f"--max-num-tokens must be positive, got {args.max_num_tokens}"
        )
    if args.fixed_prompt is None and args.model_file is None:
        raise ValueError("Pass --model-file or --fixed-prompt")
    return args


def _make_user_message(payload: dict[str, Any]) -> str:
    return (
        "Choose the next TextOp/RobotMDAR text prompt.\n"
        "Return only the prompt text.\n\n"
        f"{json.dumps(payload, separators=(',', ':'))}"
    )


def _extract_text(response: Any) -> str:
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
    return str(response)


def main() -> None:
    run_server(parse_args())


if __name__ == "__main__":
    main()
