from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TMJ best-model GUI locally.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    handler = lambda *handler_args, **handler_kwargs: QuietHandler(
        *handler_args,
        directory=str(root),
        **handler_kwargs,
    )

    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        print(f"TMJ best-model GUI: http://127.0.0.1:{args.port}/gui/index.html")
        server.serve_forever()


if __name__ == "__main__":
    main()

