#!/usr/bin/env python3
"""Route ROCK's fixed sandbox service port to QEMU guest HTTP services.

DERAIL 移植自 Long_horizon-MCUA/scripts/rock_guest_gateway.py（纯 stdlib，
沙箱内零依赖）。跑在 ROCK 沙箱容器 :8080，把平台 Proxy 打进来的
/<prefix>/... 请求按路由表转发给 guest（MyPCBench 控制面 :5000）。

Plain HTTP requests are proxied via http.client.  Requests carrying an
``Upgrade: websocket`` handshake are tunnelled instead: the raw handshake is
replayed verbatim to the guest service and, once the 101 goes through, both
sockets are pumped byte-for-byte in both directions.  Without this, CDP
clients (Playwright connect_over_cdp) died with a 500 from *this* gateway --
the HOP_HEADERS filter stripped the Upgrade/Connection headers and Chrome
never switched protocols (event #18, task 35253b65, four-model outage).
"""

from __future__ import annotations

import argparse
import http.client
import os
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_HEADERS = {
    "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
}
WS_HANDSHAKE_TIMEOUT = 15.0
WS_HEAD_LIMIT = 64 * 1024
UPSTREAM_TIMEOUT = float(os.environ.get("DERAIL_GATEWAY_UPSTREAM_TIMEOUT", "1200"))


@dataclass(frozen=True)
class Route:
    prefix: str
    host: str
    port: int


def parse_route(value: str) -> Route:
    prefix, host, port = value.split(":", 2)
    return Route(prefix.strip("/"), host, int(port))


def is_websocket_upgrade(headers) -> bool:
    return (headers.get("Upgrade", "").strip().lower() == "websocket"
            and "upgrade" in headers.get("Connection", "").lower())


def read_http_head(sock: socket.socket) -> bytes:
    """Read until the end of the response head; keep any trailing bytes.

    The server may legally push WebSocket frames immediately after the 101,
    so everything received is returned and later forwarded verbatim.
    """
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        if len(buffer) > WS_HEAD_LIMIT:
            raise OSError("websocket handshake response head too large")
        chunk = sock.recv(65536)
        if not chunk:
            raise OSError("upstream closed during websocket handshake")
        buffer += chunk
    return buffer


def head_status(head: bytes) -> int:
    try:
        return int(head.split(b"\r\n", 1)[0].split(b" ", 2)[1])
    except (IndexError, ValueError):
        raise OSError(f"malformed handshake status line: {head[:80]!r}")


def pump(read_fn, write_sock: socket.socket) -> None:
    try:
        while True:
            data = read_fn()
            if not data:
                break
            write_sock.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (write_sock,):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class Gateway:
    def __init__(self, routes: list[Route]):
        self.routes = {route.prefix: route for route in routes}

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        if urlsplit(request.path).path == "/__rock_gateway_health":
            self.health(request)
            return
        route, target = self.resolve(request.path)
        if route is None:
            request.send_error(404)
            return
        if is_websocket_upgrade(request.headers):
            self.tunnel_websocket(request, route, target)
            return
        self.proxy(request, route, target)

    def resolve(self, raw_path: str) -> tuple[Route | None, str]:
        parsed = urlsplit(raw_path)
        parts = parsed.path.lstrip("/").split("/", 1)
        route = self.routes.get(parts[0])
        path = "/" + (parts[1] if len(parts) > 1 else "")
        if parsed.query:
            path += "?" + parsed.query
        return route, path

    def health(self, request: BaseHTTPRequestHandler) -> None:
        body = ('{"status":"ok","websocket":"tunnel","routes":[' +
                ",".join(f'"{name}"' for name in sorted(self.routes)) + "]}").encode()
        request.send_response(200)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)

    def proxy(self, request: BaseHTTPRequestHandler, route: Route, path: str) -> None:
        length = int(request.headers.get("Content-Length", "0") or "0")
        body = request.rfile.read(length) if length else None
        headers = {key: value for key, value in request.headers.items()
                   if key.lower() not in HOP_HEADERS}
        connection = http.client.HTTPConnection(route.host, route.port, timeout=UPSTREAM_TIMEOUT)
        try:
            connection.request(request.command, path, body=body, headers=headers)
            self.copy_response(request, connection.getresponse())
        except (OSError, http.client.HTTPException) as exc:
            # HTTPException (e.g. BadStatusLine when the upstream slams the
            # connection) used to escape as an opaque 500; keep it a 502.
            request.send_error(502, str(exc))
        finally:
            connection.close()

    def tunnel_websocket(self, request: BaseHTTPRequestHandler, route: Route,
                         path: str) -> None:
        """Replay the client's upgrade handshake verbatim and go raw-TCP."""
        request.close_connection = True  # this socket never serves HTTP again
        lines = [f"{request.command} {path} HTTP/1.1",
                 f"Host: {route.host}:{route.port}"]
        lines += [f"{key}: {value}" for key, value in request.headers.items()
                  if key.lower() != "host"]
        handshake = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
        try:
            upstream = socket.create_connection((route.host, route.port),
                                                timeout=WS_HANDSHAKE_TIMEOUT)
        except OSError as exc:
            request.send_error(502, f"websocket connect failed: {exc}")
            return
        try:
            try:
                upstream.sendall(handshake)
                head = read_http_head(upstream)
                status = head_status(head)
            except OSError as exc:
                request.send_error(502, f"websocket handshake failed: {exc}")
                return
            client = request.connection
            client.sendall(head)  # 101 head (+ any early frames) verbatim
            if status != 101:
                # Upstream refused the upgrade: its full HTTP answer has been
                # relayed; drain briefly so short error bodies arrive intact.
                upstream.settimeout(2.0)
                try:
                    while chunk := upstream.recv(65536):
                        client.sendall(chunk)
                except OSError:
                    pass
                return
            upstream.settimeout(None)
            client.settimeout(None)
            # Client bytes may already sit in rfile's buffer, so the
            # client->upstream direction must read through rfile, not the
            # raw socket.  Upstream direction pumps in this thread.
            writer = threading.Thread(
                target=pump, args=(lambda: request.rfile.read1(65536), upstream),
                daemon=True)
            writer.start()
            pump(lambda: upstream.recv(65536), client)
            writer.join(timeout=5)
        finally:
            upstream.close()

    @staticmethod
    def copy_response(request: BaseHTTPRequestHandler, response) -> None:
        request.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() not in HOP_HEADERS:
                request.send_header(key, value)
        request.end_headers()
        while chunk := response.read(1024 ** 2):
            request.wfile.write(chunk)


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            gateway.handle(self)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = dispatch

        def log_message(self, format, *args):
            print(f"[gateway] {self.address_string()} {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--route", action="append", required=True,
                        help="PREFIX:TARGET_HOST:TARGET_PORT")
    args = parser.parse_args()
    gateway = Gateway([parse_route(value) for value in args.route])
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(gateway)).serve_forever()


if __name__ == "__main__":
    main()
