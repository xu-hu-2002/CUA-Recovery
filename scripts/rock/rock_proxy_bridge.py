"""Local compatibility bridge for ROCK Sandbox Proxy endpoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

from aiohttp import ClientSession, ClientTimeout, TCPConnector, WSMsgType, web


HOP_HEADERS = {
    "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
}
WEBSOCKET_HANDSHAKE_HEADERS = {
    "sec-websocket-accept", "sec-websocket-extensions", "sec-websocket-key",
    "sec-websocket-protocol", "sec-websocket-version",
}


@dataclass(frozen=True)
class ProxyRoute:
    local_port: int
    sandbox_port: int | None
    rewrite_cdp: bool = False
    upstream_prefix: str = ""


def sandbox_api_root(base_url: str) -> str:
    base = base_url.rstrip("/")
    suffix = "/apis/envs/sandbox/v1"
    return base if base.endswith(suffix) else base + suffix


def proxy_url(base_url: str, sandbox_id: str, port: int | None, raw_path: str) -> str:
    path = raw_path if raw_path.startswith("/") else "/" + raw_path
    root = sandbox_api_root(base_url)
    target = f"{root}/sandboxes/{quote(sandbox_id)}/proxy"
    if port is not None:
        target += f"/port/{port}"
    return target + path


def websocket_url(http_url: str) -> str:
    parsed = urlsplit(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def rewrite_cdp_urls(body: bytes, local_port: int) -> bytes:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    changed = _rewrite_cdp_value(payload, local_port)
    return json.dumps(changed).encode("utf-8")


def _rewrite_cdp_value(value, local_port: int):
    if isinstance(value, dict):
        return {key: _rewrite_cdp_value(item, local_port) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_cdp_value(item, local_port) for item in value]
    if isinstance(value, str) and value.startswith(("ws://", "wss://")):
        parsed = urlsplit(value)
        return urlunsplit(("ws", f"127.0.0.1:{local_port}", parsed.path, parsed.query, parsed.fragment))
    return value


class RockProxyBridge:
    def __init__(self, base_url: str, api_key: str, sandbox_id: str,
                 routes: list[ProxyRoute], host: str = "127.0.0.1"):
        self.base_url = base_url
        self.api_key = api_key
        self.sandbox_id = sandbox_id
        self.routes = routes
        self.host = host
        self._runners: list[web.AppRunner] = []
        self._session: ClientSession | None = None

    async def start(self) -> None:
        connector = TCPConnector(force_close=True, enable_cleanup_closed=True)
        self._session = ClientSession(
            connector=connector,
            timeout=ClientTimeout(total=None, connect=30),
            auto_decompress=False,
        )
        for route in self.routes:
            app = web.Application(client_max_size=1024 ** 3)
            app.router.add_route("*", "/{path:.*}", self._handler(route))
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, self.host, route.local_port).start()
            self._runners.append(runner)

    async def close(self) -> None:
        for runner in self._runners:
            await runner.cleanup()
        if self._session:
            await self._session.close()

    def _handler(self, route: ProxyRoute):
        async def handle(request: web.Request) -> web.StreamResponse:
            if request.headers.get("Upgrade", "").lower() == "websocket":
                return await self._proxy_websocket(request, route)
            return await self._proxy_http(request, route)
        return handle

    def _target(self, request: web.Request, route: ProxyRoute) -> str:
        path = request.raw_path or "/"
        if route.upstream_prefix:
            path = f"/{route.upstream_prefix.strip('/')}{path}"
        return proxy_url(
            self.base_url, self.sandbox_id, route.sandbox_port, path
        )

    def _headers(self, source, websocket: bool = False) -> dict[str, str]:
        blocked = HOP_HEADERS | (WEBSOCKET_HANDSHAKE_HEADERS if websocket else set())
        headers = {key: value for key, value in source.items()
                   if key.lower() not in blocked and key.lower() != "xrl-authorization"}
        if self.api_key:
            headers["XRL-Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _proxy_http(self, request: web.Request, route: ProxyRoute) -> web.StreamResponse:
        assert self._session is not None
        body = await request.read()
        async with self._session.request(
            request.method, self._target(request, route), data=body or None,
            headers=self._headers(request.headers), allow_redirects=False,
        ) as upstream:
            headers = {key: value for key, value in upstream.headers.items()
                       if key.lower() not in HOP_HEADERS}
            if "json" in upstream.headers.get("Content-Type", ""):
                content = await upstream.read()
                if _is_platform_failure(upstream.headers, content):
                    return web.Response(status=502, body=content, content_type="application/json")
                if route.rewrite_cdp:
                    content = rewrite_cdp_urls(content, route.local_port)
                return web.Response(status=upstream.status, body=content, headers=headers)
            response = web.StreamResponse(status=upstream.status, headers=headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(1024 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response

    async def _proxy_websocket(self, request: web.Request, route: ProxyRoute) -> web.WebSocketResponse:
        assert self._session is not None
        target = websocket_url(self._target(request, route))
        offered = tuple(
            item.strip() for item in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if item.strip()
        )
        async with self._session.ws_connect(
            target, headers=self._headers(request.headers, websocket=True), protocols=offered,
        ) as upstream:
            selected = (upstream.protocol,) if upstream.protocol else ()
            client = web.WebSocketResponse(protocols=selected)
            await client.prepare(request)
            await _relay_websockets(client, upstream)
        return client


def _is_platform_failure(headers, body: bytes) -> bool:
    if "json" not in headers.get("Content-Type", ""):
        return False
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "Failed" and "result" in payload


async def _relay_websockets(left, right) -> None:
    async def forward(source, destination):
        async for message in source:
            if message.type == WSMsgType.TEXT:
                await destination.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await destination.send_bytes(message.data)
            elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break

    tasks = [asyncio.create_task(forward(left, right)),
             asyncio.create_task(forward(right, left))]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)
