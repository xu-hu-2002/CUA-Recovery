from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

Transport = Callable[[str, str, Optional[Mapping[str, Any]]], Dict[str, Any]]


class ControlApiError(RuntimeError):
    pass


def urllib_transport(base_url: str, timeout_s: float) -> Transport:
    base = base_url.rstrip("/")

    def call(method: str, path: str, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            raise ControlApiError("%s %s -> HTTP %s" % (method, path, exc.code)) from exc
        except urllib.error.URLError as exc:
            raise ControlApiError("%s %s -> %s" % (method, path, exc.reason)) from exc

    return call


class RecoveryControlClient:
    def __init__(
        self, base_url: str, timeout_s: float = 30.0, transport: Optional[Transport] = None
    ):
        self.base_url = base_url
        self._call = transport or urllib_transport(base_url, timeout_s)

    def set_cursor(self, action_index: int) -> Dict[str, Any]:
        return self._call("POST", "/recovery/cursor", {"action_index": int(action_index)})

    def latest_seq(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self._call("GET", "/recovery/seq", None).get("seq", {}).items()}

    def changelog(self, since: Mapping[str, int]) -> List[Dict[str, Any]]:
        query = urllib.parse.quote(json.dumps(dict(since)))
        return list(self._call("GET", "/recovery/changelog?since=%s" % query, None).get("rows", []))

    def observations(self, action_index: int) -> List[Dict[str, Any]]:
        return list(
            self._call("GET", "/recovery/observations?action_index=%d" % int(action_index), None).get(
                "records", []
            )
        )

    def digest(self) -> Dict[str, str]:
        return dict(self._call("GET", "/recovery/digest", None))

    def status(self) -> Dict[str, Any]:
        return dict(self._call("GET", "/recovery/status", None))

    def install(self) -> Dict[str, Any]:
        return dict(self._call("POST", "/recovery/install", {}))
