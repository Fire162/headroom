"""Offline checks for the Copilot enterprise test kit scripts in tools/copilot-test."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from headroom import copilot_auth

KIT = Path(__file__).resolve().parents[1] / "tools" / "copilot-test"
LEAKED = "gho_ECHOEDSECRET123"


class _FakeProvider:
    def __init__(self, token: str | None) -> None:
        self.token = token

    async def get_api_token(self, *, integration_id: str | None = None) -> SimpleNamespace:
        if self.token is None:
            raise RuntimeError("no credential")
        return SimpleNamespace(token=self.token, api_url="https://api.example.test")


class _FakeClient:
    """Stands in for httpx.Client; records every request the doctor makes."""

    requests: list[tuple[str, dict[str, str]]] = []
    broken_model: str | None = None  # this model's reply is malformed JSON

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        self.requests.append((url, dict(headers or {})))
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})

    def post(self, url: str, headers: dict[str, str] | None = None, json: object = None):
        self.requests.append((url, dict(headers or {})))
        model = json["model"]  # type: ignore[index]
        if model == self.broken_model:
            return httpx.Response(
                200, content=b"<html>", headers={"content-type": "application/json"}
            )
        return httpx.Response(400, json={"error": {"message": f"bad token {LEAKED}"}})


def _run_doctor(
    monkeypatch: pytest.MonkeyPatch, token: str | None, broken_model: str | None = None
) -> list[tuple[str, dict]]:
    _FakeClient.requests = []
    _FakeClient.broken_model = broken_model
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "my-cli")
    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _FakeProvider(token))
    monkeypatch.setattr(copilot_auth, "iter_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "read_cached_oauth_token", lambda: None)
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    runpy.run_path(str(KIT / "copilot_doctor.py"), run_name="__main__")
    return _FakeClient.requests


def test_doctor_probes_with_the_token_integration_id(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _run_doctor(monkeypatch, "tid_minted_for_my_cli")

    assert requests  # catalog + inference probes ran
    assert {h["Copilot-Integration-Id"] for _, h in requests} == {"my-cli"}


def test_doctor_survives_a_failed_probe_and_redacts_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_doctor(monkeypatch, "tid_minted_for_my_cli", broken_model="gpt-5.5")
    out = capsys.readouterr().out

    # gpt-5.5 answered with malformed JSON, yet claude was still probed and the verdict printed.
    assert "claude-sonnet-4.6" in out.split("[7]")[1]
    assert "VERDICT" in out
    assert "request error: JSONDecodeError" in out
    # The upstream error echoed a credential; only its type prefix may be shown.
    assert "gho_…" in out
    assert "ECHOEDSECRET" not in out


def test_doctor_reports_no_token_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_doctor(monkeypatch, None)
    out = capsys.readouterr().out

    assert "Token forwarded     : none" in out


def test_harness_refuses_an_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))

    def no_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not start a proxy on an occupied port")

    monkeypatch.setattr("subprocess.Popen", no_spawn)

    with pytest.raises(SystemExit, match="already in use"):
        runpy.run_path(str(KIT / "enterprise_proxy_test.py"), run_name="__main__")


def test_harness_stops_when_its_proxy_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the spawned proxy dies, a server that comes up later must not be probed."""

    def port_free(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    exited = SimpleNamespace(poll=lambda: 1, terminate=lambda: None, wait=lambda timeout: 1)
    monkeypatch.setattr(httpx, "get", port_free)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: exited)
    _FakeClient.requests = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    with pytest.raises(SystemExit, match="did not become ready"):
        runpy.run_path(str(KIT / "enterprise_proxy_test.py"), run_name="__main__")
    assert _FakeClient.requests == []
