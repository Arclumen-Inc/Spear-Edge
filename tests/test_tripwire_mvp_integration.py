"""
Regression tests for Tripwire MVP integration: ingest shapes, FHSS span aliases, audio ignore.
Uses a minimal FastAPI app + mock orchestrator (no hardware).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spear_edge.api.http.routes_tripwire import bind
from spear_edge.core.integrate.tripwire_events import FhssClusterEvent, parse_tripwire_event


@pytest.fixture
def mock_orch_armed_capture_ok():
    """Armed mode, auto-capture policy always allows (exercise capture path)."""
    o = MagicMock()
    o.mode = "armed"
    o.tripwire_cues = []
    o.aoa_cones = []
    o.capture_mgr = MagicMock()
    o.capture_mgr.submit_nowait.return_value = True
    o.bus = MagicMock()
    o.tripwires = MagicMock()
    o.tripwires.snapshot.return_value = []
    o.can_auto_capture = lambda _p: (True, "ok")
    o.mark_auto_capture = MagicMock()
    o.record_tripwire_cue = MagicMock()
    o._record_aoa_cone = MagicMock()
    return o


@pytest.fixture
def client_armed(mock_orch_armed_capture_ok):
    app = FastAPI()
    app.include_router(bind())
    app.state.orchestrator = mock_orch_armed_capture_ok
    return TestClient(app), mock_orch_armed_capture_ok


def test_parse_fhss_cluster_freq_span_mhz_coalesces_to_span_mhz():
    ev = parse_tripwire_event(
        {
            "type": "fhss_cluster",
            "freq_hz": 915e6,
            "freq_span_mhz": 2.5,
            "timestamp": 1.0,
        }
    )
    assert isinstance(ev, FhssClusterEvent)
    assert ev.span_mhz == 2.5


def test_fhss_cluster_model_direct():
    m = FhssClusterEvent(
        type="fhss_cluster",
        freq_hz=915e6,
        freq_span_mhz=3.0,
        timestamp=1.0,
    )
    assert m.span_mhz == 3.0


def test_audio_cue_ignored_no_cue_record(client_armed):
    client, orch = client_armed
    r = client.post(
        "/api/tripwire/event",
        json={
            "type": "audio_cue",
            "snr_db": 12.0,
            "timestamp": 1.0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("action") == "ignored_non_rf"
    orch.record_tripwire_cue.assert_not_called()
    orch.capture_mgr.submit_nowait.assert_not_called()


def test_fhss_cluster_freq_span_mhz_capture_meta_and_bandwidth(client_armed):
    client, orch = client_armed
    r = client.post(
        "/api/tripwire/event",
        json={
            "schema": "spear.tripwire.event.v1",
            "type": "fhss_cluster",
            "stage": "confirmed",
            "node_id": "tw-001",
            "freq_hz": 915_000_000,
            "confidence": 0.95,
            "timestamp": 1.0,
            "freq_span_mhz": 3.2,
        },
    )
    assert r.status_code == 200
    assert r.json().get("action") == "auto_capture_started"
    orch.capture_mgr.submit_nowait.assert_called_once()
    req = orch.capture_mgr.submit_nowait.call_args[0][0]
    assert req.meta.get("span_mhz") == 3.2
    assert req.meta.get("bandwidth_hz") == int(3.2e6)


def test_aoa_cone_still_advisory(client_armed):
    client, orch = client_armed
    r = client.post(
        "/api/tripwire/event",
        json={
            "type": "aoa_cone",
            "node_id": "tw-001",
            "bearing_deg": 45.0,
            "cone_width_deg": 30.0,
            "timestamp": 1.0,
        },
    )
    assert r.status_code == 200
    assert r.json().get("action") == "aoa_update"
    orch.record_tripwire_cue.assert_called()
    orch.capture_mgr.submit_nowait.assert_not_called()


def test_scan_plan_ws_includes_use_plan():
    """set_scan_plan WebSocket JSON includes use_plan (default True)."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    sent: list = []

    async def capture_send(s: str):
        sent.append(json.loads(s))

    ws = MagicMock()
    ws.send_text = AsyncMock(side_effect=capture_send)

    orch = MagicMock()
    orch.tripwire_links = {"node-a": ws}

    app = FastAPI()
    app.include_router(bind())
    app.state.orchestrator = orch

    client = TestClient(app)
    r = client.post(
        "/api/tripwire/scan-plan",
        json={
            "node_id": "node-a",
            "scan_plan": "targeted",
            "use_plan": False,
        },
    )
    assert r.status_code == 200
    assert sent, "WebSocket send_text should have been called"
    assert sent[0]["type"] == "set_scan_plan"
    assert sent[0]["scan_plan"] == "targeted"
    assert sent[0]["use_plan"] is False
    assert "command_id" in sent[0]
