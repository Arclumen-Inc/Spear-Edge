#!/usr/bin/env python3
"""
Release gate runner for SPEAR-Edge.

Runs a deterministic set of API/contract/runtime checks and emits
machine-readable evidence JSON for go/no-go decisions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request, error


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: Dict[str, Any]


class ReleaseGateRunner:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.results: List[GateResult] = []

    def _get(self, path: str, timeout: float = 8.0) -> Tuple[int, Dict[str, Any]]:
        with request.urlopen(self.base_url + path, timeout=timeout) as r:
            body = r.read().decode("utf-8", "ignore")
            return r.status, json.loads(body) if body else {}

    def _post(self, path: str, payload: Dict[str, Any] | None = None, timeout: float = 10.0) -> Tuple[int, Dict[str, Any]]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if payload is None else {"Content-Type": "application/json"}
        req = request.Request(self.base_url + path, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "ignore")
                return r.status, json.loads(body) if body else {}
        except error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            try:
                obj = json.loads(body) if body else {}
            except Exception:
                obj = {"raw": body}
            return e.code, obj
        except Exception as e:
            return 599, {"error": f"transport_error: {e}"}

    def _record(self, name: str, passed: bool, detail: Dict[str, Any]) -> None:
        self.results.append(GateResult(name=name, passed=passed, detail=detail))

    def run_core_api_gates(self) -> None:
        # Gate: OpenAPI reachable
        try:
            code, schema = self._get("/openapi.json")
            self._record(
                "openapi_reachable",
                code == 200 and isinstance(schema, dict) and len(schema.get("paths", {})) > 0,
                {"code": code, "paths": len(schema.get("paths", {}))},
            )
        except Exception as e:
            self._record("openapi_reachable", False, {"error": str(e)})

        # Gate: health endpoints
        try:
            c1, h1 = self._get("/health/status")
            c2, h2 = self._get("/health/sdr")
            ok = c1 == 200 and c2 == 200 and isinstance(h1, dict) and isinstance(h2, dict)
            self._record("health_endpoints", ok, {"status_code": c1, "sdr_code": c2, "status": h1, "sdr": h2})
        except Exception as e:
            self._record("health_endpoints", False, {"error": str(e)})

    def run_phase13_contract_gates(self) -> None:
        # Start scan for deterministic checks
        start_code, start_resp = self._post(
            "/live/start",
            {"center_freq_hz": 915_000_000, "sample_rate_sps": 2_400_000, "fft_size": 2048, "fps": 15.0},
        )
        time.sleep(1.0)

        # Gain body contract gate
        gain_code, gain_resp = self._post("/live/sdr/gain", {"gain_mode": "manual", "gain_db": 8.0})
        gain_ok = gain_code == 200 and bool(gain_resp.get("ok"))
        self._record(
            "gain_body_contract",
            gain_ok,
            {"start_code": start_code, "start_resp": start_resp, "gain_code": gain_code, "gain_resp": gain_resp},
        )

        # Smoothing contract gate
        sm_code, sm_resp = self._post("/live/smoothing", {"alpha": 0.22})
        sm_ok = sm_code == 200 and bool(sm_resp.get("ok")) and abs(float(sm_resp.get("alpha", -1.0)) - 0.22) < 1e-6
        self._record("smoothing_contract", sm_ok, {"code": sm_code, "resp": sm_resp})

        # Requested vs effective sync
        info_code, info = self._get("/live/sdr/info")
        req_cfg = info.get("current_config", {}) if isinstance(info, dict) else {}
        eff = info.get("effective_state", {}) if isinstance(info, dict) else {}
        sync_ok = (
            info_code == 200
            and isinstance(eff, dict)
            and abs(int(req_cfg.get("center_freq_hz", 0)) - int(eff.get("center_freq_hz", 0))) <= 1000
            and abs(int(req_cfg.get("sample_rate_sps", 0)) - int(eff.get("sample_rate_sps", 0))) <= 1000
            and abs(float(req_cfg.get("gain_db", 0.0)) - float(eff.get("gain_db", 0.0))) <= 0.5
        )
        self._record(
            "requested_vs_effective_sync",
            sync_ok,
            {"info_code": info_code, "current_config": req_cfg, "effective_state": eff},
        )

        # Restore gain and stop scan
        self._post("/live/sdr/gain", {"gain_mode": "manual", "gain_db": 0.0})
        self._post("/live/stop")

    def run_phase4_ml_gate(self) -> None:
        # Build a local unvalidated model copy from validated sample if available
        validated_model = Path("/home/spear/spear-edgev1_0/data/temp_phase4/model_out.pth")
        invalid_model = Path("/home/spear/spear-edgev1_0/data/temp_phase4/invalid_model_gate.pth")

        if not validated_model.exists():
            self._record(
                "ml_activation_gate",
                False,
                {"error": f"validated sample model not found at {validated_model}"},
            )
            return

        invalid_model.parent.mkdir(parents=True, exist_ok=True)
        invalid_model.write_bytes(validated_model.read_bytes())
        invalid_validation = invalid_model.with_suffix(".validation.json")
        if invalid_validation.exists():
            invalid_validation.unlink()

        # Default block for unvalidated
        c1, r1 = self._post("/api/ml/models/activate", {"model_path": str(invalid_model)})
        blocked_ok = (c1 == 400) and ("blocked" in str(r1).lower())

        # Explicit override allow for unvalidated
        c2, r2 = self._post(
            "/api/ml/models/activate",
            {"model_path": str(invalid_model), "allow_unvalidated": True},
        )
        override_ok = c2 == 200 and bool(r2.get("ok"))

        # Validated default allow
        c3, r3 = self._post("/api/ml/models/activate", {"model_path": str(validated_model)})
        validated_ok = c3 == 200 and bool(r3.get("ok")) and bool((r3.get("validation") or {}).get("validated"))

        self._record(
            "ml_activation_gate",
            blocked_ok and override_ok and validated_ok,
            {
                "unvalidated_default": {"code": c1, "resp": r1},
                "unvalidated_override": {"code": c2, "resp": r2},
                "validated_default": {"code": c3, "resp": r3},
            },
        )

    def run_all(self) -> Dict[str, Any]:
        self.run_core_api_gates()
        self.run_phase13_contract_gates()
        self.run_phase4_ml_gate()
        summary = {
            "base_url": self.base_url,
            "timestamp_unix": time.time(),
            "passed": sum(1 for r in self.results if r.passed),
            "total": len(self.results),
            "all_passed": all(r.passed for r in self.results) if self.results else False,
            "results": [asdict(r) for r in self.results],
        }
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SPEAR-Edge release gates and emit evidence JSON.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Edge base URL")
    parser.add_argument(
        "--out",
        default="data/artifacts/release_gates/latest_release_gate_report.json",
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    runner = ReleaseGateRunner(args.base_url)
    report = runner.run_all()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[RELEASE GATE] Report written to {out_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
