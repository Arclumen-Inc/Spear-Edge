# SPEAR-Edge Release Gates

This document defines the Phase 5 release-gate command and how to interpret results.

## Command

Run the automated gate suite against a running Edge instance:

`python scripts/release_gate_runner.py --base-url http://127.0.0.1:8080`

Optional output path:

`python scripts/release_gate_runner.py --out data/artifacts/release_gates/my_report.json`

The runner exits with:
- `0` when all gates pass
- `1` when any gate fails

## Evidence Output

Default report location:

`data/artifacts/release_gates/latest_release_gate_report.json`

Report fields:
- `passed`, `total`, `all_passed`
- per-gate result objects with `name`, `passed`, and `detail`

## Gate Coverage

- `openapi_reachable`
  - Verifies `/openapi.json` is reachable and non-empty.
- `health_endpoints`
  - Verifies `/health/status` and `/health/sdr` respond with valid JSON.
- `gain_body_contract`
  - Verifies `POST /live/sdr/gain` accepts JSON body and applies.
- `smoothing_contract`
  - Verifies `POST /live/smoothing` contract.
- `requested_vs_effective_sync`
  - Verifies `/live/sdr/info` contains both requested config and effective SDR state in sync.
- `ml_activation_gate`
  - Verifies model activation safety:
    - unvalidated blocked by default
    - explicit override allowed
    - validated model allowed by default

## Notes

- This runner is a release contract gate, not a replacement for long soak/high-rate stress testing.
- Keep long soak and max-rate stress as explicit final-session gates for TRL progression.
