"""Start/stop/status for kismet.service via systemd (on-demand; do not enable at boot)."""
from __future__ import annotations

import shutil
import subprocess
from typing import Literal

KISMET_UNIT = "kismet.service"


def _get_state(unit: str) -> str:
    r = subprocess.run(
        ["systemctl", "show", unit, "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if r.returncode != 0:
        return "unknown"
    return (r.stdout or "").strip() or "unknown"


def _get_sub_state(unit: str) -> str:
    r = subprocess.run(
        ["systemctl", "show", unit, "--property=SubState", "--value"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if r.returncode != 0:
        return "unknown"
    return (r.stdout or "").strip() or "unknown"


def _systemctl(
    action: Literal["start", "stop"],
    unit: str,
    timeout_sec: int = 60,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", action, unit],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def start_kismet(timeout_sec: int = 60) -> tuple[bool, str]:
    """systemctl start kismet.service — does not enable the unit (no boot)."""
    if shutil.which("systemctl") is None:
        return False, "systemctl not found"
    r = _systemctl("start", KISMET_UNIT, timeout_sec=timeout_sec)
    if r.returncode == 0:
        return True, "started"
    return False, r.stderr or r.stdout or f"exit {r.returncode}"


def stop_kismet(timeout_sec: int = 60) -> tuple[bool, str]:
    if shutil.which("systemctl") is None:
        return False, "systemctl not found"
    r = _systemctl("stop", KISMET_UNIT, timeout_sec=timeout_sec)
    if r.returncode == 0:
        return True, "stopped"
    return False, r.stderr or r.stdout or f"exit {r.returncode}"


def kismet_status() -> dict:
    """Shape expected by SPEAR-Edge /wifi UI (renderManagerKismetResult)."""
    state = _get_state(KISMET_UNIT)
    sub = _get_sub_state(KISMET_UNIT)
    return {
        "service": "kismet",
        "unit": KISMET_UNIT,
        "state": state,
        "sub_state": sub,
        "ok": state == "active",
    }
