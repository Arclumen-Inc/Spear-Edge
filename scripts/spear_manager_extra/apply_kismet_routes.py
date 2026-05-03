#!/usr/bin/env python3
"""Idempotently add /kismet/* routes to spear_manager.main (for SPEAR-Edge /wifi)."""
from __future__ import annotations

import pathlib
import sys

MARKER = "# --- kismet routes (SPEAR-Edge) ---\n"

INJECT_IMPORTS = """from spear_manager.kismet_systemd import kismet_status, start_kismet, stop_kismet
"""

INJECT_ROUTES = (
    MARKER
    + """
@app.get("/kismet/status", dependencies=[Depends(require_auth)])
def kismet_status_api():
    return kismet_status()


@app.post("/kismet/start", dependencies=[Depends(require_auth)])
def kismet_start():
    ok, msg = start_kismet()
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    return {"ok": True, "message": msg}


@app.post("/kismet/stop", dependencies=[Depends(require_auth)])
def kismet_stop():
    ok, msg = stop_kismet()
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    return {"ok": True, "message": msg}


"""
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: apply_kismet_routes.py <path-to-main.py>", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("already patched")
        return 0
    needle = "from spear_manager.systemd_ctl import ("
    idx = text.find(needle)
    if idx == -1:
        print("unexpected main.py layout: import block not found", file=sys.stderr)
        return 1
    # Insert after the closing paren of that import (after stop_tripwire line block)
    end = text.find(")", idx)
    if end == -1:
        print("unexpected main.py: import paren", file=sys.stderr)
        return 1
    # find line end after import group
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = end
    insert_at = line_end + 1
    text = text[:insert_at] + "\n" + INJECT_IMPORTS + text[insert_at:]
    run_idx = text.find("\ndef run():")
    if run_idx == -1:
        print("unexpected main.py: def run() not found", file=sys.stderr)
        return 1
    text = text[:run_idx] + "\n" + INJECT_ROUTES + text[run_idx:]
    path.write_text(text, encoding="utf-8")
    print("patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
