#!/usr/bin/env bash
# Install SPEAR Ingest (:8000) and Edge (:8080) systemd units.
# Does NOT enable on boot — start manually:
#   sudo systemctl start spear-edge   # also starts spear-ingest (Wants)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_SRC="${ROOT}/scripts/spear-edge.service"
EDGE_DST="/etc/systemd/system/spear-edge.service"
INGEST_SRC="${ROOT}/scripts/spear-ingest.service"
INGEST_DST="/etc/systemd/system/spear-ingest.service"

if [[ ! -f "${ROOT}/venv/bin/uvicorn" ]]; then
  echo "ERROR: ${ROOT}/venv/bin/uvicorn not found." >&2
  echo "Create the venv first, e.g.:" >&2
  echo "  cd ${ROOT} && python3 -m venv venv && ./venv/bin/pip install -U pip && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

sudo install -m 644 "$INGEST_SRC" "$INGEST_DST"
sudo install -m 644 "$EDGE_SRC" "$EDGE_DST"
sudo systemctl daemon-reload
sudo systemctl disable spear-ingest.service 2>/dev/null || true
sudo systemctl disable spear-edge.service 2>/dev/null || true

echo "Installed:"
echo "  $INGEST_DST  (uvicorn ingest_app → 0.0.0.0:8000)"
echo "  $EDGE_DST    (uvicorn app → 0.0.0.0:8080)"
echo "Boot: both disabled until you: sudo systemctl enable spear-edge spear-ingest"
echo "Run:  sudo systemctl start spear-edge   # pulls up spear-ingest first (Wants)"
echo "Or:   sudo systemctl start spear-ingest && sudo systemctl start spear-edge"
echo "Stop: sudo systemctl stop spear-edge spear-ingest"
echo "Logs: journalctl -u spear-edge -u spear-ingest -f"
