#!/usr/bin/env bash
# Install Spear Manager from the deploy tarball under /home/spear/spear_manager
# and merge SPEAR-Edge Kismet control routes (systemctl start/stop kismet.service).
#
# Prerequisites: Python 3.10+, systemd, tarball at SPEAR_MANAGER_TAR (see below).
# Usage:
#   bash scripts/install_spear_manager.sh
# Or:
#   SPEAR_MANAGER_HOME=/opt/spear_manager SPEAR_MANAGER_TAR=/path/to.tgz sudo -E bash scripts/install_spear_manager.sh
#
# After install:
#   sudo mkdir -p /etc/spear-manager && echo 'SPEAR_MANAGER_TOKEN=secret' | sudo tee /etc/spear-manager/env
#   sudo systemctl enable --now spear-manager
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRA="${REPO_ROOT}/scripts/spear_manager_extra"

DEST="${SPEAR_MANAGER_HOME:-/home/spear/spear_manager}"
TAR="${SPEAR_MANAGER_TAR:-${DEST}/spear_manager_deploy.tar.gz}"

if [[ ! -f "$TAR" ]]; then
  echo "Tarball not found: $TAR" >&2
  echo "Set SPEAR_MANAGER_TAR= or place spear_manager_deploy.tar.gz in ${DEST}/" >&2
  exit 1
fi
if [[ ! -f "${EXTRA}/kismet_systemd.py" ]] || [[ ! -f "${EXTRA}/apply_kismet_routes.py" ]]; then
  echo "Missing ${EXTRA} files (kismet_systemd.py / apply_kismet_routes.py)" >&2
  exit 1
fi

echo "Extracting -> ${DEST}"
mkdir -p "${DEST}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
# Avoid chown to tarball uid (fails on some FS / containers)
tar --no-same-owner -xzf "${TAR}" -C "${TMP}"
# tarball layout: spear_manager/{pyproject.toml,spear_manager/,...}
if [[ ! -d "${TMP}/spear_manager" ]]; then
  echo "Unexpected tarball layout (expected top-level spear_manager/)" >&2
  exit 1
fi
cp -a "${TMP}/spear_manager/." "${DEST}/"

echo "Adding Kismet systemd helpers + API routes for SPEAR-Edge /wifi"
cp -a "${EXTRA}/kismet_systemd.py" "${DEST}/spear_manager/kismet_systemd.py"
python3 "${EXTRA}/apply_kismet_routes.py" "${DEST}/spear_manager/main.py"

echo "Python venv + install"
cd "${DEST}"
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
./venv/bin/pip install -q -U pip wheel
./venv/bin/pip install -q -r requirements.txt
./venv/bin/pip install -q -e .

echo "systemd unit (review paths, then enable)"
UNIT_SRC="${DEST}/systemd/spear-manager.service"
if [[ -f "$UNIT_SRC" ]]; then
  echo "Copy with: sudo cp ${UNIT_SRC} /etc/systemd/system/"
  echo "Then: sudo systemctl daemon-reload && sudo systemctl enable --now spear-manager"
else
  echo "Warning: no systemd/spear-manager.service under ${DEST}"
fi

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files kismet.service 2>/dev/null | grep -q '^kismet\.service'; then
    echo ""
    echo "Kismet: keeping package installed but disabling boot start (on-demand via Spear Manager / manual)."
    echo "Run: sudo systemctl disable kismet.service"
  fi
fi

echo ""
echo "Done. Point SPEAR-Edge at: SPEAR_WIFI_MANAGER_URL=http://127.0.0.1:8081"
echo "Docs: ${REPO_ROOT}/docs/SPEAR_MANAGER.md"
