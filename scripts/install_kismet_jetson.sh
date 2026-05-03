#!/usr/bin/env bash
# Add official Kismet apt repo and install kismet (Ubuntu LTS commonly used on Jetson).
# Usage: sudo bash scripts/install_kismet_jetson.sh
# Docs: docs/KISMET_INSTALL_JETSON.md
set -euo pipefail

if [[ "${EUID:-}" -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

if ! test -r /etc/os-release; then
  echo "/etc/os-release not found; cannot detect distro." >&2
  exit 1
fi
# shellcheck source=/dev/null
source /etc/os-release

CODENAME="${VERSION_CODENAME:-}"
case "$CODENAME" in
  jammy|noble|focal|bionic|plucky) ;;
  *)
    echo "Unsupported VERSION_CODENAME='$CODENAME' for automated repo line." >&2
    echo "Pick the correct block from https://www.kismetwireless.net/packages/ and install manually." >&2
    exit 1
    ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg

KEY=/usr/share/keyrings/kismet-archive-keyring.gpg
wget -O - https://www.kismetwireless.net/repos/kismet-release.gpg.key --quiet \
  | gpg --dearmor | tee "$KEY" >/dev/null

echo "deb [signed-by=${KEY}] https://www.kismetwireless.net/repos/apt/release/${CODENAME} ${CODENAME} main" \
  > /etc/apt/sources.list.d/kismet.list

apt-get update
apt-get install -y kismet

echo ""
echo "Kismet packages installed."
echo "1) Add users to the kismet group:  sudo usermod -aG kismet <user>   (then re-login or reboot)"
echo "2) On-demand (recommended):  sudo systemctl disable kismet.service 2>/dev/null || true"
echo "   Start when needed:  sudo systemctl start kismet   OR use Spear Manager /wifi (see docs/SPEAR_MANAGER.md)"
echo "3) Verify while running:  curl -I http://127.0.0.1:2501"
