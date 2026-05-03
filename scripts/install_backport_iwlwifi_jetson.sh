#!/bin/bash
# Install Intel iwlwifi DKMS backport (11510) on Jetson with *-tegra kernel.
# Expects offline debs under /home/spear/wifi-debs* (adjust WIFI_DEBS / WIFI_DEBS_DEPS).
# Does NOT install linux-headers-5.15.0-177-* from those folders (wrong ABI for tegra).
#
# Offline Ubuntu 22.04: wifi-debs-dkms-deps must include lto-disabled-list (dpkg-dev
# depends on it). If dpkg stops, download that deb on an online Jetson and merge here.
#
# Usage: sudo bash scripts/install_backport_iwlwifi_jetson.sh
# After success: sudo reboot  (clean load of dkms mac80211/cfg80211/iwlwifi)

set -euo pipefail

WIFI_DEBS="${WIFI_DEBS:-/home/spear/wifi-debs}"
WIFI_DEBS_DEPS="${WIFI_DEBS_DEPS:-/home/spear/wifi-debs-dkms-deps}"

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "Run as root: sudo bash $0"
  exit 1
fi

K="$(uname -r)"
if [ ! -d "/lib/modules/${K}/build" ]; then
  echo "ERROR: /lib/modules/${K}/build missing — install tegra kernel headers for ${K}, then re-run."
  exit 1
fi

echo "Kernel: ${K}"
echo "[1/7] DKMS helper debs from ${WIFI_DEBS_DEPS}"
dpkg -i "${WIFI_DEBS_DEPS}"/*.deb
echo "[2/7] dpkg --configure -a (finishes partial installs; matches proven AX200 flow)"
dpkg --configure -a || true

echo "[3/7] GCC 12 from ${WIFI_DEBS} (Ubuntu 22.04 dkms depends on gcc-12; needed offline)"
(
  cd "${WIFI_DEBS}" || exit 1
  dpkg -i \
    gcc-12-base_*.deb \
    libgcc-s1_*.deb \
    cpp-12_*.deb \
    libcc1-0_*.deb \
    libatomic1_*.deb \
    libasan8_*.deb \
    libgcc-12-dev_*.deb \
    libgomp1_*.deb \
    libitm1_*.deb \
    liblsan0_*.deb \
    libhwasan0_*.deb \
    libtsan2_*.deb \
    libubsan1_*.deb \
    libstdc++6_*.deb \
    libgfortran5_*.deb \
    gcc-12_*.deb
)

echo "[4/7] dkms"
dpkg -i "${WIFI_DEBS}"/dkms_*.deb

echo "[5/7] backport-iwlwifi-dkms"
dpkg -i "${WIFI_DEBS}"/backport-iwlwifi-dkms_*.deb

echo "[6/7] apt --fix-broken (ok if offline / nothing to do)"
apt-get install -f -y || true
dpkg --configure -a || true

echo "[7/7] DKMS status"
dkms status || true

if ! dkms status 2>/dev/null | grep -q "backport-iwlwifi/11510.*${K}.*installed"; then
  echo "DKMS did not show installed; trying explicit build..."
  dkms install backport-iwlwifi/11510 -k "${K}" || true
  dkms status || true
fi

echo ""
echo "Done. Optional (AX210 / newer firmware): apt install -y linux-firmware"
echo "Reboot recommended: sudo reboot"
echo "Then (if swapping module): power off, swap Wi-Fi, boot — check: sudo dmesg | grep -i iwl"
