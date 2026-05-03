#!/bin/bash
# One-time: grants NOPASSWD sudo for the invoking user (Jetson / dev convenience).
# Run:   bash scripts/enable_passwordless_sudo.sh
# You will be prompted for your password once; after that, sudo is passwordless.
#
# Security: anyone with shell access as this user can run root commands. Use only
# on machines where you accept that tradeoff.

set -euo pipefail

TARGET_USER="${SUDO_USER:-$USER}"
if [ -z "${TARGET_USER}" ] || [ "${TARGET_USER}" = root ]; then
  echo "Run as your normal login user (not a root-only session). Got: TARGET_USER=${TARGET_USER:-empty}"
  exit 1
fi

DROPIN="/etc/sudoers.d/99-${TARGET_USER}-nopasswd"
LINE="${TARGET_USER} ALL=(ALL:ALL) NOPASSWD: ALL"

echo "Installing ${DROPIN}"
echo "${LINE}" | sudo tee "${DROPIN}" >/dev/null
sudo chmod 0440 "${DROPIN}"
sudo visudo -c -f "${DROPIN}"

echo "Syntax OK. New shells should have passwordless sudo for ${TARGET_USER}."
echo "Verify: sudo -n true && echo 'sudo -n: OK'"
