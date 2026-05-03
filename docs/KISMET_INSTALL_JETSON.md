# Installing Kismet for SPEAR-Edge (Jetson / Ubuntu)

SPEAR’s Wi‑Fi monitor (`/wifi`, `KismetProvider`) expects a **running Kismet server** with the REST API on port **2501** (default `http://127.0.0.1:2501`). If Kismet is not installed, **Test Kismet Connection** and live intel will fail until you install and start it.

Official package index (pick the line that matches **your** Ubuntu/Debian codename):  
https://www.kismetwireless.net/packages/

## Quick install (Ubuntu 22.04 Jammy or 24.04 Noble on Jetson)

Use the **release** repository (stable). Replace `jammy` with your codename from `lsb_release -cs` if different.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

wget -O - https://www.kismetwireless.net/repos/kismet-release.gpg.key --quiet \
  | gpg --dearmor | sudo tee /usr/share/keyrings/kismet-archive-keyring.gpg >/dev/null

# Jammy (22.04):
echo 'deb [signed-by=/usr/share/keyrings/kismet-archive-keyring.gpg] https://www.kismetwireless.net/repos/apt/release/jammy jammy main' | sudo tee /etc/apt/sources.list.d/kismet.list >/dev/null

# Noble (24.04) — use instead of jammy when on Noble:
# echo 'deb [signed-by=/usr/share/keyrings/kismet-archive-keyring.gpg] https://www.kismetwireless.net/repos/apt/release/noble noble main' | sudo tee /etc/apt/sources.list.d/kismet.list >/dev/null

sudo apt-get update
sudo apt-get install -y kismet
```

During install, prefer **suid-root capture helpers** when the package prompts (recommended by Kismet for Wi‑Fi capture).

Add users that should run Kismet or SPEAR against local Kismet to the **`kismet`** group, then **log out and back in** (or reboot):

```bash
sudo usermod -aG kismet "$USER"
```

## On-demand start (recommended — not on boot)

Avoid enabling Kismet at boot unless you want it always running. **Spear Manager** (see `docs/SPEAR_MANAGER.md`) can start/stop `kismet.service` from the SPEAR `/wifi` UI, or start manually:

```bash
sudo systemctl disable kismet.service 2>/dev/null || true   # package may have enabled it
sudo systemctl start kismet.service
sudo systemctl status kismet.service --no-pager
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:2501/
```

You should see HTTP **401** or **200** (not connection refused) while Kismet is running.

## SPEAR configuration

- Set **`SPEAR_WIFI_MONITOR_KISMET_URL`** if Kismet runs elsewhere (default remains `http://127.0.0.1:2501`).
- Wi‑Fi interface default in SPEAR is **`wlP1p1s0`** (Jetson built-in); USB adapters use names like `wlx…` — set **`SPEAR_WIFI_MONITOR_IFACE`** accordingly.

## Optional: scripted install

From the repo root (requires `sudo`):

```bash
sudo bash scripts/install_kismet_jetson.sh
```

The script only supports Ubuntu codenames that have official Kismet **release** packages; for others, follow the packages page manually.

## Without Kismet

You can still run SPEAR; only the **Wi‑Fi / RID-over-Wi‑Fi** path stays idle. The bladeRF spectrum/capture path is unrelated. Alternatively configure **`SPEAR_WIFI_MONITOR_KISMET_CMD`** with a command that prints JSON (advanced; normal path is HTTP to Kismet).
