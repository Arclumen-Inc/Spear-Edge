# Spear Manager (Jetson / Pi host)

**Spear Manager** is a small FastAPI service (default port **8081**) that controls **systemd** units on the same machine—Tripwire, SPEAR-Edge, ingest, and (with the SPEAR-Edge install script) **Kismet** on demand.

It is **not** the same repository as SPEAR-Edge. On your machine the deploy bundle lives under:

`/home/spear/spear_manager/`  
(typically `spear_manager_deploy.tar.gz` plus optional notes.)

## Install from the tarball + SPEAR-Edge helper

From the **spear-edgev1_0** repo root:

```bash
bash scripts/install_spear_manager.sh
```

This will:

1. Extract `spear_manager_deploy.tar.gz` (default: `/home/spear/spear_manager/spear_manager_deploy.tar.gz`, override with `SPEAR_MANAGER_TAR=`).
2. Install into `SPEAR_MANAGER_HOME` (default: `/home/spear/spear_manager`).
3. Copy **`kismet_systemd.py`** and patch **`main.py`** so these routes exist for the `/wifi` page:
   - `GET /kismet/status`
   - `POST /kismet/start` → `systemctl start kismet.service` (does **not** `enable` → **not** on boot)
   - `POST /kismet/stop` → `systemctl stop kismet.service`
4. Create/update a **venv** and `pip install -e .`

Then install the unit file and optional token env:

```bash
sudo mkdir -p /etc/spear-manager
echo 'SPEAR_MANAGER_TOKEN=your-secret' | sudo tee /etc/spear-manager/env   # optional; empty = no auth

sudo cp /home/spear/spear_manager/systemd/spear-manager.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now spear-manager.service
sudo systemctl status spear-manager.service --no-pager
```

## SPEAR-Edge configuration

In `spear-edge.service` (or environment):

```ini
Environment=SPEAR_WIFI_MANAGER_URL=http://127.0.0.1:8081
Environment=SPEAR_WIFI_MANAGER_TOKEN=your-secret
```

If `SPEAR_MANAGER_TOKEN` is unset/empty on the manager, **Bearer auth is disabled** and SPEAR can omit `SPEAR_WIFI_MANAGER_TOKEN`.

## Kismet: on-demand only

Prefer **`systemctl disable kismet.service`** after installing the `kismet` package so it does **not** start at boot. Start/stop from the **`/wifi`** page (Spear Manager buttons) or manually:

```bash
sudo systemctl start kismet.service
```

See also **`docs/KISMET_INSTALL_JETSON.md`**.

## API summary

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | no | Liveness |
| GET/POST | `/tripwire/...` | Bearer if token set | Tripwire units |
| GET/POST | `/edge/...` | Bearer if token set | Edge + ingest |
| GET | `/kismet/status` | Bearer if token set | Kismet unit state (for SPEAR UI) |
| POST | `/kismet/start` | same | Start Kismet |
| POST | `/kismet/stop` | same | Stop Kismet |

Upstream README inside the tarball also documents Tripwire/Edge endpoints.
