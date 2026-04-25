const wifiIfaceInput = document.getElementById("wifiIfaceInput");
const wifiBackendSelect = document.getElementById("wifiBackendSelect");
const wifiChannelModeSelect = document.getElementById("wifiChannelModeSelect");
const wifiPollInput = document.getElementById("wifiPollInput");
const wifiKismetUrlInput = document.getElementById("wifiKismetUrlInput");
const wifiKismetUserInput = document.getElementById("wifiKismetUserInput");
const wifiKismetPassInput = document.getElementById("wifiKismetPassInput");
const wifiKismetTimeoutInput = document.getElementById("wifiKismetTimeoutInput");
const wifiKismetCmdInput = document.getElementById("wifiKismetCmdInput");
const wifiHopChannelsInput = document.getElementById("wifiHopChannelsInput");
const wifiSaveConfigBtn = document.getElementById("wifiSaveConfigBtn");
const wifiTestBtn = document.getElementById("wifiTestBtn");
const wifiStartBtn = document.getElementById("wifiStartBtn");
const wifiStopBtn = document.getElementById("wifiStopBtn");
const wifiRefreshBtn = document.getElementById("wifiRefreshBtn");
const wifiExportKindSelect = document.getElementById("wifiExportKindSelect");
const wifiExportFormatSelect = document.getElementById("wifiExportFormatSelect");
const wifiExportLimitInput = document.getElementById("wifiExportLimitInput");
const wifiExportBtn = document.getElementById("wifiExportBtn");
const wifiStatusSummary = document.getElementById("wifiStatusSummary");
const wifiMgrKismetStatusBtn = document.getElementById("wifiMgrKismetStatusBtn");
const wifiMgrKismetStartBtn = document.getElementById("wifiMgrKismetStartBtn");
const wifiMgrKismetStopBtn = document.getElementById("wifiMgrKismetStopBtn");
const wifiMgrKismetSummary = document.getElementById("wifiMgrKismetSummary");

const wifiRidDetections = document.getElementById("wifiRidDetections");
const wifiTrafficStrip = document.getElementById("wifiTrafficStrip");
const wifiChannels = document.getElementById("wifiChannels");
const wifiEmitters = document.getElementById("wifiEmitters");
const wifiSources = document.getElementById("wifiSources");
const wifiDevices = document.getElementById("wifiDevices");
const wifiAnomalies = document.getElementById("wifiAnomalies");
const wifiAlertDefs = document.getElementById("wifiAlertDefs");

const wifiAddSourceInput = document.getElementById("wifiAddSourceInput");
const wifiListInterfacesBtn = document.getElementById("wifiListInterfacesBtn");
const wifiAddSourceBtn = document.getElementById("wifiAddSourceBtn");
const wifiRefreshSourcesBtn = document.getElementById("wifiRefreshSourcesBtn");
const wifiSourceUuidInput = document.getElementById("wifiSourceUuidInput");
const wifiSourceChannelInput = document.getElementById("wifiSourceChannelInput");
const wifiSourceHopInput = document.getElementById("wifiSourceHopInput");
const wifiSourceHopRateInput = document.getElementById("wifiSourceHopRateInput");
const wifiSetChannelBtn = document.getElementById("wifiSetChannelBtn");
const wifiSetHopBtn = document.getElementById("wifiSetHopBtn");
const wifiOpenSourceBtn = document.getElementById("wifiOpenSourceBtn");
const wifiCloseSourceBtn = document.getElementById("wifiCloseSourceBtn");
const wifiAlertTypeSelect = document.getElementById("wifiAlertTypeSelect");
const wifiAlertActionSelect = document.getElementById("wifiAlertActionSelect");
const wifiAlertMacInput = document.getElementById("wifiAlertMacInput");
const wifiApplyAlertBtn = document.getElementById("wifiApplyAlertBtn");
const wifiListAlertsBtn = document.getElementById("wifiListAlertsBtn");

let notifyWs = null;
const ridBuffer = [];
const trafficHistory = [];
const TRAFFIC_HISTORY_MAX = 36;

async function apiGetStatus() {
  const r = await fetch("/api/wifi-monitor/status", { cache: "no-store" });
  if (!r.ok) throw new Error("Failed to fetch Wi-Fi monitor status");
  return r.json();
}

async function apiPost(path, payload = null) {
  const r = await fetch(path, {
    method: "POST",
    headers: payload ? { "Content-Type": "application/json" } : undefined,
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(txt || `Request failed: ${path}`);
  }
  return r.json();
}

function setStatusLine(text) {
  wifiStatusSummary.textContent = text;
}

function buildExportFilename(kind, format) {
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const ext = format === "csv" ? "csv" : "jsonl";
  return `wifi_${kind}_${ts}.${ext}`;
}

async function downloadExport(kind, format, limit) {
  const qs = new URLSearchParams({
    kind,
    format,
    limit: String(limit),
  });
  const url = `/api/wifi-monitor/export?${qs.toString()}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "Export failed");
  }
  const blob = await res.blob();
  const dl = document.createElement("a");
  dl.href = URL.createObjectURL(blob);
  dl.download = buildExportFilename(kind, format);
  document.body.appendChild(dl);
  dl.click();
  dl.remove();
  URL.revokeObjectURL(dl.href);
}

function formatTs(ts) {
  if (!ts) return "—";
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString();
}

function renderStatus(status) {
  const cfg = status.config || {};
  wifiStatusSummary.textContent =
    `State: ${status.running ? "RUNNING" : "STOPPED"} | Backend: ${status.backend || "—"} | Interface: ${status.iface || "—"} | ` +
    `Last Seen: ${formatTs(status.last_seen_ts)} | RID: ${status.rid_detections || 0} | Intel Updates: ${status.wifi_updates || 0}` +
    (status.last_error ? ` | Last Error: ${status.last_error}` : "");

  wifiIfaceInput.value = cfg.iface || status.iface || "";
  wifiBackendSelect.value = cfg.backend || status.backend || "kismet";
  wifiChannelModeSelect.value = cfg.channel_mode || status.channel_mode || "hop";
  wifiPollInput.value = cfg.poll_interval_s || 2;
  wifiHopChannelsInput.value = Array.isArray(cfg.hop_channels) ? cfg.hop_channels.join(",") : "";
  wifiKismetUrlInput.value = cfg.kismet_url || "";
  wifiKismetUserInput.value = cfg.kismet_username || "";
  wifiKismetTimeoutInput.value = cfg.kismet_timeout_s || 3;
  wifiKismetCmdInput.value = "";
  if (cfg.kismet_cmd_configured) {
    wifiKismetCmdInput.placeholder = "Configured (hidden for safety)";
  } else {
    wifiKismetCmdInput.placeholder = "Fallback only if Kismet URL is unavailable";
  }
}

function renderRid() {
  if (ridBuffer.length === 0) {
    wifiRidDetections.innerHTML = '<div class="muted">No RID detections yet...</div>';
    return;
  }
  wifiRidDetections.innerHTML = ridBuffer.map((item) => {
    const fields = item.protocol_result?.decoded_fields || item.decoded_fields || {};
    return `
      <div class="protocol-card protocol-verified">
        <div class="protocol-title">REMOTE_ID · ${(item.protocol_result?.status || item.status || "update").toUpperCase()}</div>
        <div class="protocol-line">UAS ID: ${fields.uas_id || "—"}</div>
        <div class="protocol-line">Operator: ${fields.operator_id || "—"}</div>
        <div class="protocol-meta">
          <span>Source: ${item.source || "RID_WIFI"}</span>
          <span>Seen: ${formatTs(item.ts)}</span>
        </div>
      </div>
    `;
  }).join("");
}

function renderList(el, rows, emptyText) {
  if (!Array.isArray(rows) || rows.length === 0) {
    el.innerHTML = `<div class="muted">${emptyText}</div>`;
    return;
  }
  el.innerHTML = rows.map((row) => `<div class="wifi-list-row">${row}</div>`).join("");
}

function renderTrafficStrip() {
  if (!wifiTrafficStrip) return;
  if (trafficHistory.length === 0) {
    wifiTrafficStrip.innerHTML = '<div class="muted">Waiting for live traffic...</div>';
    return;
  }
  const maxPackets = Math.max(1, ...trafficHistory.map((s) => s.packets));
  const maxData = Math.max(1, ...trafficHistory.map((s) => s.data));
  wifiTrafficStrip.innerHTML = trafficHistory.map((s) => {
    const p = Math.max(6, Math.round((s.packets / maxPackets) * 100));
    const d = Math.max(4, Math.round((s.data / maxData) * 92));
    const tip = `t=${formatTs(s.ts)} | packets=${s.packets} | data=${s.data}`;
    return `
      <div class="wifi-pulse-col" title="${tip}">
        <div class="wifi-pulse-bar wifi-pulse-packets" style="height:${p}%"></div>
        <div class="wifi-pulse-bar wifi-pulse-data" style="height:${d}%"></div>
      </div>
    `;
  }).join("");
}

function pushTrafficSample(payload) {
  const frameMix = payload?.frame_mix || {};
  const channels = Array.isArray(payload?.channels) ? payload.channels : [];
  const packetsFromChannels = channels.reduce((sum, c) => sum + Number(c?.packets || c?.packet_rate || 0), 0);
  const packets = Math.max(0, Number(packetsFromChannels || frameMix.total || 0));
  const data = Math.max(0, Number(frameMix.data || 0));
  trafficHistory.push({ ts: Number(payload?.ts || Date.now() / 1000), packets, data });
  if (trafficHistory.length > TRAFFIC_HISTORY_MAX) trafficHistory.shift();
  renderTrafficStrip();
}

function startNotifyWs() {
  if (notifyWs) {
    notifyWs.close();
  }
  const url = `${window.location.protocol.replace("http", "ws")}//${window.location.host}/ws/notify`;
  notifyWs = new WebSocket(url);
  notifyWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "rid_update") {
        ridBuffer.unshift(msg.payload || {});
        if (ridBuffer.length > 20) ridBuffer.pop();
        renderRid();
      } else if (msg.type === "wifi_intel_update") {
        const p = msg.payload || {};
        pushTrafficSample(p);
        renderList(
          wifiChannels,
          (p.channels || []).slice(0, 10).map((c) => `Ch ${c.channel ?? "?"}: ${c.packets ?? c.packet_rate ?? 0} pkts`),
          "No channel data yet..."
        );
        renderList(
          wifiEmitters,
          (p.top_emitters || []).slice(0, 10).map((d) => `${d.bssid || d.mac || "unknown"} · ${d.vendor || "unknown"} · ${d.packets || 0} pkts · ch ${d.channel ?? "?"}`),
          "No emitter data yet..."
        );
        renderList(
          wifiSources,
          (p.datasources || []).slice(0, 12).map((s) =>
            `${s.name || "source"} · ${s.running ? "RUNNING" : "STOPPED"} · ${s.packets || 0} pkts · ` +
            `ch ${s.channel ?? "?"}${s.hopping ? " (hop)" : ""}`
          ),
          "No data sources yet..."
        );
        renderList(
          wifiDevices,
          (p.devices || []).slice(0, 15).map((d) =>
            `${d.mac || "unknown"} · ${d.vendor || "unknown"} · ${d.type || "unknown"} · ` +
            `ch ${d.channel ?? "?"} · sig ${d.signal ?? "?"} · ${d.packets || 0} pkts`
          ),
          "No device details yet..."
        );
        renderList(
          wifiAnomalies,
          (p.anomalies || []).slice(0, 10).map((a) => `${a.type || "anomaly"}${a.detail ? `: ${a.detail}` : ""}`),
          "No anomalies yet..."
        );
      }
    } catch (_err) {
      // ignore parse issues
    }
  };
  notifyWs.onclose = () => setTimeout(startNotifyWs, 3000);
}

async function refreshStatus() {
  try {
    const result = await apiGetStatus();
    renderStatus(result.status || {});
  } catch (e) {
    wifiStatusSummary.textContent = `Status error: ${e.message}`;
  }
}

wifiSaveConfigBtn.addEventListener("click", async () => {
  const hopChannels = wifiHopChannelsInput.value
    .split(",")
    .map((x) => Number(x.trim()))
    .filter((n) => Number.isFinite(n));
  await apiPost("/api/wifi-monitor/config", {
    iface: wifiIfaceInput.value.trim(),
    backend: wifiBackendSelect.value,
    channel_mode: wifiChannelModeSelect.value,
    poll_interval_s: Number(wifiPollInput.value || 2),
    hop_channels: hopChannels,
    kismet_url: wifiKismetUrlInput.value.trim(),
    kismet_username: wifiKismetUserInput.value.trim(),
    kismet_password: wifiKismetPassInput.value,
    kismet_timeout_s: Number(wifiKismetTimeoutInput.value || 3),
    kismet_cmd: wifiKismetCmdInput.value.trim(),
  });
  await refreshStatus();
});

wifiTestBtn.addEventListener("click", async () => {
  try {
    await apiPost("/api/wifi-monitor/config", {
      iface: wifiIfaceInput.value.trim(),
      backend: wifiBackendSelect.value,
      channel_mode: wifiChannelModeSelect.value,
      poll_interval_s: Number(wifiPollInput.value || 2),
      hop_channels: wifiHopChannelsInput.value
        .split(",")
        .map((x) => Number(x.trim()))
        .filter((n) => Number.isFinite(n)),
      kismet_url: wifiKismetUrlInput.value.trim(),
      kismet_username: wifiKismetUserInput.value.trim(),
      kismet_password: wifiKismetPassInput.value,
      kismet_timeout_s: Number(wifiKismetTimeoutInput.value || 3),
      kismet_cmd: wifiKismetCmdInput.value.trim(),
    });
    const test = await apiPost("/api/wifi-monitor/test-kismet");
    const r = test.result || {};
    if (r.ok) {
      wifiStatusSummary.textContent =
        `Kismet test OK | Status: ${r.status || "ok"} | Channels: ${r.channels_seen || 0} | Emitters: ${r.emitters_seen || 0} | RID candidates: ${r.rid_candidates || 0}`;
    } else {
      wifiStatusSummary.textContent =
        `Kismet test FAILED | Backend: ${r.backend || "kismet"} | ${r.error || r.status || "unknown error"}`;
    }
  } catch (e) {
    wifiStatusSummary.textContent = `Kismet test error: ${e.message}`;
  }
});

wifiStartBtn.addEventListener("click", async () => {
  await apiPost("/api/wifi-monitor/start");
  await refreshStatus();
});

wifiStopBtn.addEventListener("click", async () => {
  await apiPost("/api/wifi-monitor/stop");
  await refreshStatus();
});

wifiRefreshBtn.addEventListener("click", refreshStatus);

wifiExportBtn.addEventListener("click", async () => {
  const kind = wifiExportKindSelect.value || "rid";
  const format = wifiExportFormatSelect.value || "jsonl";
  const limit = Math.max(1, Math.min(1000, Number(wifiExportLimitInput.value || 500)));
  try {
    await downloadExport(kind, format, limit);
    setStatusLine(`Export downloaded: ${kind.toUpperCase()} (${format.toUpperCase()}) limit ${limit}`);
  } catch (e) {
    setStatusLine(`Export failed: ${e.message}`);
  }
});

wifiListInterfacesBtn.addEventListener("click", async () => {
  try {
    const r = await fetch("/api/wifi-monitor/interfaces", { cache: "no-store" });
    const data = await r.json();
    if (!data.ok) {
      setStatusLine(`Interface list failed: ${data.error || "unknown error"}`);
      return;
    }
    const rows = (data.interfaces || []).slice(0, 20).map((i) => {
      const n = i["kismet.datasource.probed.interface"] || i.interface || i.name || "iface";
      const t = i["kismet.datasource.probed.type"] || i.type || "unknown";
      return `${n} · ${t}`;
    });
    renderList(wifiSources, rows, "No interfaces found...");
    setStatusLine(`Loaded ${rows.length} interfaces`);
  } catch (e) {
    setStatusLine(`Interface list error: ${e.message}`);
  }
});

wifiAddSourceBtn.addEventListener("click", async () => {
  const source = (wifiAddSourceInput.value || "").trim();
  if (!source) {
    setStatusLine("Enter a source definition first.");
    return;
  }
  try {
    const res = await apiPost("/api/wifi-monitor/datasource/add", { source });
    setStatusLine(res.ok ? "Source add command sent." : `Source add failed: ${res.error || "unknown error"}`);
    await refreshStatus();
  } catch (e) {
    setStatusLine(`Source add error: ${e.message}`);
  }
});

wifiRefreshSourcesBtn.addEventListener("click", async () => {
  try {
    const data = await apiGet("/api/wifi-monitor/datasources");
    if (!data.ok) {
      setStatusLine(`Datasource refresh failed: ${data.error || "unknown error"}`);
      return;
    }
    const rows = (data.datasources || []).slice(0, 20).map((s) => {
      const name = s["kismet.datasource.name"] || s.name || s["kismet.datasource.uuid"] || "source";
      const uuid = s["kismet.datasource.uuid"] || s.uuid || "?";
      const run = Boolean(s["kismet.datasource.running"] ?? s.running) ? "RUNNING" : "STOPPED";
      const ch = s["kismet.datasource.channel"] || s.channel || "?";
      return `${name} · ${run} · ch ${ch} · ${uuid}`;
    });
    renderList(wifiSources, rows, "No data sources found...");
    setStatusLine(`Loaded ${rows.length} data sources`);
  } catch (e) {
    setStatusLine(`Datasource refresh error: ${e.message}`);
  }
});

function requireSourceUuid() {
  const uuid = (wifiSourceUuidInput.value || "").trim();
  if (!uuid) {
    setStatusLine("Enter source UUID first.");
    return null;
  }
  return uuid;
}

wifiSetChannelBtn.addEventListener("click", async () => {
  const uuid = requireSourceUuid();
  if (!uuid) return;
  const channel = (wifiSourceChannelInput.value || "").trim();
  if (!channel) {
    setStatusLine("Enter channel first.");
    return;
  }
  try {
    const res = await apiPost("/api/wifi-monitor/datasource/set-channel", { uuid, channel });
    setStatusLine(res.ok ? "Set-channel command sent." : `Set-channel failed: ${res.error || "unknown error"}`);
  } catch (e) {
    setStatusLine(`Set-channel error: ${e.message}`);
  }
});

wifiSetHopBtn.addEventListener("click", async () => {
  const uuid = requireSourceUuid();
  if (!uuid) return;
  const channels = (wifiSourceHopInput.value || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
  if (channels.length === 0) {
    setStatusLine("Enter hop channels first.");
    return;
  }
  const rate = Number(wifiSourceHopRateInput.value || 5);
  try {
    const res = await apiPost("/api/wifi-monitor/datasource/set-hop", { uuid, channels, rate });
    setStatusLine(res.ok ? "Set-hop command sent." : `Set-hop failed: ${res.error || "unknown error"}`);
  } catch (e) {
    setStatusLine(`Set-hop error: ${e.message}`);
  }
});

wifiOpenSourceBtn.addEventListener("click", async () => {
  const uuid = requireSourceUuid();
  if (!uuid) return;
  try {
    const res = await apiPost("/api/wifi-monitor/datasource/open", { uuid });
    setStatusLine(res.ok ? "Open-source command sent." : `Open-source failed: ${res.error || "unknown error"}`);
  } catch (e) {
    setStatusLine(`Open-source error: ${e.message}`);
  }
});

wifiCloseSourceBtn.addEventListener("click", async () => {
  const uuid = requireSourceUuid();
  if (!uuid) return;
  try {
    const res = await apiPost("/api/wifi-monitor/datasource/close", { uuid });
    setStatusLine(res.ok ? "Close-source command sent." : `Close-source failed: ${res.error || "unknown error"}`);
  } catch (e) {
    setStatusLine(`Close-source error: ${e.message}`);
  }
});

wifiApplyAlertBtn.addEventListener("click", async () => {
  const mac = (wifiAlertMacInput.value || "").trim();
  if (!mac) {
    setStatusLine("Enter MAC address first.");
    return;
  }
  try {
    const res = await apiPost("/api/wifi-monitor/alerts/presence", {
      alert_type: wifiAlertTypeSelect.value,
      action: wifiAlertActionSelect.value,
      mac,
    });
    setStatusLine(res.ok ? "Presence alert command sent." : `Presence alert failed: ${res.error || "unknown error"}`);
  } catch (e) {
    setStatusLine(`Presence alert error: ${e.message}`);
  }
});

wifiListAlertsBtn.addEventListener("click", async () => {
  try {
    const data = await apiGet("/api/wifi-monitor/alerts");
    if (!data.ok) {
      setStatusLine(`List alerts failed: ${data.error || "unknown error"}`);
      return;
    }
    const rows = (data.alerts || []).slice(0, 30).map((a) => {
      const n = a["kismet.alert.name"] || a.name || "alert";
      const t = a["kismet.alert.type"] || a.type || "generic";
      return `${n} · ${t}`;
    });
    renderList(wifiAlertDefs, rows, "No alert definitions returned...");
    setStatusLine(`Loaded ${rows.length} alert definitions`);
  } catch (e) {
    setStatusLine(`List alerts error: ${e.message}`);
  }
});

async function apiGet(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(txt || `Request failed: ${path}`);
  }
  return r.json();
}

function renderManagerKismetResult(result) {
  if (!result || result.ok !== true) {
    wifiMgrKismetSummary.textContent = `Manager call failed: ${result?.error || "unknown error"}`;
    return;
  }
  const data = result.data || {};
  if (data.service === "kismet") {
    wifiMgrKismetSummary.textContent = `Kismet service: ${data.state || "unknown"} (${data.sub_state || "—"})`;
  } else if (typeof data.message === "string") {
    wifiMgrKismetSummary.textContent = `Kismet service: ${data.message}`;
  } else {
    wifiMgrKismetSummary.textContent = "Kismet service command completed.";
  }
}

async function managerKismetCall(path) {
  try {
    const res = await apiPost(path);
    renderManagerKismetResult(res);
  } catch (e) {
    wifiMgrKismetSummary.textContent = `Manager call error: ${e.message}`;
  }
}

wifiMgrKismetStatusBtn.addEventListener("click", async () => {
  try {
    const res = await apiGet("/api/wifi-monitor/manager/kismet/status");
    renderManagerKismetResult(res);
  } catch (e) {
    wifiMgrKismetSummary.textContent = `Status error: ${e.message}`;
  }
});

wifiMgrKismetStartBtn.addEventListener("click", () => managerKismetCall("/api/wifi-monitor/manager/kismet/start"));
wifiMgrKismetStopBtn.addEventListener("click", () => managerKismetCall("/api/wifi-monitor/manager/kismet/stop"));

async function init() {
  await refreshStatus();
  startNotifyWs();
  setInterval(refreshStatus, 5000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
