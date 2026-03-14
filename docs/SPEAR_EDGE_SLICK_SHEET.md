# SPEAR-Edge  
## Tactical RF Monitoring & Analysis System

**One-line**: SDR-based RF monitoring, automated capture, and ML classification on Jetson Orin Nano with Tripwire and ATAK integration.

---

### Key Capabilities

| Area | Capability |
|------|------------|
| **Spectrum** | Real-time FFT/waterfall, 15–30 FPS, configurable FFT (e.g. 2048), noise floor and smoothing |
| **Capture** | Manual and armed (Tripwire-triggered); IQ streamed to disk; spectrogram + metadata; 5 s default |
| **ML** | PyTorch/ONNX GPU classification; 23+ classes; quick fine-tune from 1–2 captures; export/import models |
| **Tripwire** | WebSocket + HTTP; event ingest; AoA fusion (cones + triangulation); remote scan-plan setting |
| **ATAK** | CoT chat, markers, TAI polygons; status and capture notifications |
| **UI** | Web dashboard (FFT, SDR, capture, Tripwire, network config); separate ML dashboard for labeling/training |

---

### Specifications

| Item | Detail |
|------|--------|
| **Platform** | NVIDIA Jetson Orin Nano (6-core ARM, Ampere GPU, 8 GB LPDDR5) |
| **SDR** | bladeRF 2.0 micro (xA4/xA9); 47 MHz–6 GHz; up to ~30–40 MS/s on Jetson; 1–2 RX; 0–60 dB gain |
| **IQ** | SC16_Q11 (device); CF32_LE + SigMF (storage); power-of-two read sizes (e.g. 8192) |
| **Modes** | Manual, Armed (auto-capture from Tripwire), Tasked (during capture) |
| **Capture** | Configurable 1–60 s; spectrogram ≤512×512; GPU classification; artifacts: IQ, PNG, JSON |
| **Interfaces** | REST + WebSockets (live FFT, notify, Tripwire link); ingest on port 8000, main app on 8080 |

---

### Applications

- **Standalone RF monitoring** — Manual spectrum analysis and signal capture  
- **Distributed sensing** — Coordination with Tripwire nodes; AoA fusion and triangulation  
- **Tactical ops** — ATAK integration for CoT, markers, and TAI  
- **ML data collection** — Armed/manual capture with labeling and quick fine-tuning  

---

### Differentiators

- **Edge-optimized**: Single-board (Jetson Orin Nano); memory-efficient IQ pipeline; no runtime crashes by design  
- **Tripwire-native**: Event ingest, cue handling, AoA fusion API, remote scan-plan over WebSocket  
- **ML-ready**: On-device training (quick fine-tune), export/import, batch labeling, class labels and stats  
- **Standards-aware**: SigMF metadata; CoT for ATAK; REST + WebSocket APIs documented  

---

### Platform Requirements

- **OS**: Jetson Linux (Ubuntu-based)  
- **Storage**: NVMe SSD recommended for IQ capture  
- **USB**: USB 3.0 for sample rates > ~20 MS/s  
- **Optional**: BT200 LNA, GPS (e.g. gpsd) for time sync  

---

*Generate PDF: `pandoc SPEAR_EDGE_SLICK_SHEET.md -o SPEAR_EDGE_SLICK_SHEET.pdf` or open in a viewer and print to PDF. For a branded slick sheet, paste sections into your organization’s one-page template (e.g. Word/InDesign).*
