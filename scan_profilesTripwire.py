from __future__ import annotations

from typing import Dict, Any, List, Tuple

# Profile schema (all required):
#   name: str
#   bands: List[(start_hz, stop_hz)]
#   sample_rate: float
#   fft_size: int
#   chunk_bw: float
#   dwell_ms: float
#   threshold_db: float
# Optional:
#   bias: dict
#   persistence_hits: int
#   channels_hz: List[float]   # Priority center frequencies

PROFILES: Dict[str, Dict[str, Any]] = {

    # =================================================
    # RF SURVEY — situational awareness only
    # (not for alerts or correlation)
    # =================================================
    "survey_wide": {
        "name": "RF Survey (Wide)",
        "bands": [
            (100e6, 200e6),      # VHF
            (902e6, 928e6),       # ISM 915 MHz
            (2400e6, 2484e6),    # ISM 2.4 GHz
            (5725e6, 5875e6),    # ISM 5.8 GHz
        ],
        "sample_rate": 12.5e6,
        "fft_size": 4096,
        "chunk_bw": 9e6,
        "dwell_ms": 30,
        "threshold_db": 12.0,
        "bias": {},
        "persistence_hits": 1,
    },

    # =================================================
    # FHSS CONTROL — SUB-GHZ (PRIMARY EARLY WARNING)
    # ELRS / Crossfire / RC control links
    # =================================================
    "fhss_subghz": {
        "name": "FHSS Control (Sub-GHz)",
        "bands": [
            (902e6, 928e6),   # US ISM
            # (863e6, 870e6), # EU optional
            # (433e6, 435e6), # optional
        ],
        "sample_rate": 10e6,
        "fft_size": 4096,
        "chunk_bw": 5e6,
        "dwell_ms": 15,
        "threshold_db": 6.5,
        "bias": {"fhss": 1.0},
        "persistence_hits": 1,
    },
    "fhss_subghz_ibw": {
        "name": "FHSS Control IBW (Sub-GHz)",
        "bands": [(902e6, 928e6)],
        "sample_rate": 12.5e6,  # Reduced from 30e6 for Pi4 + Pluto-like stability
        "fft_size": 4096,
        "chunk_bw": 26e6,  # Full band width
        "dwell_ms": 0,  # Not used in IBW
        "threshold_db": 10.0,
        "bias": {"fhss": 1.0},
        "persistence_hits": 1,
    },
    "fhss_eu_868_ibw": {
        "name": "FHSS Control IBW (EU 868)",
        "bands": [(863e6, 870e6)],
        "sample_rate": 10e6,
        "fft_size": 4096,
        "chunk_bw": 7e6,
        "dwell_ms": 0,
        "threshold_db": 10.0,
        "bias": {"fhss": 1.0},
        "persistence_hits": 1,
    },
    "fhss_eu_433_ibw": {
        "name": "FHSS Control IBW (EU 433)",
        "bands": [(433.05e6, 434.79e6)],
        "sample_rate": 5e6,
        "fft_size": 2048,  # Smaller FFT for narrower band
        "chunk_bw": 1.74e6,
        "dwell_ms": 0,
        "threshold_db": 10.0,
        "bias": {"fhss": 1.0},
        "persistence_hits": 1,
    },

    # =================================================
    # FHSS CONTROL — 2.4 GHz (SECONDARY / NOISY)
    # Use only as corroborating evidence
    # Note: Requires lower persistence (1 hit) due to wideband noise floors
    # and channel-prioritized scanning timing constraints.
    # =================================================
    "fhss_24g": {
        "name": "FHSS Control (2.4 GHz)",
        "bands": [
            (2400e6, 2484e6),
        ],
        "sample_rate": 12.5e6,
        "fft_size": 4096,
        "chunk_bw": 9e6,
        "dwell_ms": 20,
        "threshold_db": 9.5,
        "bias": {"fhss": 0.6},
        "persistence_hits": 1,  # Lower persistence due to wideband noise and channel scanning
    },

    # =================================================
    # DIGITAL VIDEO — 5.8 GHz (PRIMARY VIDEO CONFIRM)
    # DJI / Walksnail / HDZero (OFDM-like)
    # Note: Requires lower persistence (1 hit) due to wideband noise floors
    # and channel-prioritized scanning timing constraints.
    # =================================================
    "digital_vtx_5g8": {
        "name": "Digital VTX (5.8 GHz)",
        "bands": [
            (5645e6, 5945e6),
        ],
        "sample_rate": 12.5e6,
        "fft_size": 4096,
        "chunk_bw": 9e6,
        "dwell_ms": 40,
        "threshold_db": 10.0,
        "bias": {"ofdm": 1.0},
        "persistence_hits": 1,  # Lower persistence due to wideband noise and channel scanning
        "channels_hz": [
            5765e6,
            5770e6,
            5785e6,
            5825e6,
            5835e6,
        ],
    },

    # =================================================
    # ANALOG FPV VIDEO — 5.8 GHz
    # =================================================
    "analog_vtx_5g8": {
        "name": "Analog FPV VTX (5.8 GHz)",
        "bands": [
            (5645e6, 5945e6),
        ],
        "sample_rate": 12.5e6,
        "fft_size": 4096,
        "chunk_bw": 9e6,
        "dwell_ms": 15,
        "threshold_db": 8.5,
        "bias": {"wideband": 0.8},
        "persistence_hits": 1,
        "channels_hz": [
            5658e6, 5695e6, 5732e6, 5769e6,
            5806e6, 5843e6, 5880e6, 5917e6
        ],
    },

    # =================================================
    # VOICE / PTT — narrowband radios
    # =================================================
    "voice_narrowband": {
        "name": "Voice / PTT (Narrowband)",
        "bands": [
            (136e6, 174e6),
            (400e6, 470e6),
            (806e6, 870e6),
        ],
        "sample_rate": 5e6,
        "fft_size": 4096,
        "chunk_bw": 3e6,
        "dwell_ms": 80,
        "threshold_db": 7.0,
        "bias": {"narrowband": 0.7},
        "persistence_hits": 1,
    },

    # =================================================
    # Wi-Fi / Bluetooth — awareness only
    # =================================================
    "wifi_bt_24g": {
        "name": "Wi-Fi / Bluetooth (2.4 GHz)",
        "bands": [
            (2400e6, 2484e6),
        ],
        "sample_rate": 12.5e6,
        "fft_size": 4096,
        "chunk_bw": 9e6,
        "dwell_ms": 20,
        "threshold_db": 11.0,
        "bias": {"ofdm": 0.7},
        "persistence_hits": 1,
    },
}

DEFAULT_PROFILE_ID = "fhss_subghz"


def _normalize_profile(p: Dict[str, Any]) -> Dict[str, Any]:
    # Required keys + safe defaults
    name = str(p.get("name", "Profile"))
    bands_in = p.get("bands", [])
    bands: List[Tuple[float, float]] = []
    for b in bands_in:
        try:
            s = float(b[0])
            e = float(b[1])
            if e > s:
                bands.append((s, e))
        except Exception:
            continue

    if not bands:
        bands = [(902e6, 928e6)]

    # Normalize channels_hz safely
    channels = []
    for f in p.get("channels_hz", []):
        try:
            f = float(f)
            if f > 0:
                channels.append(f)
        except Exception:
            pass

    out = {
        "id": p.get("id"),
        "name": name,
        "bands": bands,
        "sample_rate": float(p.get("sample_rate", 20e6)),
        "fft_size": int(p.get("fft_size", 4096)),
        "chunk_bw": float(p.get("chunk_bw", float(p.get("sample_rate", 20e6)) * 0.75)),
        "dwell_ms": float(p.get("dwell_ms", 40)),
        "threshold_db": float(p.get("threshold_db", 10.0)),
        "bias": dict(p.get("bias", {})),
        "persistence_hits": int(p.get("persistence_hits", 1)),
        "channels_hz": channels,
    }

    # clamp some values
    if out["fft_size"] < 512:
        out["fft_size"] = 512
    if out["chunk_bw"] <= 0:
        out["chunk_bw"] = out["sample_rate"] * 0.75
    if out["sample_rate"] <= 0:
        out["sample_rate"] = 12.5e6

    return out


def build_manual_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    start_hz = float(cfg.get("start_mhz", 902.0)) * 1e6
    stop_hz = float(cfg.get("stop_mhz", 928.0)) * 1e6
    if stop_hz < start_hz:
        start_hz, stop_hz = stop_hz, start_hz

    p = {
        "name": "Manual",
        "bands": [(start_hz, stop_hz)],
        "sample_rate": float(cfg.get("sample_rate", 20e6)),
        "fft_size": int(cfg.get("fft_size", 4096)),
        "chunk_bw": float(cfg.get("chunk_bw", float(cfg.get("sample_rate", 20e6)) * 0.75)),
        "dwell_ms": float(cfg.get("dwell_ms", 40)),
        "threshold_db": float(cfg.get("threshold_db", 10.0)),
        "bias": {},
        "persistence_hits": 1,
    }
    return _normalize_profile(p)


def list_profiles(cfg: Dict[str, Any] | None = None) -> List[Dict[str, str]]:
    # Base profiles
    items = [{"id": k, "name": v["name"]} for k, v in PROFILES.items()]

    # Optional custom profiles stored in config.json under "custom_profiles"
    if cfg and isinstance(cfg.get("custom_profiles"), dict):
        for k, v in cfg["custom_profiles"].items():
            try:
                items.append({"id": str(k), "name": str(v.get("name", k))})
            except Exception:
                continue

    # Always include manual (virtual) option
    items.append({"id": "manual", "name": "Manual (Start/Stop MHz)"})
    return items


def resolve_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    use_plan = bool(cfg.get("use_plan", True))
    plan_id = str(cfg.get("band_plan", DEFAULT_PROFILE_ID))

    # manual override
    if not use_plan or plan_id == "manual":
        return build_manual_profile(cfg)

    # custom profiles (saved)
    custom = cfg.get("custom_profiles")
    if isinstance(custom, dict) and plan_id in custom:
        return _normalize_profile(custom[plan_id])

    # builtin profiles
    p = dict(PROFILES.get(plan_id) or PROFILES.get(DEFAULT_PROFILE_ID))
    p["id"] = plan_id
    return _normalize_profile(p if p else build_manual_profile(cfg))


def save_custom_profile(cfg: Dict[str, Any], profile_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper for C-stage: store a profile into cfg["custom_profiles"].
    Call save_config() in API after using this.
    """
    profile_id = str(profile_id).strip()
    if not profile_id:
        raise ValueError("profile_id is required")

    norm = _normalize_profile(profile)
    if "custom_profiles" not in cfg or not isinstance(cfg["custom_profiles"], dict):
        cfg["custom_profiles"] = {}

    cfg["custom_profiles"][profile_id] = norm
    return cfg
