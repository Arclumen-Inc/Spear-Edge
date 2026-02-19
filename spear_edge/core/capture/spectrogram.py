from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from PIL import Image, ImageDraw, ImageFont


def compute_spectrogram(
    iq: np.ndarray,
    sample_rate_sps: int,
    fft_size: int = 1024,
    hop_size: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute a normalized power spectrogram in dB from complex IQ.

    Returns:
        spec_db: shape (time_bins, freq_bins)
        - Frequency axis is FFT-shifted (DC in center)
        - Units are relative dB (normalized by window energy)
    """
    if hop_size is None:
        hop_size = fft_size // 4

    x = np.asarray(iq, dtype=np.complex64)
    if x.size < fft_size:
        raise ValueError("IQ too short for FFT")

    # Window
    if window == "hann":
        win = np.hanning(fft_size).astype(np.float32)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # Normalize by window energy so levels are comparable across FFT sizes/windows
    win_energy = float(np.sum(win * win)) + 1e-12

    frames = []
    for start in range(0, x.size - fft_size, hop_size):
        seg = x[start:start + fft_size] * win

        # FFT -> shift -> POWER
        X = np.fft.fftshift(np.fft.fft(seg))
        P = (np.abs(X) ** 2) / win_energy

        frames.append(P.astype(np.float32, copy=False))

    spec_power = np.stack(frames, axis=0)

    # Power to dB
    spec_db = 10.0 * np.log10(spec_power + 1e-12)

    return spec_db


def compute_spectrogram_chunked(
    iq_path: Path,
    sample_rate_sps: int,
    fft_size: int = 1024,
    hop_size: int | None = None,
    window: str = "hann",
    chunk_size_samples: int = 5_000_000,
) -> Tuple[np.ndarray, dict]:
    """
    Memory-efficient spectrogram computation that processes IQ file in chunks.
    
    This function:
    1. Reads IQ file in chunks (default 5M samples = ~40 MB)
    2. Computes spectrogram incrementally, downsampling on-the-fly
    3. Returns downsampled spectrogram (≤512x512) for ML + thumbnail
    4. Also returns basic stats
    
    Args:
        iq_path: Path to IQ file (complex64 binary)
        sample_rate_sps: Sample rate in samples/second
        fft_size: FFT size (default 1024)
        hop_size: Hop size (default fft_size // 4)
        window: Window type (default "hann")
        chunk_size_samples: Number of samples to process per chunk (default 5M)
    
    Returns:
        (spec_ml, stats): ML-ready spectrogram (≤512x512) and basic stats dict
    """
    if hop_size is None:
        hop_size = fft_size // 4
    
    # Window
    if window == "hann":
        win = np.hanning(fft_size).astype(np.float32)
    else:
        raise ValueError(f"Unsupported window: {window}")
    
    win_energy = float(np.sum(win * win))
    
    # Get file size to estimate total samples
    file_size = iq_path.stat().st_size
    total_samples = file_size // 8  # complex64 = 8 bytes per sample
    
    if total_samples < fft_size:
        raise ValueError(f"IQ file too short: {total_samples} samples < {fft_size}")
    
    # Estimate number of time bins
    total_time_bins = 1 + (total_samples - fft_size) // hop_size
    
    # Target ML shape
    T_TARGET, F_TARGET = ML_SPECTROGRAM_TARGET_SHAPE
    
    # Strategy: Process in chunks, accumulate downsampled frames directly
    # We'll downsample frequency axis immediately, then accumulate time axis
    
    # Accumulator for downsampled spectrogram (time, freq)
    # We accumulate power (linear) then convert to dB at end
    accumulated_frames = []  # List of (time_bins, F_TARGET) arrays
    
    # For stats: sample some frames (not all to save memory)
    stats_frames = []  # Sample of frames for stats
    
    # Process file in chunks
    with open(iq_path, 'rb') as f:
        overlap_buffer = np.empty(0, dtype=np.complex64)
        bytes_read = 0
        
        while bytes_read < file_size:
            # Read chunk
            chunk_bytes = min(chunk_size_samples * 8, file_size - bytes_read)
            chunk_data = f.read(chunk_bytes)
            
            if len(chunk_data) == 0:
                break
            
            chunk_iq = np.frombuffer(chunk_data, dtype=np.complex64)
            
            # Prepend overlap buffer
            if overlap_buffer.size > 0:
                chunk_iq = np.concatenate([overlap_buffer, chunk_iq])
            
            # Process frames in this chunk
            chunk_frames = []
            pos = 0
            
            while pos + fft_size <= chunk_iq.size:
                seg = chunk_iq[pos:pos + fft_size]
                seg = seg * win
                
                # FFT
                fft = np.fft.fftshift(np.fft.fft(seg))
                mag = np.abs(fft)
                power = (mag ** 2) / max(win_energy, 1e-12)
                power_db = 10.0 * np.log10(power + 1e-12)
                
                chunk_frames.append(power_db)
                pos += hop_size
            
            # Save overlap (last fft_size samples needed for next chunk)
            if chunk_iq.size >= fft_size:
                overlap_buffer = chunk_iq[-fft_size:].copy()
            else:
                overlap_buffer = chunk_iq.copy()
            
            if not chunk_frames:
                bytes_read += chunk_bytes
                continue
            
            # Stack frames from this chunk: (time_bins, freq_bins)
            chunk_spec_db = np.stack(chunk_frames, axis=0)
            
            # Downsample frequency axis immediately (freq_bins -> F_TARGET)
            f_bins = chunk_spec_db.shape[1]
            if f_bins > F_TARGET:
                factor = f_bins // F_TARGET
                chunk_spec_ds = chunk_spec_db[:, :factor * F_TARGET]
                chunk_spec_ds = chunk_spec_ds.reshape(chunk_spec_ds.shape[0], F_TARGET, factor).mean(axis=2)
            else:
                chunk_spec_ds = chunk_spec_db
            
            # Accumulate this chunk (will downsample time axis later)
            accumulated_frames.append(chunk_spec_ds)
            
            # Sample frames for stats (limit to ~1000 frames total)
            if len(stats_frames) < 1000:
                step = max(1, chunk_spec_db.shape[0] // 50)
                stats_frames.append(chunk_spec_db[::step])
            
            bytes_read += chunk_bytes
            
            # Periodic GC
            if len(accumulated_frames) % 10 == 0:
                import gc
                gc.collect()
        
        # Concatenate all accumulated frames
        if not accumulated_frames:
            raise ValueError("No frames computed from IQ file")
        
        # Stack: (total_time_bins, F_TARGET)
        full_spec_db = np.concatenate(accumulated_frames, axis=0)
        
        # Downsample time axis to T_TARGET
        t_bins, f_bins = full_spec_db.shape
        if t_bins > T_TARGET:
            factor = t_bins // T_TARGET
            spec_ml = full_spec_db[:factor * T_TARGET]
            spec_ml = spec_ml.reshape(T_TARGET, factor, F_TARGET).mean(axis=1)
        else:
            spec_ml = full_spec_db
        
        # Normalize to noise floor
        noise_floor = np.median(spec_ml)
        spec_ml = spec_ml - noise_floor
        spec_ml = spec_ml.astype(ML_SPECTROGRAM_DTYPE)
        
        # Compute stats from sampled frames
        if stats_frames:
            stats_spec = np.concatenate(stats_frames, axis=0)
            stats = extract_basic_stats(stats_spec)
        else:
            stats = extract_basic_stats(spec_ml)
        
        return spec_ml, stats


def save_spectrogram_thumbnail(
    spec_db: np.ndarray,
    out_path: Path,
    vmin: float = -90.0,
    vmax: float = 0.0,
    center_freq_hz: Optional[int] = None,
    sample_rate_sps: Optional[int] = None,
    duration_s: Optional[float] = None,
    requested_duration_s: Optional[float] = None,
    fft_size: Optional[int] = None,
    hop_size: Optional[int] = None,
) -> None:
    """
    Save a contrast-normalized PNG spectrogram for operators.
    Inverted colormap: strong signal → dark, noise → light (matches GNU Radio convention).
    Adds axes, labels, and capture context annotations.
    """
    time_bins, freq_bins = spec_db.shape
    
    # Auto-scale if defaults are used (prevents "all black" thumbnails)
    # Use robust percentiles so strong stations + weak signals both look good.
    if vmin == -90.0 and vmax == 0.0:
        lo = float(np.percentile(spec_db, 5))
        hi = float(np.percentile(spec_db, 99))
        # Prevent degenerate scaling
        if hi - lo < 10.0:
            hi = lo + 10.0
        vmin, vmax = lo, hi
    
    # Invert colormap: strong signal → dark, noise → light (matches GNU Radio convention)
    img = 1.0 - np.clip((spec_db - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    # Flip frequency axis so low freq is at bottom
    img = np.flipud(img.T)

    # Create image with margins for axes
    margin_left = 40  # For frequency labels
    margin_bottom = 20  # For time labels
    margin_top = 30  # For capture context
    margin_right = 10
    
    img_h, img_w = img.shape
    total_h = img_h + margin_top + margin_bottom
    total_w = img_w + margin_left + margin_right
    
    # Create RGB image (L mode doesn't support colored text easily)
    im = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
    
    # Paste spectrogram into image (offset by margins)
    spec_img = Image.fromarray(img, mode="L")
    # Convert to RGB for compositing
    spec_rgb = Image.new("RGB", spec_img.size)
    spec_rgb.paste(spec_img)
    im.paste(spec_rgb, (margin_left, margin_top))
    
    # Draw axes and labels
    draw = ImageDraw.Draw(im)
    
    # Try to use a monospace font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    text_color = (255, 255, 255)  # White text
    
    # Compute frequency axis correctly: center_freq ± sample_rate/2
    # This is the correct formula for FFT-shifted spectrograms
    # DO NOT use baseband frequencies (-fs/2 to +fs/2) without adding center_freq
    if center_freq_hz is not None and sample_rate_sps is not None:
        freq_start_hz = center_freq_hz - sample_rate_sps / 2
        freq_end_hz = center_freq_hz + sample_rate_sps / 2
        # Use endpoint=False to match FFT bin centers
        freq_axis = np.linspace(freq_start_hz, freq_end_hz, num=freq_bins, endpoint=False)
    else:
        freq_axis = None
    
    if duration_s is not None:
        time_axis = np.linspace(0, duration_s, num=time_bins)
    else:
        time_axis = None
    
    # Draw frequency axis (Y-axis, left side)
    if freq_axis is not None:
        num_freq_ticks = 5
        for i in range(num_freq_ticks):
            t = i / (num_freq_ticks - 1) if num_freq_ticks > 1 else 0
            freq_idx = int(t * (freq_bins - 1))
            freq_hz = freq_axis[freq_idx]
            freq_mhz = freq_hz / 1e6
            
            # Y position in image (accounting for flip and margins)
            y_img = margin_top + int((1 - t) * img_h)
            
            # Draw tick mark
            draw.line([(margin_left - 5, y_img), (margin_left, y_img)], fill=text_color, width=1)
            
            # Draw label (use 1 decimal place for readability, matches user example)
            label = f"{freq_mhz:.1f} MHz"
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w = len(label) * 6
                text_h = 10
            draw.text((margin_left - text_w - 5, y_img - text_h // 2), label, fill=text_color, font=font)
    
    # Draw time axis (X-axis, bottom)
    if time_axis is not None:
        num_time_ticks = 5
        for i in range(num_time_ticks):
            t = i / (num_time_ticks - 1) if num_time_ticks > 1 else 0
            time_idx = int(t * (time_bins - 1))
            time_s = time_axis[time_idx]
            
            # X position in image
            x_img = margin_left + int(t * img_w)
            
            # Draw tick mark
            draw.line([(x_img, margin_top + img_h), (x_img, margin_top + img_h + 5)], fill=text_color, width=1)
            
            # Draw label
            label = f"{time_s:.1f} s"
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                text_w = len(label) * 6
            draw.text((x_img - text_w // 2, margin_top + img_h + 7), label, fill=text_color, font=font)
    
    # Add capture context overlay (top-left corner)
    if center_freq_hz is not None and sample_rate_sps is not None and duration_s is not None:
        # Show actual duration, and indicate if it's partial (shortened)
        if requested_duration_s is not None and duration_s < requested_duration_s * 0.95:
            # Capture ended early (likely overflow/throughput issue)
            dur_label = f"Dur: {duration_s:.2f} s (partial, requested {requested_duration_s:.1f} s)"
        else:
            dur_label = f"Dur: {duration_s:.2f} s"
        
        context_lines = [
            f"CF: {center_freq_hz / 1e6:.3f} MHz",
            f"SR: {sample_rate_sps / 1e6:.1f} MS/s",
            dur_label,
        ]
        
        # Draw semi-transparent background for text
        if font:
            max_w = max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in context_lines)
            text_h = draw.textbbox((0, 0), "A", font=font)[3] - draw.textbbox((0, 0), "A", font=font)[1]
        else:
            max_w = max(len(line) * 6 for line in context_lines)
            text_h = 10
        
        bg_h = len(context_lines) * (text_h + 2) + 4
        bg_rect = Image.new("RGBA", (max_w + 8, bg_h), color=(0, 0, 0, 180))  # Semi-transparent black
        im.paste(bg_rect, (margin_left + 5, 5), bg_rect)
        
        # Draw text
        y_offset = 7
        for line in context_lines:
            draw.text((margin_left + 9, y_offset), line, fill=text_color, font=font)
            y_offset += text_h + 2
    
    im.save(out_path)


# ML Feature Contract (LOCKED)
# Tensor shape: (512, 512)
# Axes: (time_bins, freq_bins)
# dtype: float32
# Units: relative dB (noise-referenced)
# Max size: ~1 MB per capture
ML_SPECTROGRAM_TARGET_SHAPE = (512, 512)
ML_SPECTROGRAM_DTYPE = np.float32


def downsample_for_ml(spec_db: np.ndarray) -> np.ndarray:
    """
    Downsample spectrogram to ML contract: (≤512, ≤512) float32.
    
    This function:
    1. Downsamples time axis to ≤512 bins (averaging)
    2. Downsamples frequency axis to ≤512 bins (averaging)
    3. Normalizes to noise floor (relative dB)
    4. Converts to float32
    
    Args:
        spec_db: Full-resolution spectrogram in dB, shape (time_bins, freq_bins)
        
    Returns:
        ML-ready spectrogram, shape (≤512, ≤512), dtype float32
    """
    T_TARGET, F_TARGET = ML_SPECTROGRAM_TARGET_SHAPE
    t_bins, f_bins = spec_db.shape
    
    spec = spec_db.copy()
    
    # Downsample TIME to ≤512 bins
    if t_bins > T_TARGET:
        factor = t_bins // T_TARGET
        # Truncate to exact multiple, then reshape and average
        spec = spec[:factor * T_TARGET]
        spec = spec.reshape(T_TARGET, factor, spec.shape[1]).mean(axis=1)
    elif t_bins < T_TARGET:
        # Pad with zeros if needed (shouldn't happen in practice)
        pad = T_TARGET - t_bins
        spec = np.pad(spec, ((0, pad), (0, 0)), mode='constant', constant_values=np.min(spec))
    
    # Downsample FREQUENCY to ≤512 bins
    if f_bins > F_TARGET:
        factor = f_bins // F_TARGET
        # Truncate to exact multiple, then reshape and average
        spec = spec[:, :factor * F_TARGET]
        spec = spec.reshape(spec.shape[0], F_TARGET, factor).mean(axis=2)
    elif f_bins < F_TARGET:
        # Pad with zeros if needed (shouldn't happen in practice)
        pad = F_TARGET - f_bins
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant', constant_values=np.min(spec))
    
    # Normalize to noise floor (relative dB)
    noise_floor = np.median(spec)
    spec = spec - noise_floor
    
    # Convert to float32 and enforce contract
    spec = spec.astype(ML_SPECTROGRAM_DTYPE)
    
    # Memory discipline: enforce contract
    assert spec.shape[0] <= T_TARGET, f"Time bins {spec.shape[0]} exceeds {T_TARGET}"
    assert spec.shape[1] <= F_TARGET, f"Freq bins {spec.shape[1]} exceeds {F_TARGET}"
    assert spec.dtype == ML_SPECTROGRAM_DTYPE, f"dtype {spec.dtype} != {ML_SPECTROGRAM_DTYPE}"
    
    return spec


def extract_basic_stats(
    spec_db: np.ndarray,
    activity_margin_db: float = 6.0,
) -> dict:
    """
    Compute basic stats from spectrogram.

    duty_cycle is defined as fraction of TIME bins where any frequency bin
    exceeds noise_floor + activity_margin_db.
    """
    noise_floor_db = float(np.median(spec_db))
    peak_db = float(np.max(spec_db))
    snr_db = float(peak_db - noise_floor_db)

    # Time occupancy: is there signal in this time slice?
    threshold = noise_floor_db + activity_margin_db
    time_active = np.any(spec_db > threshold, axis=1)
    duty_cycle = float(np.mean(time_active))

    return {
        "noise_floor_db": noise_floor_db,
        "peak_db": peak_db,
        "snr_db": snr_db,
        "duty_cycle": duty_cycle,
    }
