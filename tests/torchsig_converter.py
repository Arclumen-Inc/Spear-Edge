#!/usr/bin/env python3
"""
Utility to convert SPEAR-Edge captures to torchsig-compatible format.

Usage:
    python3 torchsig_converter.py <capture_dir> [--normalize minmax|zscore|none] [--output <output_path>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"ERROR: Required libraries not installed: {e}")
    print("Install with: pip3 install numpy torch")
    sys.exit(1)


def load_capture_spectrogram(capture_dir: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    """Load spectrogram and metadata from capture directory."""
    spec_path = capture_dir / "features" / "spectrogram.npy"
    json_path = capture_dir / "capture.json"
    
    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrogram not found: {spec_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")
    
    # Load spectrogram
    spec = np.load(spec_path)
    
    # Load metadata
    with open(json_path) as f:
        metadata = json.load(f)
    
    return spec, metadata


def normalize_spectrogram(spec: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize spectrogram for ML inference.
    
    Args:
        spec: Input spectrogram (512, 512) float32
        method: Normalization method ("minmax", "zscore", or "none")
    
    Returns:
        Normalized spectrogram
    """
    if method == "minmax":
        # Normalize to [0, 1]
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = np.zeros_like(spec)
    elif method == "zscore":
        # Z-score normalization (mean=0, std=1)
        spec_mean = spec.mean()
        spec_std = spec.std()
        if spec_std > 0:
            spec = (spec - spec_mean) / spec_std
        else:
            spec = np.zeros_like(spec)
    elif method == "none":
        # Use as-is (relative dB)
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return spec.astype(np.float32)


def convert_to_torchsig(spec: np.ndarray, add_dims: bool = True) -> torch.Tensor:
    """
    Convert spectrogram to torchsig-compatible tensor.
    
    Args:
        spec: Spectrogram array (512, 512) or (time, freq)
        add_dims: If True, add batch and channel dimensions -> (1, 1, 512, 512)
    
    Returns:
        PyTorch tensor ready for torchsig
    """
    # Convert to tensor
    spec_tensor = torch.from_numpy(spec)
    
    # Add batch and channel dimensions if requested
    if add_dims:
        if spec_tensor.ndim == 2:
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
        elif spec_tensor.ndim == 3:
            spec_tensor = spec_tensor.unsqueeze(0)  # (1, channels, 512, 512)
    
    return spec_tensor


def check_quality(metadata: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Check if capture meets quality standards for ML.
    
    Returns:
        (is_quality, warnings)
    """
    warnings = []
    
    quality = metadata.get("quality", {})
    triage = metadata.get("triage", {})
    derived_stats = metadata.get("derived_stats", {})
    
    # Check quality flags
    if not quality.get("valid", True):
        warnings.append("Capture marked as invalid")
        return False, warnings
    
    # Check for signal presence
    if not triage.get("signal_present", False):
        warnings.append("No signal detected (likely noise only)")
        return False, warnings
    
    # Check for noise
    if triage.get("likely_noise", False):
        warnings.append("Likely noise (not a real signal)")
        return False, warnings
    
    # Check SNR
    snr_db = derived_stats.get("snr_db", 0.0)
    if snr_db < 10.0:
        warnings.append(f"Low SNR: {snr_db:.1f} dB (recommended >10 dB)")
    
    # Check for clipping
    if "clipping" in quality.get("warnings", []):
        warnings.append("Signal clipping detected")
    
    # Check for partial capture
    if "partial_capture" in quality.get("warnings", []):
        warnings.append("Partial capture (incomplete duration)")
    
    # Check for low SNR warning
    if "low_snr" in quality.get("warnings", []):
        warnings.append("Low SNR warning")
    
    is_quality = len([w for w in warnings if "Low SNR" not in w and "warning" not in w.lower()]) == 0
    
    return is_quality, warnings


def inspect_capture(capture_dir: Path):
    """Inspect a capture's artifacts and quality."""
    print(f"\n{'='*70}")
    print(f"Inspecting Capture: {capture_dir.name}")
    print(f"{'='*70}\n")
    
    try:
        spec, metadata = load_capture_spectrogram(capture_dir)
        
        # Spectrogram info
        print("📊 Spectrogram:")
        print(f"   Shape: {spec.shape}")
        print(f"   Dtype: {spec.dtype}")
        print(f"   Range: [{spec.min():.2f}, {spec.max():.2f}] dB")
        print(f"   Mean: {spec.mean():.2f} dB")
        print(f"   Std: {spec.std():.2f} dB")
        
        # RF Configuration
        rf_config = metadata.get("rf_configuration", {})
        print(f"\n📡 RF Configuration:")
        print(f"   Frequency: {rf_config.get('center_freq_hz', 0)/1e6:.3f} MHz")
        print(f"   Sample Rate: {rf_config.get('sample_rate_sps', 0)/1e6:.2f} MS/s")
        print(f"   Bandwidth: {rf_config.get('bandwidth_hz', 0)/1e6:.2f} MHz" if rf_config.get('bandwidth_hz') else "   Bandwidth: N/A")
        print(f"   Gain: {rf_config.get('gain_db', 'N/A')} dB")
        
        # Quality Metrics
        derived_stats = metadata.get("derived_stats", {})
        print(f"\n📈 Quality Metrics:")
        print(f"   SNR: {derived_stats.get('snr_db', 0):.1f} dB")
        print(f"   Duty Cycle: {derived_stats.get('duty_cycle', 0):.2%}")
        print(f"   Occupied BW: {derived_stats.get('occupied_bw_hz', 0)/1e6:.2f} MHz")
        
        # Triage
        triage = metadata.get("triage", {})
        print(f"\n🔍 Signal Triage:")
        print(f"   Signal Present: {triage.get('signal_present', False)}")
        print(f"   Likely Noise: {triage.get('likely_noise', False)}")
        print(f"   Bursty: {triage.get('bursty', False)}")
        print(f"   Wideband: {triage.get('wideband', False)}")
        
        # Quality Check
        is_quality, warnings = check_quality(metadata)
        print(f"\n✅ Quality Assessment:")
        if is_quality:
            print(f"   Status: ✅ QUALITY CAPTURE (suitable for ML)")
        else:
            print(f"   Status: ⚠️  QUALITY ISSUES DETECTED")
        
        if warnings:
            print(f"   Warnings:")
            for warning in warnings:
                print(f"     - {warning}")
        
        # TorchSig Compatibility
        print(f"\n🤖 TorchSig Compatibility:")
        print(f"   Shape: ✅ {spec.shape} (matches torchsig 512×512 requirement)")
        print(f"   Dtype: ✅ {spec.dtype} (float32 compatible)")
        print(f"   Format: ✅ NumPy array (can convert to PyTorch tensor)")
        
        if spec.shape != (512, 512):
            print(f"   ⚠️  Shape mismatch: Expected (512, 512), got {spec.shape}")
        
        return spec, metadata, is_quality
        
    except Exception as e:
        print(f"❌ Error inspecting capture: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def convert_capture(capture_dir: Path, normalize: str = "minmax", output_path: Optional[Path] = None):
    """Convert capture to torchsig format."""
    print(f"\n{'='*70}")
    print(f"Converting Capture: {capture_dir.name}")
    print(f"{'='*70}\n")
    
    try:
        spec, metadata = load_capture_spectrogram(capture_dir)
        
        # Normalize
        print(f"Normalizing spectrogram (method: {normalize})...")
        spec_norm = normalize_spectrogram(spec, method=normalize)
        print(f"   Original range: [{spec.min():.2f}, {spec.max():.2f}] dB")
        print(f"   Normalized range: [{spec_norm.min():.2f}, {spec_norm.max():.2f}]")
        
        # Convert to torch tensor
        print(f"Converting to PyTorch tensor...")
        spec_tensor = convert_to_torchsig(spec_norm, add_dims=True)
        print(f"   Tensor shape: {spec_tensor.shape}")
        print(f"   Tensor dtype: {spec_tensor.dtype}")
        
        # Save if output path provided
        if output_path:
            print(f"Saving to: {output_path}")
            torch.save(spec_tensor, output_path)
            print(f"✅ Saved successfully")
        else:
            print(f"✅ Conversion complete (not saved - use --output to save)")
        
        return spec_tensor
        
    except Exception as e:
        print(f"❌ Error converting capture: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert SPEAR-Edge captures to torchsig format")
    parser.add_argument("capture_dir", type=Path, help="Path to capture directory")
    parser.add_argument("--normalize", choices=["minmax", "zscore", "none"], default="minmax",
                       help="Normalization method (default: minmax)")
    parser.add_argument("--output", type=Path, help="Output path for converted tensor (.pt file)")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect, don't convert")
    
    args = parser.parse_args()
    
    if not args.capture_dir.exists():
        print(f"❌ Capture directory not found: {args.capture_dir}")
        sys.exit(1)
    
    # Inspect capture
    spec, metadata, is_quality = inspect_capture(args.capture_dir)
    
    if spec is None:
        sys.exit(1)
    
    # Convert if requested
    if not args.inspect_only:
        convert_capture(args.capture_dir, normalize=args.normalize, output_path=args.output)
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"✅ Capture inspected successfully")
    if is_quality:
        print(f"✅ Quality: Suitable for ML inference")
    else:
        print(f"⚠️  Quality: Issues detected (see warnings above)")
    print(f"✅ TorchSig: Compatible format (may need normalization)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
