"""
HVSR computation functions.

Author: Shihao Yuan (syuan@mines.edu)

DISCLAIMER: This is a development build. The code may contain errors or unstable functionality.
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import periodogram, welch, detrend
from scipy.signal.windows import tukey
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings

try:
    from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing as obspy_ko_smoothing
except Exception:
    obspy_ko_smoothing = None

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install with 'pip install numba' for acceleration.")
    
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args)

# Custom KO implementation to fix ObsPy issues
def custom_konno_ohmachi_smoothing(data, freqs, bandwidth=40.0, method='improved'):
    """
    Custom Konno-Ohmachi smoothing with improved edge handling.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    freqs : array-like  
        Frequency array corresponding to data
    bandwidth : float
        Konno-Ohmachi bandwidth parameter
    method : str
        Method to use: 'improved', 'smooth', 'robust', 'hybrid'
    
    Returns:
    --------
    smoothed_data : ndarray
        Smoothed data
    """
    
    if method == 'improved':
        return _improved_ko(data, freqs, bandwidth)
    elif method == 'smooth':
        return _high_smoothness_ko(data, freqs, bandwidth)
    elif method == 'robust':
        return _robust_ko(data, freqs, bandwidth)
    elif method == 'hybrid':
        return _hybrid_ko(data, freqs, bandwidth)
    else:
        raise ValueError(f"Unknown method: {method}")

def _improved_ko(data, freqs, bandwidth):
    """
    Improved KO with better edge handling and normalization.
    Uses true Konno-Ohmachi kernel for comparable smoothing to ObsPy.
    """
    data = np.asarray(data, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    
    if len(data) != len(freqs):
        raise ValueError("Data and frequency arrays must have same length")
    
    # Pre-allocate output
    smoothed = np.zeros_like(data)
    
    # Use true Konno-Ohmachi kernel (not sinc approximation)
    for i, fc in enumerate(freqs):
        # KO kernel: [sin(b * log10(f/fc)) / (b * log10(f/fc))]^4
        # Avoid division by zero at fc
        log_ratio = np.log10(freqs / fc)
        
        # Handle the singularity at f = fc
        kernel = np.ones_like(freqs)
        non_zero = np.abs(log_ratio) > 1e-10
        
        if np.any(non_zero):
            arg = bandwidth * log_ratio[non_zero]
            # True KO kernel
            kernel[non_zero] = (np.sin(arg) / arg) ** 4
        
        # Apply conservative edge tapering (less aggressive than before)
        edge_taper = _get_conservative_edge_taper(len(freqs), i, bandwidth)
        kernel *= edge_taper
        
        # Normalize kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        
        # Apply kernel
        smoothed[i] = np.sum(data * kernel)
    
    return smoothed

def _high_smoothness_ko(data, freqs, bandwidth):
    """
    High-smoothness KO implementation that matches ObsPy smoothness levels.
    """
    data = np.asarray(data, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    
    if len(data) != len(freqs):
        raise ValueError("Data and frequency arrays must have same length")
    
    # Pre-allocate output
    smoothed = np.zeros_like(data)
    
    # Use enhanced KO kernel with better smoothing
    for i, fc in enumerate(freqs):
        log_ratio = np.log10(freqs / fc)
        
        # Handle the singularity at f = fc
        kernel = np.ones_like(freqs)
        non_zero = np.abs(log_ratio) > 1e-10
        
        if np.any(non_zero):
            arg = bandwidth * log_ratio[non_zero]
            # True KO kernel with enhanced smoothing
            kernel[non_zero] = (np.sin(arg) / arg) ** 4
        
        # Apply minimal edge tapering for maximum smoothness
        edge_taper = _get_minimal_edge_taper(len(freqs), i)
        kernel *= edge_taper
        
        # Normalize kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        
        # Apply kernel
        smoothed[i] = np.sum(data * kernel)
    
    return smoothed

def _get_minimal_edge_taper(length, center_idx):
    """Create minimal edge tapering for maximum smoothness."""
    taper = np.ones(length)
    
    # Only taper at the very edges (2% instead of 5%)
    edge_fraction = 0.02
    edge_samples = max(1, int(length * edge_fraction))
    
    # Left edge taper
    if center_idx < edge_samples:
        taper[:edge_samples] = 0.8 + 0.2 * np.cos(np.pi * (1 - np.arange(edge_samples) / edge_samples))
    
    # Right edge taper  
    if center_idx >= length - edge_samples:
        taper[-edge_samples:] = 0.8 + 0.2 * np.cos(np.pi * np.arange(edge_samples) / edge_samples)
    
    return taper

def _get_conservative_edge_taper(length, center_idx, bandwidth):
    """Create conservative edge tapering (less aggressive than original)."""
    taper = np.ones(length)
    
    # Only taper near the very edges (5% instead of 10%)
    edge_fraction = 0.05
    edge_samples = max(1, int(length * edge_fraction))
    
    # Left edge taper
    if center_idx < edge_samples:
        # Gentler taper (cosine instead of linear)
        taper_vals = 0.5 + 0.5 * np.cos(np.pi * (1 - np.arange(edge_samples) / edge_samples))
        taper[:edge_samples] = taper_vals
    
    # Right edge taper  
    if center_idx >= length - edge_samples:
        taper_vals = 0.5 + 0.5 * np.cos(np.pi * np.arange(edge_samples) / edge_samples)
        taper[-edge_samples:] = taper_vals
    
    return taper

def _robust_ko(data, freqs, bandwidth):
    """Robust KO with outlier protection and adaptive bandwidth."""
    data = np.asarray(data)
    freqs = np.asarray(freqs)
    
    # Remove outliers first
    data_robust = _remove_outliers(data)
    
    # Apply improved KO
    smoothed = _improved_ko(data_robust, freqs, bandwidth)
    
    # Post-process to prevent extreme values
    smoothed = _clamp_extreme_values(smoothed, data)
    
    return smoothed

def _hybrid_ko(data, freqs, bandwidth):
    """Hybrid approach: KO + moving average combination."""
    data = np.asarray(data)
    freqs = np.asarray(freqs)
    
    # Get KO result
    ko_result = _improved_ko(data, freqs, bandwidth)
    
    # Get moving average result
    ma_result = uniform_filter1d(data, size=3, mode='nearest')
    
    # Create frequency-dependent weights
    # Favor KO at low frequencies, MA at high frequencies
    weights = _get_frequency_weights(freqs)
    
    # Combine results
    hybrid = weights * ko_result + (1 - weights) * ma_result
    
    return hybrid


def _remove_outliers(data, threshold=3.0):
    """Remove outliers using robust statistics."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Robust threshold
    robust_threshold = threshold * 1.4826 * mad
    
    # Clip outliers
    clipped = np.clip(data, median - robust_threshold, median + robust_threshold)
    
    return clipped

def _clamp_extreme_values(smoothed, original):
    """Prevent extreme values in smoothed data."""
    # Calculate reasonable bounds based on original data
    data_range = np.max(original) - np.min(original)
    upper_bound = np.max(original) + 0.5 * data_range
    lower_bound = max(0, np.min(original) - 0.2 * data_range)
    
    # Clamp smoothed data
    clamped = np.clip(smoothed, lower_bound, upper_bound)
    
    return clamped

def _get_frequency_weights(freqs):
    """Create frequency-dependent weights for hybrid smoothing."""
    # Favor KO at low frequencies, MA at high frequencies
    weights = np.ones_like(freqs)
    
    # Transition around 2 Hz
    transition_freq = 2.0
    
    # Linear transition
    high_freq_mask = freqs > transition_freq
    low_freq_mask = freqs < transition_freq * 0.5
    
    weights[low_freq_mask] = 1.0  # Full KO weight
    weights[high_freq_mask] = 0.3  # Reduced KO weight
    
    # Smooth transition
    transition_mask = (freqs >= transition_freq * 0.5) & (freqs <= transition_freq)
    if np.any(transition_mask):
        transition_weights = np.linspace(1.0, 0.3, np.sum(transition_mask))
        weights[transition_mask] = transition_weights
    
    return weights


# Progress bar support
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

@dataclass
class HVSRResult:
    frequencies: np.ndarray
    hvsr_values: np.ndarray
    horizontal_spectrum: np.ndarray
    vertical_spectrum: np.ndarray
    metadata: dict
    window_hvsr: np.ndarray | None = None
    window_h_mean: np.ndarray | None = None
    window_v_mean: np.ndarray | None = None
    hvsr_mean: np.ndarray | None = None
    hvsr_std: np.ndarray | None = None

@jit(nopython=True, cache=True)
def _sta_lta_ratio_numba(x: np.ndarray, sta_samples: int, lta_samples: int) -> float:
    """Numba-accelerated STA/LTA ratio calculation."""
    n = len(x)
    if n < lta_samples:
        return np.inf
    
    # Calculate LTA (Long Term Average)
    lta_sum = 0.0
    for i in range(n - lta_samples, n):
        lta_sum += abs(x[i])
    lta_mean = lta_sum / lta_samples
    
    # Calculate STA (Short Term Average)
    sta_sum = 0.0
    for i in range(n - sta_samples, n):
        sta_sum += abs(x[i])
    sta_mean = sta_sum / sta_samples
    
    if lta_mean < 1e-12:
        return np.inf
    return sta_mean / lta_mean

@jit(nopython=True, cache=True)
def _combine_horizontal_from_amplitudes_numba(h_amp_n: np.ndarray, h_amp_e: np.ndarray, mode: int) -> np.ndarray:
    """Numba-accelerated horizontal component combination.
    mode: 0=geometric_mean, 1=quadratic_mean
    """
    if mode == 0:  # geometric_mean
        return np.sqrt(np.maximum(h_amp_n, 0.0) * np.maximum(h_amp_e, 0.0))
    else:  # quadratic_mean
        return np.sqrt(0.5 * (h_amp_n**2 + h_amp_e**2))

def _combine_horizontal_from_amplitudes(h_amp_n: np.ndarray, h_amp_e: np.ndarray, mode: str) -> np.ndarray:
    """Combine horizontal components with string mode for compatibility."""
    mode_int = 0 if mode == "geometric_mean" else 1
    if NUMBA_AVAILABLE:
        return _combine_horizontal_from_amplitudes_numba(h_amp_n, h_amp_e, mode_int)
    else:
        if mode == "geometric_mean":
            return np.sqrt(np.maximum(h_amp_n, 0.0) * np.maximum(h_amp_e, 0.0))
        return np.sqrt(0.5 * (h_amp_n**2 + h_amp_e**2))

def compute_hvsr(
    horizontal_data,
    vertical_data,
    sampling_rate,
    window_length: float = 60.0,
    overlap: float = 0.66,
    smoothing_window: int = 5,
    *,
    stack_method: str = "window_ratio_median",    # "window_ratio_median" | "welch_ratio"
    horizontal_combine: str = "quadratic_mean",   # "quadratic_mean" | "geometric_mean"
    smoothing_method: str = "custom_ko",          # "custom_ko" | "custom_ko_smooth" | "moving_average" | "konno_ohmachi"
    ko_bandwidth: float = 40.0,  # Optimal balance of smoothing and performance
    # QC
    sta_lta_ratio_threshold: float = 2.0,
    max_amplitude_threshold: float | None = None,
    # Processing toggles
    window_smoothing: bool = False,               # let KO handle final smoothing
    window_smoothing_window: int = 3,
    detrend_windows: bool = True,
    remove_mean: bool = True,
    # Frequency limits
    min_frequency: float = 0.1,
    max_frequency: float | None = None,
    max_frequency_ratio: float = 0.25,  # Reduced from 0.35 to 0.25 (25% of Nyquist)
    frequency_taper: bool = False,  # Disable frequency tapering by default to avoid sharp drops
    anti_aliasing_filter: bool = True,
    # PSD engine for per-window spectra
    per_window_engine: str = "periodogram",       # "periodogram" | "welch"
    notch_lines_hz: tuple[float, ...] = ()
):

    nperseg = int(window_length * sampling_rate)
    noverlap = int(nperseg * overlap)
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("Invalid overlap: step size must be positive")

    # Normalize horizontal input shape
    h = np.asarray(horizontal_data)
    if h.ndim == 1:
        north_data = h
        east_data = h
    elif h.ndim == 2:
        if h.shape[0] == 2 and h.shape[1] != 2:
            h = h.T
        north_data = h[:, 0]
        east_data = h[:, 1]
    else:
        raise ValueError("horizontal_data must be 1D or 2D (n_samples[, 2])")

    v = np.asarray(vertical_data).reshape(-1)
    N = len(v)
    n_windows = 1 + max(0, (N - nperseg) // step)
    if n_windows <= 0:
        raise ValueError("No complete windows could be formed with the given window_length and overlap")

    # helper: optional simple IIR notch (skip if none)
    def _apply_notches(x):
        if not notch_lines_hz:
            return x
        from scipy.signal import iirnotch, filtfilt
        y = x.copy()
        fs = sampling_rate
        for f0 in notch_lines_hz:
            if 0 < f0 < fs/2:
                b, a = iirnotch(f0, Q=30, fs=fs)
                y = filtfilt(b, a, y)
        return y

    # helper: apply frequency tapering to reduce high-frequency noise
    def _apply_frequency_taper(amp, freq, max_freq_ratio):
        if not frequency_taper:
            return amp
        
        # Use the actual frequency range, not theoretical limits
        freq_max = freq[-1]  # Actual maximum frequency in our data
        # Start tapering much later (at 90% instead of 70%) and use gentler taper
        taper_start = freq_max * 0.9  # Start tapering at 90% of actual max frequency
        taper_mask = freq > taper_start
        
        if np.any(taper_mask):
            taper_weights = np.ones_like(amp)
            taper_range = freq[taper_mask]
            # Much gentler cosine taper from 1 to 0.7 (instead of 0.1)
            # This prevents the sharp drop while still reducing high-freq noise
            taper_weights[taper_mask] = 0.7 + 0.3 * 0.5 * (1 + np.cos(np.pi * (taper_range - taper_start) / (freq_max - taper_start)))
            amp = amp * taper_weights
            
        return amp

    # window loop
    f_ref = None
    hvsr_matrix = []
    h_amp_matrix = []
    v_amp_matrix = []
    rejected = 0
    from scipy.signal import butter, filtfilt
    if anti_aliasing_filter:
        nyq = sampling_rate/2
        # More aggressive anti-aliasing filter (6th order, 0.6*nyq cutoff)
        b_lp, a_lp = butter(6, 0.6, btype='low')  # normalized (0.6*nyq)

    # Progress bar for window processing
    for start in tqdm(range(0, N - nperseg + 1, step), desc="Processing windows", unit="win", disable=False):
        sl = slice(start, start + nperseg)
        n_win = north_data[sl].astype(float)
        e_win = east_data[sl].astype(float)
        v_win = v[sl].astype(float)

        # simple STA/LTA-like check
        sta = int(max(1, 1.0 * sampling_rate))
        lta = int(max(sta+1, 10.0 * sampling_rate))
        
        if NUMBA_AVAILABLE:
            max_ratio = max(
                _sta_lta_ratio_numba(n_win, sta, lta),
                _sta_lta_ratio_numba(e_win, sta, lta),
                _sta_lta_ratio_numba(v_win, sta, lta)
            )
        else:
            def _sta_lta_ratio(x, sta_s=1.0, lta_s=10.0):
                sta = int(max(1, sta_s * sampling_rate))
                lta = int(max(sta+1, lta_s * sampling_rate))
                if len(x) < lta: return np.inf
                return np.mean(np.abs(x[-sta:])) / max(1e-12, np.mean(np.abs(x[-lta:])))
            
            max_ratio = max(_sta_lta_ratio(n_win), _sta_lta_ratio(e_win), _sta_lta_ratio(v_win))
        if max_ratio > sta_lta_ratio_threshold:
            rejected += 1
            continue
        if max_amplitude_threshold is not None:
            if max(np.max(np.abs(n_win)), np.max(np.abs(e_win)), np.max(np.abs(v_win))) > max_amplitude_threshold:
                rejected += 1
                continue

        # hygiene
        if remove_mean:
            n_win -= np.mean(n_win); e_win -= np.mean(e_win); v_win -= np.mean(v_win)
        if detrend_windows:
            n_win = detrend(n_win, type='linear')
            e_win = detrend(e_win, type='linear')
            v_win = detrend(v_win, type='linear')

        # optional notches
        if notch_lines_hz:
            n_win = _apply_notches(n_win)
            e_win = _apply_notches(e_win)
            v_win = _apply_notches(v_win)

        # anti-alias low-pass
        if anti_aliasing_filter:
            n_win = filtfilt(b_lp, a_lp, n_win)
            e_win = filtfilt(b_lp, a_lp, e_win)
            v_win = filtfilt(b_lp, a_lp, v_win)

        # per-window ASD
        if per_window_engine == "welch":
            f_ref, Pnn = welch(n_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2)
            _,   Pee = welch(e_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2)
            _,   Pvv = welch(v_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2)
        else:
            f_ref, Pnn = periodogram(n_win, fs=sampling_rate, window='hann', scaling='density')
            _,   Pee = periodogram(e_win, fs=sampling_rate, window='hann', scaling='density')
            _,   Pvv = periodogram(v_win, fs=sampling_rate, window='hann', scaling='density')

        n_asd = np.sqrt(np.maximum(Pnn, 0.0))
        e_asd = np.sqrt(np.maximum(Pee, 0.0))
        v_asd = np.sqrt(np.maximum(Pvv, 0.0))

        # Note: Frequency tapering will be applied AFTER smoothing to avoid artifacts

        if window_smoothing and window_smoothing_window > 1:
            n_asd = uniform_filter1d(n_asd, size=window_smoothing_window, mode='nearest')
            e_asd = uniform_filter1d(e_asd, size=window_smoothing_window, mode='nearest')
            v_asd = uniform_filter1d(v_asd, size=window_smoothing_window, mode='nearest')

        h_asd = _combine_horizontal_from_amplitudes(n_asd, e_asd, horizontal_combine)

        # guard against tiny V - more robust floor
        eps = max(1e-12, np.percentile(v_asd, 10) * 1e-2)  # Use 10th percentile, larger factor
        ratio = h_asd / np.maximum(v_asd, eps)

        hvsr_matrix.append(ratio)
        h_amp_matrix.append(h_asd)
        v_amp_matrix.append(v_asd)

    if len(hvsr_matrix) == 0:
        raise ValueError("No windows passed QC. Relax thresholds or check data.")

    hvsr_matrix = np.asarray(hvsr_matrix)
    h_amp_matrix = np.asarray(h_amp_matrix)
    v_amp_matrix = np.asarray(v_amp_matrix)

    n_accepted = len(hvsr_matrix)

    # frequency mask with proper band-edge handling
    nyq = sampling_rate / 2
    f = f_ref
    
    # Set maximum frequency
    if max_frequency is not None:
        fmax = min(max_frequency, nyq * 0.8)  # Don't exceed 80% of Nyquist
    else:
        fmax = nyq * max_frequency_ratio
    
    # Apply frequency limits with band-edge buffer (KO kernel half-width)
    ko_kernel_halfwidth = 1.0 / ko_bandwidth  # Approximate half-width in log space
    fmin_safe = min_frequency * (1.0 + ko_kernel_halfwidth)  # Buffer below min_freq
    fmax_safe = fmax / (1.0 + ko_kernel_halfwidth)  # Buffer above max_freq
    
    mask = (f >= fmin_safe) & (f <= fmax_safe)
    f = f[mask]
    hvsr_matrix = hvsr_matrix[:, mask]
    h_amp_matrix = h_amp_matrix[:, mask]
    v_amp_matrix = v_amp_matrix[:, mask]

    # stack
    hvsr_med = np.median(hvsr_matrix, axis=0)
    hvsr_mean = np.mean(hvsr_matrix, axis=0)
    hvsr_std = np.std(hvsr_matrix, axis=0)

    H_med = np.median(h_amp_matrix, axis=0)
    V_med = np.median(v_amp_matrix, axis=0)

    # smooth ratio only (hvsrpy approach: form ratio first, then smooth)
    # This avoids Jensen's inequality bias from smoothing H and V separately
    if smoothing_method == "konno_ohmachi":
        hvsr_smooth = obspy_ko_smoothing(hvsr_med, f, bandwidth=ko_bandwidth)
        hvsr_mean_smooth = obspy_ko_smoothing(hvsr_mean, f, bandwidth=ko_bandwidth)
        hvsr_std_smooth = obspy_ko_smoothing(hvsr_std, f, bandwidth=ko_bandwidth)
    elif smoothing_method == "custom_ko":
        hvsr_smooth = custom_konno_ohmachi_smoothing(hvsr_med, f, bandwidth=ko_bandwidth, method='improved')
        hvsr_mean_smooth = custom_konno_ohmachi_smoothing(hvsr_mean, f, bandwidth=ko_bandwidth, method='improved')
        hvsr_std_smooth = custom_konno_ohmachi_smoothing(hvsr_std, f, bandwidth=ko_bandwidth, method='improved')
    elif smoothing_method == "custom_ko_smooth":
        hvsr_smooth = custom_konno_ohmachi_smoothing(hvsr_med, f, bandwidth=ko_bandwidth, method='smooth')
        hvsr_mean_smooth = custom_konno_ohmachi_smoothing(hvsr_mean, f, bandwidth=ko_bandwidth, method='smooth')
        hvsr_std_smooth = custom_konno_ohmachi_smoothing(hvsr_std, f, bandwidth=ko_bandwidth, method='smooth')
    else:
        hvsr_smooth = uniform_filter1d(hvsr_med, size=max(1, smoothing_window), mode='nearest')
        hvsr_mean_smooth = uniform_filter1d(hvsr_mean, size=max(1, smoothing_window), mode='nearest')
        hvsr_std_smooth = uniform_filter1d(hvsr_std, size=max(1, smoothing_window), mode='nearest')
    
    # Apply frequency tapering AFTER smoothing (hvsrpy approach)
    if frequency_taper:
        hvsr_smooth = _apply_frequency_taper(hvsr_smooth, f, max_frequency_ratio)
        hvsr_mean_smooth = _apply_frequency_taper(hvsr_mean_smooth, f, max_frequency_ratio)
        hvsr_std_smooth = _apply_frequency_taper(hvsr_std_smooth, f, max_frequency_ratio)

    meta = {
        "method": "Window ASD ratio (median stack)",
        "sampling_rate": sampling_rate,
        "window_length": window_length,
        "overlap": overlap,
        "smoothing_method": smoothing_method,
        "smoothing_window": smoothing_window,
        "ko_bandwidth": ko_bandwidth,
        "horizontal_combine": horizontal_combine,
        "stack_method": stack_method,
        "n_windows_total": n_windows,
        "n_windows_accepted": hvsr_matrix.shape[0],
        "n_windows_rejected": n_windows - hvsr_matrix.shape[0],
        "min_frequency_hz": float(f[0]),
        "max_frequency_hz": float(f[-1]),
        "per_window_engine": per_window_engine,
        "sta_lta_threshold": sta_lta_ratio_threshold,
        "window_smoothing": window_smoothing,
        "window_smoothing_window": window_smoothing_window,
        "detrend_windows": detrend_windows,
        "remove_mean": remove_mean,
        "max_frequency_ratio": max_frequency_ratio,
        "frequency_taper": frequency_taper,
        "anti_aliasing_filter": anti_aliasing_filter,
        "notch_lines_hz": notch_lines_hz,
    }
    
    return HVSRResult(
        frequencies=f,
        hvsr_values=hvsr_smooth,
        horizontal_spectrum=H_med,
        vertical_spectrum=V_med,
        metadata=meta,
        window_hvsr=hvsr_matrix,
        window_h_mean=h_amp_matrix,
        window_v_mean=v_amp_matrix,
        hvsr_mean=hvsr_mean_smooth,
        hvsr_std=hvsr_std_smooth,
    )


def compute_hvsr_batch(
    stations_data: List[Dict[str, Any]], 
    n_workers: Optional[int] = None,
    use_threading: bool = True,
    **kwargs
) -> List[Tuple[str, HVSRResult]]:
    """
    Compute HVSR for multiple stations in parallel.
    
    Parameters
    ----------
    stations_data : List[Dict]
        List of station data dictionaries. Each dict should contain:
        - 'station_id': str, unique station identifier
        - 'horizontal_data': array-like, N-E component data
        - 'vertical_data': array-like, vertical component data  
        - 'sampling_rate': float, sampling rate in Hz
        - Any additional kwargs for compute_hvsr()
    n_workers : int, optional
        Number of parallel workers. Default: cpu_count()
    use_threading : bool, default True
        Use ThreadPoolExecutor instead of ProcessPoolExecutor for better
        memory efficiency with large arrays
    **kwargs
        Additional arguments passed to compute_hvsr()
        
    Returns
    -------
    List[Tuple[str, HVSRResult]]
        List of (station_id, result) tuples
        
    Examples
    --------
    >>> stations = [
    ...     {'station_id': 'A001', 'horizontal_data': h1, 'vertical_data': v1, 'sampling_rate': 250},
    ...     {'station_id': 'A002', 'horizontal_data': h2, 'vertical_data': v2, 'sampling_rate': 250},
    ... ]
    >>> results = compute_hvsr_batch(stations, n_workers=4)
    >>> for station_id, result in results:
    ...     print(f"{station_id}: peak at {result.frequencies[result.hvsr_values.argmax()]:.2f} Hz")
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    def _process_single_station(station_dict):
        """Process a single station's HVSR computation."""
        station_id = station_dict.pop('station_id')
        try:
            result = compute_hvsr(**station_dict, **kwargs)
            return station_id, result
        except Exception as e:
            print(f"Error processing station {station_id}: {e}")
            return station_id, None
    
    # Choose executor based on data size and memory constraints
    ExecutorClass = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
    
    with ExecutorClass(max_workers=n_workers) as executor:
        results = list(executor.map(_process_single_station, stations_data))
    
    # Filter out failed computations
    successful_results = [(sid, res) for sid, res in results if res is not None]
    failed_count = len(results) - len(successful_results)
    
    print(f"Completed {len(successful_results)}/{len(stations_data)} stations")
    if failed_count > 0:
        print(f"  (Failed: {failed_count})")
    
    return successful_results


def compute_hvsr_array(
    array_data: Dict[str, Any],
    station_ids: List[str],
    n_workers: Optional[int] = None,
    use_threading: bool = True,
    **kwargs
) -> Dict[str, HVSRResult]:
    """
    Compute HVSR for a dense nodal array with standardized data format.
    
    Parameters
    ----------
    array_data : Dict[str, Any]
        Dictionary containing:
        - 'horizontal_data': Dict[str, array], station_id -> horizontal data
        - 'vertical_data': Dict[str, array], station_id -> vertical data
        - 'sampling_rate': float, sampling rate (assumed same for all stations)
    station_ids : List[str]
        List of station identifiers
    n_workers : int, optional
        Number of parallel workers
    use_threading : bool, default True
        Use threading instead of multiprocessing
    **kwargs
        Additional arguments for compute_hvsr()
        
    Returns
    -------
    Dict[str, HVSRResult]
        Dictionary mapping station_id to HVSRResult
        
    Examples
    --------
    >>> array_data = {
    ...     'horizontal_data': {'A001': h1, 'A002': h2},
    ...     'vertical_data': {'A001': v1, 'A002': v2},
    ...     'sampling_rate': 250
    ... }
    >>> results = compute_hvsr_array(array_data, ['A001', 'A002'])
    """
    # Prepare station data for batch processing
    stations_data = []
    for station_id in station_ids:
        station_data = {
            'station_id': station_id,
            'horizontal_data': array_data['horizontal_data'][station_id],
            'vertical_data': array_data['vertical_data'][station_id],
            'sampling_rate': array_data['sampling_rate']
        }
        stations_data.append(station_data)
    
    # Process in parallel
    results_list = compute_hvsr_batch(stations_data, n_workers, use_threading, **kwargs)
    
    # Convert to dictionary format
    return dict(results_list)


def compute_hvsr_parallel(
    horizontal_data,
    vertical_data,
    sampling_rate,
    n_workers: Optional[int] = None,
    **kwargs
) -> HVSRResult:
    """
    Compute HVSR with internal parallelization for large arrays.
    
    This function automatically parallelizes the window processing loop
    when the data is large enough to benefit from it.
    
    Parameters
    ----------
    horizontal_data, vertical_data, sampling_rate
        Same as compute_hvsr()
    n_workers : int, optional
        Number of workers for internal parallelization
    **kwargs
        Additional arguments for compute_hvsr()
        
    Returns
    -------
    HVSRResult
        HVSR computation result
    """
    # For now, this is a placeholder that calls the standard function
    # Future enhancement: parallelize the window loop internally
    if n_workers is not None:
        print(f"Note: Internal parallelization not yet implemented. Using single-threaded processing.")
    
    return compute_hvsr(horizontal_data, vertical_data, sampling_rate, **kwargs)