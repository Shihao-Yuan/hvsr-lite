"""
HVSR computation functions.

Author: Shihao Yuan (syuan@mines.edu)

DISCLAIMER: This is a development build. The code may contain errors or unstable functionality.
"""

try:
    # Python >=3.7
    from dataclasses import dataclass
except ImportError as e: 
    raise ImportError(
        "Missing 'dataclasses'. Use Python >= 3.7 (project requires >= 3.10) "
        "or install the backport: `pip install dataclasses`."
    ) from e
import numpy as np
from scipy.signal import periodogram, welch
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import List, Dict, Any, Tuple, Optional
import warnings


try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install with 'pip install numba' for acceleration.")
    
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args)

# Custom KO implementation to fix ObsPy issues
def custom_konno_ohmachi_smoothing(data, freqs, bandwidth=40.0):
    """
    Custom Konno-Ohmachi smoothing with improved edge handling.
    
    Uses the true Konno-Ohmachi kernel: [sin(b * log10(f/fc)) / (b * log10(f/fc))]^4
    with proper singularity handling and normalization.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    freqs : array-like  
        Frequency array corresponding to data
    bandwidth : float
        Konno-Ohmachi bandwidth parameter (default: 40.0)
    
    Returns:
    --------
    smoothed_data : ndarray
        Smoothed data
    """
    data = np.asarray(data, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    
    if len(data) != len(freqs):
        raise ValueError("Data and frequency arrays must have same length")
    
    # Pre-allocate output
    smoothed = np.zeros_like(data)
    
    # Use Konno-Ohmachi kernel with proper normalization
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
        
        # Normalize kernel
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        
        # Apply kernel
        smoothed[i] = np.sum(data * kernel)
    
    return smoothed


def konno_ohmachi_smoothing_to_centers(
    data: np.ndarray,
    freqs: np.ndarray,
    center_frequencies: np.ndarray,
    bandwidth: float = 40.0,
) -> np.ndarray:
    """
    Konno-Ohmachi smoothing evaluated at arbitrary center frequencies.

    This is useful when you want to report a smoothed spectrum on a
    user-specified frequency grid (instead of the native FFT grid).

    Parameters
    ----------
    data : ndarray
        Input spectrum values defined on `freqs`.
    freqs : ndarray
        Frequency grid for `data`.
    center_frequencies : ndarray
        Center frequencies (Hz) at which to evaluate the smoothed spectrum.
    bandwidth : float
        Konno-Ohmachi bandwidth parameter.

    Returns
    -------
    ndarray
        Smoothed spectrum evaluated at `center_frequencies`.
    """
    data = np.asarray(data, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    centers = np.asarray(center_frequencies, dtype=float)

    if len(data) != len(freqs):
        raise ValueError("Data and frequency arrays must have same length")

    out = np.zeros_like(centers, dtype=float)

    # Evaluate KO kernel on the original frequency grid for each target center
    for i, fc in enumerate(centers):
        if fc <= 0:
            out[i] = np.nan
            continue
        log_ratio = np.log10(freqs / fc)
        kernel = np.ones_like(freqs)
        non_zero = np.abs(log_ratio) > 1e-10
        if np.any(non_zero):
            arg = bandwidth * log_ratio[non_zero]
            kernel[non_zero] = (np.sin(arg) / arg) ** 4
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        out[i] = np.sum(data * kernel)

    return out


# Progress bar support
try:
    from tqdm.auto import tqdm
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
def _sta_lta_ratio_max_numba(x: np.ndarray, sta_samples: int, lta_samples: int) -> float:
    """Numba-accelerated sliding-window STA/LTA ratio calculation.
    
    Calculates the maximum STA/LTA ratio across the entire window using a sliding window.
    This catches transients anywhere in the window.
    
    The sliding window approach:
    - LTA window slides through the entire signal
    - STA is always computed from the last sta_samples within the current LTA window
    - Returns the maximum ratio found anywhere in the window
    
    Parameters:
    -----------
    x : np.ndarray
        Window data
    sta_samples : int
        Number of samples for STA window
    lta_samples : int
        Number of samples for LTA window
        
    Returns:
    --------
    max_ratio : float
        Maximum STA/LTA ratio found anywhere in the window
    """
    n = len(x)
    if n < lta_samples or lta_samples <= sta_samples:
        return np.inf
    
    ax = np.abs(x)
    
    # Initialize: LTA from first lta_samples
    lta_sum = 0.0
    for i in range(lta_samples):
        lta_sum += ax[i]
    lta_mean = lta_sum / lta_samples
    
    # Initial STA from the last sta_samples of the initial LTA window
    sta_sum = 0.0
    for i in range(lta_samples - sta_samples, lta_samples):
        sta_sum += ax[i]
    sta_mean = sta_sum / sta_samples
    
    # Calculate initial ratio
    if lta_mean < 1e-12:
        max_ratio = np.inf
    else:
        max_ratio = sta_mean / lta_mean
    
    # Slide the LTA window through the entire signal
    # At each position, STA is computed from the last sta_samples within the current LTA window
    for i in range(lta_samples, n):
        # Update LTA: remove oldest sample, add newest
        lta_sum = lta_sum + ax[i] - ax[i - lta_samples]
        lta_mean = lta_sum / lta_samples
        
        sta_sum = sta_sum + ax[i] - ax[i - sta_samples]
        sta_mean = sta_sum / sta_samples
        
        # Calculate ratio and track maximum
        if lta_mean < 1e-12:
            current_ratio = np.inf
        else:
            current_ratio = sta_mean / lta_mean
        
        if current_ratio > max_ratio:
            max_ratio = current_ratio
    
    return max_ratio

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
    *,
    horizontal_combine: str = "quadratic_mean",   # "quadratic_mean" | "geometric_mean"
    ko_bandwidth: float = 40.0,  # Konno-Ohmachi smoothing bandwidth parameter
    # QC - STA/LTA
    sta_lta_ratio_threshold: float = 2.5,     # Maximum STA/LTA ratio
    min_sta_lta_ratio: float = 0.2,           # Minimum STA/LTA ratio
    sta_window_seconds: float = 1.0,          # STA window duration in seconds
    lta_window_seconds: float = 30.0,         # LTA window duration in seconds
    # QC - Maximum Value Window Rejection
    maximum_value_threshold: float | None = None,  # Maximum (optionally normalized) amplitude threshold
    maximum_value_normalized: bool = True,     # Whether to normalize before checking
    # Legacy amplitude thresholding (kept for backward compatibility)
    max_amplitude_threshold: float | None = None,
    # Adaptive amplitude thresholding
    amplitude_threshold_factor: float | None = None,   # k_local multiplier for MAD(v_win)
    amplitude_global_cap: float | None = None,         # optional global cap (scalar)
    # Processing toggles
    window_smoothing: bool = False,               # let KO handle final smoothing
    window_smoothing_window: int = 3,
    # Stacking / reporting
    stacking: str = "logmean",                     # "median" | "mean" | "logmean" (geometric mean)
    ko_center_frequencies: np.ndarray | None = None,  # evaluate KO at these centers (Hz)
    # Frequency limits
    min_frequency: float = 0.1,
    max_frequency: float | None = None,
    max_frequency_ratio: float = 0.25,  # Reduced from 0.35 to 0.25 (25% of Nyquist)
    # PSD engine for per-window spectra
    per_window_engine: str = "periodogram",       # "periodogram" | "welch"
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

    # window loop
    f_ref = None
    hvsr_matrix = []
    h_amp_matrix = []
    v_amp_matrix = []
    rejected = 0

    # Progress bar for window processing
    for start in tqdm(range(0, N - nperseg + 1, step), desc="Processing windows", unit="win", disable=False):
        sl = slice(start, start + nperseg)
        n_win = north_data[sl].astype(float)
        e_win = east_data[sl].astype(float)
        v_win = v[sl].astype(float)

        # STA/LTA window rejection
        sta = int(max(1, sta_window_seconds * sampling_rate))
        lta = int(max(sta+1, lta_window_seconds * sampling_rate))
        
        if NUMBA_AVAILABLE:
            ratios = [
                _sta_lta_ratio_max_numba(n_win, sta, lta),
                _sta_lta_ratio_max_numba(e_win, sta, lta),
                _sta_lta_ratio_max_numba(v_win, sta, lta)
            ]
        else:
            def _sta_lta_ratio_max(x, sta_samples, lta_samples):
                n = len(x)
                if n < lta_samples or lta_samples <= sta_samples:
                    return np.inf
                ax = np.abs(x)
                lta_sum = np.sum(ax[:lta_samples])
                sta_sum = np.sum(ax[lta_samples - sta_samples:lta_samples])
                lta_mean = lta_sum / lta_samples
                if lta_mean < 1e-12:
                    max_ratio = np.inf
                else:
                    max_ratio = (sta_sum / sta_samples) / lta_mean
                for i in range(lta_samples, n):
                    lta_sum += ax[i] - ax[i - lta_samples]
                    sta_sum += ax[i] - ax[i - sta_samples]
                    lta_mean = lta_sum / lta_samples
                    if lta_mean < 1e-12:
                        current_ratio = np.inf
                    else:
                        current_ratio = (sta_sum / sta_samples) / lta_mean
                    if current_ratio > max_ratio:
                        max_ratio = current_ratio
                return max_ratio
            
            ratios = [
                _sta_lta_ratio_max(n_win, sta, lta),
                _sta_lta_ratio_max(e_win, sta, lta),
                _sta_lta_ratio_max(v_win, sta, lta)
            ]
        
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        
        # Two-sided STA/LTA check
        # Reject if ratio is outside [min_sta_lta_ratio, sta_lta_ratio_threshold]
        if max_ratio > sta_lta_ratio_threshold or min_ratio < min_sta_lta_ratio:
            rejected += 1
            continue
        
        # Maximum Value Window Rejection
        if maximum_value_threshold is not None:
            components = [n_win, e_win, v_win]
            reject_max_value = False
            
            for comp in components:
                if maximum_value_normalized:
                    comp_std = np.std(comp)
                    if comp_std > 1e-12:
                        comp_normalized = comp / comp_std
                        comp_max_normalized = np.max(np.abs(comp_normalized))
                    else:
                        comp_max_normalized = 0.0
                else:
                    # Use raw maximum absolute value
                    comp_max_normalized = np.max(np.abs(comp))
                
                if comp_max_normalized > maximum_value_threshold:
                    reject_max_value = True
                    break
            
            if reject_max_value:
                rejected += 1
                continue
        # Amplitude-based rejection 
        threshold_limit = None
        if amplitude_threshold_factor is not None:
            
            v_med = np.median(v_win)
            v_mad = np.median(np.abs(v_win - v_med))
            # Fallback to std if MAD is ~0 
            v_scale = 1.4826 * v_mad if v_mad > 0 else np.std(v_win)
            local_limit = amplitude_threshold_factor * max(1e-12, v_scale)
            threshold_limit = local_limit
        if max_amplitude_threshold is not None:
            threshold_limit = max_amplitude_threshold if threshold_limit is None else min(threshold_limit, max_amplitude_threshold)
        if amplitude_global_cap is not None:
            threshold_limit = amplitude_global_cap if threshold_limit is None else min(threshold_limit, amplitude_global_cap)
        if threshold_limit is not None:
            if max(np.max(np.abs(n_win)), np.max(np.abs(e_win)), np.max(np.abs(v_win))) > threshold_limit:
                rejected += 1
                continue

        # per-window ASD
        if per_window_engine == "welch":
            f_ref, Pnn = welch(n_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2, average="median")
            _,   Pee = welch(e_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2, average="median")
            _,   Pvv = welch(v_win, fs=sampling_rate, window='hann', nperseg=nperseg//2, noverlap=(nperseg//2)//2, average="median")
        else:
            f_ref, Pnn = periodogram(n_win, fs=sampling_rate, window='hann', scaling='density')
            _,   Pee = periodogram(e_win, fs=sampling_rate, window='hann', scaling='density')
            _,   Pvv = periodogram(v_win, fs=sampling_rate, window='hann', scaling='density')

        n_asd = np.sqrt(np.maximum(Pnn, 0.0))
        e_asd = np.sqrt(np.maximum(Pee, 0.0))
        v_asd = np.sqrt(np.maximum(Pvv, 0.0))

        if window_smoothing and window_smoothing_window > 1:
            n_asd = uniform_filter1d(n_asd, size=window_smoothing_window, mode='nearest')
            e_asd = uniform_filter1d(e_asd, size=window_smoothing_window, mode='nearest')
            v_asd = uniform_filter1d(v_asd, size=window_smoothing_window, mode='nearest')

        h_asd = _combine_horizontal_from_amplitudes(n_asd, e_asd, horizontal_combine)

        ratio = h_asd / v_asd 

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

    # Stack across windows
    hvsr_med = np.median(hvsr_matrix, axis=0)
    hvsr_mean = np.mean(hvsr_matrix, axis=0)
    hvsr_std = np.std(hvsr_matrix, axis=0)

    # Log-mean (geometric mean) across windows
    _eps = 1e-20
    log_hvsr = np.log(np.maximum(hvsr_matrix, _eps))
    hvsr_logmean = np.exp(np.mean(log_hvsr, axis=0))

    H_med = np.median(h_amp_matrix, axis=0)
    V_med = np.median(v_amp_matrix, axis=0)

    # Smooth ratio only
    if ko_center_frequencies is not None:
        centers = np.asarray(ko_center_frequencies, dtype=float)
        centers = centers[(centers >= f[0]) & (centers <= f[-1])]
        if len(centers) == 0:
            raise ValueError("ko_center_frequencies has no values inside the usable frequency band")

        hvsr_med_smooth = konno_ohmachi_smoothing_to_centers(hvsr_med, f, centers, bandwidth=ko_bandwidth)
        hvsr_mean_smooth = konno_ohmachi_smoothing_to_centers(hvsr_mean, f, centers, bandwidth=ko_bandwidth)
        hvsr_std_smooth = konno_ohmachi_smoothing_to_centers(hvsr_std, f, centers, bandwidth=ko_bandwidth)
        hvsr_logmean_smooth = konno_ohmachi_smoothing_to_centers(hvsr_logmean, f, centers, bandwidth=ko_bandwidth)

        H_med_s = konno_ohmachi_smoothing_to_centers(H_med, f, centers, bandwidth=ko_bandwidth)
        V_med_s = konno_ohmachi_smoothing_to_centers(V_med, f, centers, bandwidth=ko_bandwidth)

        f_out = centers
        H_out = H_med_s
        V_out = V_med_s
    else:
        hvsr_med_smooth = custom_konno_ohmachi_smoothing(hvsr_med, f, bandwidth=ko_bandwidth)
        hvsr_mean_smooth = custom_konno_ohmachi_smoothing(hvsr_mean, f, bandwidth=ko_bandwidth)
        hvsr_std_smooth = custom_konno_ohmachi_smoothing(hvsr_std, f, bandwidth=ko_bandwidth)
        hvsr_logmean_smooth = custom_konno_ohmachi_smoothing(hvsr_logmean, f, bandwidth=ko_bandwidth)
        f_out = f
        H_out = H_med
        V_out = V_med

    if stacking not in {"median", "mean", "logmean"}:
        raise ValueError("stacking must be one of: 'median', 'mean', 'logmean'")

    if stacking == "median":
        hvsr_primary = hvsr_med_smooth
    elif stacking == "mean":
        hvsr_primary = hvsr_mean_smooth
    else:  # "logmean"
        hvsr_primary = hvsr_logmean_smooth

    meta = {
        "method": "Window ASD ratio (stacked)",
        "sampling_rate": sampling_rate,
        "window_length": window_length,
        "overlap": overlap,
        "ko_bandwidth": ko_bandwidth,
        "horizontal_combine": horizontal_combine,
        "n_windows_total": n_windows,
        "n_windows_accepted": hvsr_matrix.shape[0],
        "n_windows_rejected": n_windows - hvsr_matrix.shape[0],
        "min_frequency_hz": float(f_out[0]),
        "max_frequency_hz": float(f_out[-1]),
        "stacking": stacking,
        "ko_center_frequencies": None if ko_center_frequencies is None else f"len={len(f_out)}",
        "per_window_engine": per_window_engine,
        "sta_lta_threshold": sta_lta_ratio_threshold,
        "min_sta_lta_ratio": min_sta_lta_ratio,
        "sta_window_seconds": sta_window_seconds,
        "lta_window_seconds": lta_window_seconds,
        "maximum_value_threshold": maximum_value_threshold,
        "maximum_value_normalized": maximum_value_normalized,
        "max_amplitude_threshold": max_amplitude_threshold,
        "amplitude_threshold_factor": amplitude_threshold_factor,
        "amplitude_global_cap": amplitude_global_cap,
        "window_smoothing": window_smoothing,
        "window_smoothing_window": window_smoothing_window,
        "max_frequency_ratio": max_frequency_ratio,
    }
    
    return HVSRResult(
        frequencies=f_out,
        hvsr_values=hvsr_primary,
        horizontal_spectrum=H_out,
        vertical_spectrum=V_out,
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
    
    results_list = compute_hvsr_batch(stations_data, n_workers, use_threading, **kwargs)
    
    return dict(results_list)