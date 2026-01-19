# hvsr-lite

**Minimal HVSR (Horizontal-to-Vertical Spectral Ratio) analysis for seismic data**

A Python package for HVSR computation, supporting both single-station analysis and batch processing of dense nodal array data.

**Author:** Shihao Yuan (syuan@mines.edu)

> **DISCLAIMER:** This is a development build. The code may contain errors or unstable functionality. Contributions and feedback are welcome.

## Features

- **Minimal dependencies** - Only NumPy, SciPy, and ObsPy required (Numba optional for acceleration)
- **Parallel processing** - Multi-core batch processing for dense arrays
- **Quality control** - Automatic window rejection based on STA/LTA and amplitude thresholds

## Installation

Install from source:

```bash
git clone https://github.com/Shihao-Yuan/hvsr-lite.git
cd hvsr-lite
pip install -e .
```

## Quick Start

```python
from obspy import read
import numpy as np
from hvsr_lite.core import compute_hvsr
from hvsr_lite.utils import stream_to_dict

# Load seismic data (3 components: N, E, Z)
st = read('path/to/data/*.mseed')

# Option 1: Use the stream_to_dict utility
data = stream_to_dict(st)
horizontal_data = np.column_stack([data['north'], data['east']])
vertical_data = data['vertical']
sampling_rate = data['sampling_rate']

# Option 2: Extract components manually
# north = st.select(component='N')[0].data
# east = st.select(component='E')[0].data
# vertical = st.select(component='Z')[0].data
# sampling_rate = st[0].stats.sampling_rate
# horizontal_data = np.column_stack([north, east])

# Compute HVSR
result = compute_hvsr(
    horizontal_data=horizontal_data,
    vertical_data=vertical_data,
    sampling_rate=sampling_rate,
    window_length=40.0,  # seconds
    overlap=0.5,  # 50% overlap
    ko_bandwidth=40.0,  # Konno-Ohmachi smoothing bandwidth
    min_frequency=0.1,  # Hz
    max_frequency=20.0,  # Hz
    horizontal_combine="geometric_mean", # "geometric_mean" or "quadratic_mean"
    stacking="logmean"  # "logmean", "median", or "mean"
)

# Access results
frequencies = result.frequencies
hvsr_curve = result.hvsr_values
print(f"Peak frequency: {frequencies[hvsr_curve.argmax()]:.2f} Hz")
```

### Parallel Processing Functions

**`compute_hvsr_batch(stations_data, n_workers=None, use_threading=True, **kwargs)`** - Batch processing

Process multiple stations in parallel using ThreadPoolExecutor or ProcessPoolExecutor.

```python
stations_data = [
    {'station_id': 'A001', 'horizontal_data': h1, 'vertical_data': v1, 'sampling_rate': 250},
    {'station_id': 'A002', 'horizontal_data': h2, 'vertical_data': v2, 'sampling_rate': 250},
]
results = compute_hvsr_batch(stations_data, n_workers=4)
# Returns: List[Tuple[str, HVSRResult]]
```

**`compute_hvsr_array(array_data, station_ids, n_workers=None, **kwargs)`** - Array processing

Process dense nodal arrays with standardized data format.

```python
array_data = {
    'horizontal_data': {'A001': h1, 'A002': h2},
    'vertical_data': {'A001': v1, 'A002': v2},
    'sampling_rate': 250
}
results = compute_hvsr_array(array_data, ['A001', 'A002'], n_workers=4)
# Returns: Dict[str, HVSRResult]
```

### `compute_hvsr()`

Main function for HVSR computation with extensive parameter control:

**Parameters:**
- `horizontal_data` - North and East component data (2D array [n_samples, 2] or 1D)
- `vertical_data` - Vertical component data (1D array)
- `sampling_rate` - Sampling rate in Hz
- `window_length` - Time window length in seconds (default: 60.0)
- `overlap` - Window overlap fraction (default: 0.66)
- `horizontal_combine` - Method to combine horizontal components: 'quadratic_mean' (default) or 'geometric_mean'
- `ko_bandwidth` - Konno-Ohmachi smoothing bandwidth parameter (default: 40.0)
- `stacking` - Method to stack windows: 'logmean' (default, geometric mean), 'median', or 'mean'
- `min_frequency` - Minimum frequency in Hz (default: 0.1)
- `max_frequency` - Maximum frequency in Hz (default: None)
- `sta_lta_ratio_threshold` - Maximum STA/LTA ratio for window rejection (default: 2.5)
- `per_window_engine` - PSD engine for per-window spectra: 'periodogram' (default) or 'welch'
- And many more QC and processing options (adaptive amplitude thresholding, frequency ratio limits, etc.)

**Returns:**
- `HVSRResult` object with:
  - `frequencies` - Frequency array
  - `hvsr_values` - Smoothed HVSR curve
  - `hvsr_mean` - Mean HVSR across windows
  - `hvsr_std` - Standard deviation
  - `window_hvsr` - Individual window HVSR curves
  - `horizontal_spectrum` - Smoothed horizontal amplitude spectrum
  - `vertical_spectrum` - Smoothed vertical amplitude spectrum
  - `metadata` - Dictionary of processing parameters and stats

