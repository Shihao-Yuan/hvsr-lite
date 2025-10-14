#!/usr/bin/env python3
"""
Batch HVSR processing script using hvsr-lite.


Author: Shihao Yuan (syuan@mines.edu)

DISCLAIMER: This is a development build. The code may contain errors or unstable functionality.
"""

import numpy as np
import pandas as pd
import os
import glob
from obspy import read, Stream  
from scipy import signal
import pathlib
from collections import defaultdict
from multiprocessing import cpu_count
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# hvsr-lite imports
import sys
sys.path.append('..')
from hvsr_lite.core import compute_hvsr_batch, compute_hvsr
from hvsr_lite.utils import stream_to_dict

# Visualization imports
from matplotlib import pyplot as plt
try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except ImportError:
    HAS_BASEMAP = False

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

# Configuration
# DATA_FOLDER = "/data2/syuan/BHP/data_3rd"  
# OUTPUT_DIR = '/data2/syuan/BHP/Figures_hvsr_lite'  
# EXCEL_FILE = '2022_OCELOT_deployement_summary.xlsx'  

DATA_FOLDER = "/Users/shihao/Research/CSM/Projects/BHP/BHP_OCELOT/SEISMIC/Data" 
OUTPUT_DIR = "./output" 
EXCEL_FILE = '2022_OCELOT_deployement_summary.xlsx'  
N_WORKERS = max(1, int(cpu_count() * 0.6))  # Use 60% of available cores

def load_and_group_seismic_files(data_folder):
    """
    Load and group seismic files by station/date combination.
    
    File naming pattern: {station_id}..{network}.{location}.{year}.{month}.{day}.{hour}.{minute}.{second}.{millisecond}.{component}_RESPREMOVED.miniseed
    Example: 453000893..0.22.2022.08.05.00.00.00.000.E_RESPREMOVED.miniseed
    
    Returns:
        List of tuples (file_list, station_id, date_string)
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    
    # Find all .mseed files (both .mseed and .miniseed extensions)
    all_files = glob.glob(os.path.join(data_folder, "*_RESPREMOVED.miniseed"))
    all_files.extend(glob.glob(os.path.join(data_folder, "*_RESPREMOVED.mseed")))
    
    frames_dict = defaultdict(list)
    for f in all_files:
        base = os.path.basename(f)
        
        parts = base.split('.')
        if len(parts) < 11:  
            continue
            
        key = '.'.join(parts[0:9])  # e.g., 453000893..0.22.2022.08.05.00.00.00.000
        frames_dict[key].append(f)
    
    # Extract station_id and date from the key
    frames = []
    for key, file_list in frames_dict.items():
        if len(file_list) == 3:  # Only keep groups with all three components
            parts = key.split('.')
            station_id = parts[0]  # Station ID
            # Date: parts[4:7] are year, month, day
            date_string = f"{parts[4]}.{parts[5]}.{parts[6]}"  # e.g., "2022.08.05"
            frames.append((sorted(file_list), station_id, date_string))
    
    # Verify all files exist
    for file_list, _, _ in frames:
        for file in file_list:
            if not pathlib.Path(file).exists():
                raise FileNotFoundError(f"file {file} not found; check spelling.")
    
    return frames

def load_station_metadata(excel_file):
    """Load station metadata from Excel file."""
    if not os.path.exists(excel_file):
        return None
    
    try:
        df = pd.read_excel(excel_file)
        return df
    except Exception as e:
        return None

def get_station_coordinates(df, filename):
    """Get station coordinates from metadata DataFrame."""
    if df is None:
        return None, None, None
    
    try:
        df['File'] = df['File'].astype(str).str.strip()
        filename = str(filename).strip()
        row = df[df['File'] == filename]
        
        if not row.empty:
            latitude = row.iloc[0]['Latitude']
            longitude = row.iloc[0]['Longitude']
            elevation = row.iloc[0].get('Elevation', None)
            return latitude, longitude, elevation
        else:
            return None, None, None
    except Exception as e:
        return None, None, None

def process_single_station_hvsr_lite(file_paths, station_id):
    """
    Process a single station using hvsr-lite instead of hvsrpy.
    Equivalent settings to the original hvsrpy processing.
    """
    try:
        
        st_all = Stream()
        for file_path in file_paths:
            st_all += read(file_path)
        
        # Preprocess data
        st_preprocessed = st_all.copy()
        st_preprocessed.detrend(type='linear') 
        st_preprocessed.filter('bandpass', freqmin=0.1, freqmax=24, corners=4, zerophase=True) 
        
        hvsr_data = stream_to_dict(st_preprocessed)
        horizontal_data = np.column_stack([hvsr_data['north'], hvsr_data['east']])
        vertical_data = hvsr_data['vertical']
        sampling_rate = hvsr_data['sampling_rate']
        
        # Process with hvsr-lite
        result = compute_hvsr(
            horizontal_data=horizontal_data,
            vertical_data=vertical_data,
            sampling_rate=sampling_rate,
            window_length=200.0, 
            overlap=0.5, 
            smoothing_method='custom_ko', 
            ko_bandwidth=40.0, 
            horizontal_combine='geometric_mean', 
            min_frequency=0.1, 
            max_frequency=21.0, 
            anti_aliasing_filter=True
        )
        
        return result
        
    except Exception as e:
        return None

def batch_process_stations_hvsr_lite(seismic_frames, n_workers=None):
    """
    Batch process multiple stations using hvsr-lite parallel processing.
    
    Args:
        seismic_frames: List of tuples (file_list, station_id, date_string)
        
    Returns:
        Dictionary mapping (station_id, date_string) -> HVSRResult
    """
    if not seismic_frames:
        return {}
    
    # Prepare station data for batch processing
    stations_data = []
    station_metadata_map = [] 
    
    print("Loading and preprocessing data...")
    for file_paths, station_id, date_string in tqdm(seismic_frames, desc="Preparing data", unit="station"):
        try:
            # Load and preprocess data
            st_all = Stream()
            for file_path in file_paths:
                st_all += read(file_path)
            
            # Preprocess data
            st_preprocessed = st_all.copy()
            st_preprocessed.detrend(type='linear')
            st_preprocessed.filter('bandpass', freqmin=0.1, freqmax=24, corners=4, zerophase=True)
            
            # Convert to hvsr-lite format
            hvsr_data = stream_to_dict(st_preprocessed)
            horizontal_data = np.column_stack([hvsr_data['north'], hvsr_data['east']])
            vertical_data = hvsr_data['vertical']
            sampling_rate = hvsr_data['sampling_rate']
            
            # Add to batch processing list
            station_data = {
                'station_id': f"{station_id}_{date_string}",  
                'horizontal_data': horizontal_data,
                'vertical_data': vertical_data,
                'sampling_rate': sampling_rate
            }
            stations_data.append(station_data)
            station_metadata_map.append((station_id, date_string))
            
        except Exception as e:
            continue
    
    print(f"\nComputing HVSR using {n_workers or cpu_count()} workers...")
    results = compute_hvsr_batch(
        stations_data,
        n_workers=n_workers or cpu_count(),
        use_threading=True,  # Use threading for better memory efficiency
        window_length=200.0,
        smoothing_method='custom_ko',
        ko_bandwidth=40.0,
        horizontal_combine='geometric_mean',
        min_frequency=0.1,
        max_frequency=21.0,
        anti_aliasing_filter=True
    )
    
    results_dict = {}
    for (station_id_date, result), (station_id, date_string) in zip(results, station_metadata_map):
        results_dict[(station_id, date_string)] = result
    
    return results_dict

def plot_hvsr_results_with_map(station_id, date_string, result, coordinates, output_dir):
    """
    Create visualization with map and HVSR curve (adapted from 2_process_data.py).
    
    Args:
        station_id: Station identifier
        date_string: Date string (e.g., "2022.08.05")
        result: HVSRResult object
        coordinates: Tuple of (latitude, longitude, elevation)
        output_dir: Output directory for plots
    """
    try:
        if HAS_BASEMAP and coordinates and coordinates[0] is not None:
            
            fig, (ax_map, ax_hvsr) = plt.subplots(2, 1, figsize=(8, 10), dpi=200, 
                                                   gridspec_kw={'height_ratios': [1, 1]})
            
            latitude, longitude, elevation = coordinates
            
            # Top panel: Basemap
            m = Basemap(resolution='i', projection='cyl',
                        llcrnrlat=33.375, urcrnrlat=33.435,
                        llcrnrlon=-110.855, urcrnrlon=-110.765, ax=ax_map)
            m.arcgisimage(service='World_Imagery', xpixels=2000, dpi=200, verbose=False)
            m.drawstates()
            m.drawparallels(np.arange(33.38, 33.44, 0.02), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-110.85, -110.76, 0.02), labels=[1,0,0,1])
            
            # Plot station
            m.scatter(longitude, latitude, 100, marker='H', color='red', alpha=0.9)
            
            ax_map.annotate(
                f'Station {station_id}',
                xy=(longitude, latitude),
                xytext=(longitude - 0.01, latitude + 0.01),
                arrowprops=dict(facecolor='white', edgecolor='none', width=3, headwidth=10),
                fontsize=14, color='white'
            )
            
        else:
            # Simplified plot without map
            fig, ax_hvsr = plt.subplots(1, 1, figsize=(8, 10), dpi=200)
        
        # Bottom panel: HVSR curve with standard deviation band
        if result.hvsr_std is not None:
            ax_hvsr.fill_between(
                result.frequencies,
                result.hvsr_values - result.hvsr_std,
                result.hvsr_values + result.hvsr_std,
                alpha=0.3, color='lightblue', label='± 1 Std Band'
            )
        
        ax_hvsr.semilogx(result.frequencies, result.hvsr_values, 'r-', linewidth=3, 
                       label='Accepted HVSR Curve')
        
        # Add peak marker
        peak_idx = result.hvsr_values.argmax()
        peak_freq = result.frequencies[peak_idx]
        peak_amp = result.hvsr_values[peak_idx]
        ax_hvsr.plot(peak_freq, peak_amp, 'ro', markersize=10, markerfacecolor='white', 
                     markeredgewidth=2, label=f'f₀: {peak_freq:.3f} Hz')
        
        ax_hvsr.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='H/V=1')
        
        ax_hvsr.set_ylabel("HVSR Amplitude", fontsize=14)
        ax_hvsr.set_xlabel("Frequency (Hz)", fontsize=14)
        ax_hvsr.set_title(f"Accepted HVSR Curves - Station {station_id}", fontsize=14)
        ax_hvsr.legend(fontsize=12, loc='upper right')
        ax_hvsr.grid(True, alpha=0.3)
        
        ax_hvsr.set_xlim(0.2, 20)
        ax_hvsr.set_ylim(bottom=0)
        ax_hvsr.grid(True, linestyle='--', alpha=0.3)
        
        output_file = os.path.join(output_dir, f"{station_id}_{date_string}_hvsr.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        pass

def save_results_to_numpy(results_dict, output_dir):
    """
    Save HVSR results to a single numpy file for all stations.
    """
    if not results_dict:
        return
    
    numpy_dir = os.path.join(output_dir, "numpy_results")
    os.makedirs(numpy_dir, exist_ok=True)
    
    station_ids = []
    date_strings = []
    frequencies_list = []
    hvsr_values_list = []
    hvsr_std_list = []
    horizontal_spectra_list = []
    vertical_spectra_list = []
    metadata_list = []
    peak_frequencies = []
    peak_amplitudes = []
    
    for (station_id, date_string), result in results_dict.items():
        station_ids.append(station_id)
        date_strings.append(date_string)
        frequencies_list.append(result.frequencies)
        hvsr_values_list.append(result.hvsr_values)
        hvsr_std_list.append(result.hvsr_std if result.hvsr_std is not None else np.zeros_like(result.frequencies))
        horizontal_spectra_list.append(result.horizontal_spectrum)
        vertical_spectra_list.append(result.vertical_spectrum)
        metadata_list.append(result.metadata)
        
        # Calculate peak frequency and amplitude
        peak_idx = result.hvsr_values.argmax()
        peak_freq = result.frequencies[peak_idx]
        peak_amp = result.hvsr_values[peak_idx]
        peak_frequencies.append(peak_freq)
        peak_amplitudes.append(peak_amp)
    
    # Save all results in a single file
    all_results_file = os.path.join(numpy_dir, "all_hvsr_results.npz")
    np.savez(all_results_file,
             station_ids=np.array(station_ids),
             date_strings=np.array(date_strings),
             frequencies=np.array(frequencies_list),
             hvsr_values=np.array(hvsr_values_list),
             hvsr_std=np.array(hvsr_std_list),
             horizontal_spectra=np.array(horizontal_spectra_list),
             vertical_spectra=np.array(vertical_spectra_list),
             metadata=np.array(metadata_list),
             peak_frequencies=np.array(peak_frequencies),
             peak_amplitudes=np.array(peak_amplitudes),
             mean_peak_frequency=np.mean(peak_frequencies),
             std_peak_frequency=np.std(peak_frequencies),
             n_stations=len(station_ids))
    
def load_numpy_results(output_dir):
    """
    Load previously saved numpy results for plotting and analysis.
    
    Returns:
        Dictionary containing all HVSR results and statistics
    """
    numpy_dir = os.path.join(output_dir, "numpy_results")
    
    if not os.path.exists(numpy_dir):
        return None
    
    # Load all results from single file
    results_file = os.path.join(numpy_dir, "all_hvsr_results.npz")
    if os.path.exists(results_file):
        data = np.load(results_file, allow_pickle=True)
        return {
            'station_ids': data['station_ids'],
            'date_strings': data['date_strings'],
            'frequencies': data['frequencies'],
            'hvsr_values': data['hvsr_values'],
            'hvsr_std': data['hvsr_std'],
            'horizontal_spectra': data['horizontal_spectra'],
            'vertical_spectra': data['vertical_spectra'],
            'metadata': data['metadata'],
            'peak_frequencies': data['peak_frequencies'],
            'peak_amplitudes': data['peak_amplitudes'],
            'mean_peak_frequency': float(data['mean_peak_frequency']),
            'std_peak_frequency': float(data['std_peak_frequency']),
            'n_stations': int(data['n_stations'])
        }
    
    return None

def create_plots_from_numpy(output_dir):
    """
    Create plots from saved numpy results.
    """
    # Load numpy results
    data = load_numpy_results(output_dir)
    if data is None:
        return
    
    # Load station metadata
    station_metadata = load_station_metadata(EXCEL_FILE)
    
    # Create individual station plots
    for i, station_id in enumerate(data['station_ids']):
        # Reconstruct HVSRResult-like object for plotting
        class SimpleResult:
            def __init__(self, frequencies, hvsr_values, hvsr_std, horizontal_spectrum, vertical_spectrum):
                self.frequencies = frequencies
                self.hvsr_values = hvsr_values
                self.hvsr_std = hvsr_std
                self.horizontal_spectrum = horizontal_spectrum
                self.vertical_spectrum = vertical_spectrum
        
        result = SimpleResult(
            data['frequencies'][i],
            data['hvsr_values'][i],
            data['hvsr_std'][i],
            data['horizontal_spectra'][i],
            data['vertical_spectra'][i]
        )
        
        coordinates = get_station_coordinates(station_metadata, station_id)
        plot_hvsr_results_with_map(station_id, result, coordinates, output_dir)

def main():
    """Main processing function."""
    print(f"\n{'='*60}")
    print(f"HVSR Batch Processing")
    print(f"{'='*60}")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nScanning for data files...")
    seismic_frames = load_and_group_seismic_files(DATA_FOLDER)
    print(f"Found {len(seismic_frames)} station-date combinations")
    
    station_metadata = load_station_metadata(EXCEL_FILE)
    
    # Process stations in parallel
    results = batch_process_stations_hvsr_lite(seismic_frames, n_workers=N_WORKERS)
    
    if not results:
        print("\nNo results generated")
        return
    
    print(f"Successfully processed {len(results)} recordings\n")
    
    # Save results to numpy files for future analysis
    print("Saving results to numpy files...")
    save_results_to_numpy(results, OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/numpy_results/\n")
    
    # Create visualizations
    print("Generating plots...")
    for (station_id, date_string), result in tqdm(results.items(), desc="Creating plots", unit="plot"):
        coordinates = get_station_coordinates(station_metadata, station_id)
        plot_hvsr_results_with_map(station_id, date_string, result, coordinates, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  - {len(results)} HVSR plots saved")
    print(f"  - Numpy results saved")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
