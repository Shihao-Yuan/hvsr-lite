#!/usr/bin/env python3
"""
Example script showing how to load and use HVSR results from the single numpy file.

Author: Shihao Yuan (syuan@mines.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_hvsr_results(output_dir):
    """
    Load HVSR results from the single numpy file.
    
    Returns:
        Dictionary containing all HVSR results and statistics
    """
    results_file = os.path.join(output_dir, "numpy_results", "all_hvsr_results.npz")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    data = np.load(results_file, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"Loaded HVSR Results")
    print(f"{'='*60}")
    print(f"File: {results_file}")
    print(f"Number of stations: {data['n_stations']}")
    print(f"Mean peak frequency: {float(data['mean_peak_frequency']):.2f} Hz")
    print(f"Peak frequency std: {float(data['std_peak_frequency']):.2f} Hz")
    print(f"{'='*60}\n")
    
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

def plot_individual_station(data, index):
    """
    Plot HVSR for a single station by index.
    """
    station_id = data['station_ids'][index]
    date_string = data['date_strings'][index]
    frequencies = data['frequencies'][index]
    hvsr_values = data['hvsr_values'][index]
    hvsr_std = data['hvsr_std'][index]
    peak_freq = data['peak_frequencies'][index]
    peak_amp = data['peak_amplitudes'][index]
    
    plt.figure(figsize=(10, 6))
    
    # Plot standard deviation band
    plt.fill_between(frequencies, hvsr_values - hvsr_std, hvsr_values + hvsr_std,
                     alpha=0.3, color='lightblue', label='±1σ')
    
    # Plot HVSR curve
    plt.semilogx(frequencies, hvsr_values, 'r-', linewidth=2, label='HVSR')
    
    # Plot peak
    plt.plot(peak_freq, peak_amp, 'ro', markersize=10, markerfacecolor='white',
             markeredgewidth=2, label=f'f₀: {peak_freq:.2f} Hz')
    
    # H/V=1 reference
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='H/V=1')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('HVSR Amplitude', fontsize=12)
    plt.title(f'HVSR - Station {station_id} ({date_string})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 20)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def main():
    """Main function with examples."""
    # Default output directory
    output_dir = "/data2/syuan/BHP/Figures_hvsr_lite"
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    # Load results
    data = load_hvsr_results(output_dir)
    
    if data is None:
        print("Failed to load results")
        return
    
    # Print station summary
    print_station_summary(data)
    
    # Example analyses (uncomment to run)
    
    # 1. Plot first station
    # print("Plotting first station...")
    # plot_individual_station(data, 0)
    


if __name__ == "__main__":
    main()

