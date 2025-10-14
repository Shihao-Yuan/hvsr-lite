"""
Utility functions for hvsr-lite notebooks and examples.

Author: Shihao Yuan (syuan@mines.edu)

This module provides simple helper functions for common tasks in notebooks,
such as converting between ObsPy Stream objects and dictionary format.

DISCLAIMER: This is a development build. The code may contain errors or unstable functionality.
"""

from obspy import Stream


def stream_to_dict(st: Stream) -> dict:
    """
    Convert an ObsPy Stream to dictionary format for hvsr-lite.
    
    Parameters
    ----------
    st : obspy.Stream
        ObsPy Stream containing seismic data with N, E, Z components
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'sampling_rate': float, sampling rate in Hz
        - 'north': ndarray, north component data
        - 'east': ndarray, east component data  
        - 'vertical': ndarray, vertical component data
        
    Examples
    --------
    >>> from obspy import read
    >>> from hvsr_lite.utils import stream_to_dict
    >>> st = read('data/*.mseed')
    >>> data = stream_to_dict(st)
    >>> print(data.keys())
    dict_keys(['sampling_rate', 'north', 'east', 'vertical'])
    """
    data = {}
    data['sampling_rate'] = st[0].stats.sampling_rate
    
    for tr in st:
        comp = tr.stats.channel[-1].lower()
        if comp in {'z', 'n', 'e'}:
            name = {'z': 'vertical', 'n': 'north', 'e': 'east'}[comp]
            data[name] = tr.data
            
    return data

