"""
Seismic utility functions for ASTF-net.
"""

from typing import List, Tuple, Union
import numpy as np
from obspy.core.trace import Trace
from obspy.core import UTCDateTime
from scipy.signal import convolve
from pathlib import Path


def read_lst_file(lst_file: Union[str, Path]) -> List[Path]:
    """
    Read LST file and get SAC file paths.

    Args:
        lst_file: Path to LST file

    Returns:
        sac_files: List of SAC file paths
    """
    with open(lst_file, "r") as f:
        sac_files = [Path(line.strip()) for line in f.readlines()]

    return sac_files


def compute_M0(magnitude: float, is_mw: bool = False) -> float:
    """
    Convert magnitude to seismic moment M0.

    Args:
        magnitude: Earthquake magnitude
        is_mw: Whether the magnitude is moment magnitude.
                Default is False, which means the magnitude is in local magnitude scale.

    Returns:
        m0: Seismic moment in Nm
    """
    if not is_mw:
        moment_magnitude = 0.754 * magnitude + 0.88
    else:
        moment_magnitude = magnitude

    m0 = 10 ** (1.5 * moment_magnitude + 9.105)

    return m0


def get_window_times(
    Z_hdr, before_sec: float = 0.25, after_sec: float = 1.85
) -> Tuple[int, int]:
    """
    Get time window based on P-wave and S-wave arrival times.

    Args:
        Z_hdr: SAC header object
        before_sec: Time before P-wave arrival time in seconds
        after_sec: Time after S-wave arrival time in seconds

    Returns:
        start_time: Start time index
        end_time: End time index
    """

    a = Z_hdr.get("a", None)  # P-wave arrival time
    t0 = Z_hdr.get("t0", None)  # S-wave arrival time
    sampling_interval = Z_hdr.get("delta", None)  # Sampling interval in seconds

    if a is not None and t0 is not None:
        t_p = a  # P-wave arrival time
        t_s = t0  # S-wave arrival time
    elif Z_hdr.t3 > 0 and Z_hdr.t4 > 0:
        t_p = Z_hdr.t3  # P-wave arrival time
        t_s = Z_hdr.t4  # S-wave arrival time
    elif a is not None and Z_hdr.t4 > 0:
        t_p = a  # P-wave arrival time
        t_s = Z_hdr.t4  # S-wave arrival time
    elif Z_hdr.t3 > 0 and t0 is not None:
        t_p = Z_hdr.t3  # P-wave arrival time
        t_s = t0  # S-wave arrival time
    else:
        raise ValueError("No valid P-wave or S-wave arrival times found.")

    # Determine time window based on P-wave or S-wave arrival times
    if Z_hdr.kcmpnm == "HHZ":  # Z component
        start_time = int((t_p - before_sec) / sampling_interval)
        end_time = int((t_p + after_sec) / sampling_interval)
    else:  # T component
        start_time = int((t_s - before_sec) / sampling_interval)
        end_time = int((t_s + after_sec) / sampling_interval)

    return start_time, end_time


def resample_waveform(input_waveform: Trace, target_sampling_rate: int = 100) -> Trace:
    """
    Resample waveform to target sampling rate if they are not the same.

    Args:
        input_waveform: ObsPy trace object
        target_sampling_rate: Target sampling rate in Hz

    Returns:
        Resampled ObsPy trace object
    """
    current_sampling_rate = input_waveform.stats.sampling_rate
    target_sampling_interval = 1 / target_sampling_rate
    if current_sampling_rate != target_sampling_rate:
        print(
            f"Sampling rate {current_sampling_rate} Hz, resampled to {target_sampling_rate} Hz..."
        )
        input_waveform = input_waveform.resample(target_sampling_rate)
        input_waveform.stats.sac.delta = target_sampling_interval
    return input_waveform


def convolve_waveforms(
    EGF_waveform: Trace, ASTF_waveform: Trace, start_time: int, end_time: int
) -> np.ndarray:
    """
    Convolution operation: ASTF divided by M0 then convolved.

    Args:
        EGF_waveform: EGF ObsPy trace object
        ASTF_waveform: ASTF ObsPy trace object
        start_time: Start time index
        end_time: End time index

    Returns:
        Convolved target waveform
    """
    # Get EGF magnitude and calculate M0 of EGF
    Z_hdr_egf = EGF_waveform.stats.sac  # Get SAC header info
    magnitude_egf = Z_hdr_egf.mag
    m0_egf = compute_M0(magnitude_egf)

    # Extract signal data within time window
    EGF_segment = EGF_waveform.data[start_time:end_time]
    ASTF_segment = ASTF_waveform.data

    # Divide ASTF by M0 of EGF
    ASTF_segment /= m0_egf

    # Perform convolution
    target_waveform = convolve(EGF_segment, ASTF_segment, mode="full")

    return target_waveform


def set_sac_header(trace, ASTF_waveform, EGF_waveform) -> None:
    """
    Set SAC header information.

    Args:
        trace: ObsPy trace object to modify
        ASTF_waveform: ASTF ObsPy trace object
        EGF_waveform: EGF ObsPy trace object
    """
    trace.stats.network = EGF_waveform.stats.network  # Use EGF network name
    trace.stats.station = EGF_waveform.stats.station  # Use EGF station name
    trace.stats.sac.kcmpnm = (
        EGF_waveform.stats.sac.kcmpnm
    )  # Set component to EGF component

    # Set station and event coordinates
    trace.stats.sac.evla = EGF_waveform.stats.sac.evla  # Event latitude
    trace.stats.sac.evlo = EGF_waveform.stats.sac.evlo  # Event longitude
    trace.stats.sac.stla = EGF_waveform.stats.sac.evla  # Station latitude
    trace.stats.sac.stlo = EGF_waveform.stats.sac.evlo  # Station longitude

    # Custom SAC fields
    trace.stats.sac.user8 = ASTF_waveform.stats.sac.user8  # Azimuth
    trace.stats.sac.mag = ASTF_waveform.stats.sac.mag  # Magnitude
    trace.stats.sac.user0 = ASTF_waveform.stats.sac.user0  # Seismic moment
    trace.stats.sac.user1 = ASTF_waveform.stats.sac.user1  # Stress drop
    trace.stats.sac.user2 = ASTF_waveform.stats.sac.user2  # Length
    trace.stats.sac.user3 = ASTF_waveform.stats.sac.user3  # Width
    trace.stats.sac.user4 = ASTF_waveform.stats.sac.user4  # Rupture start point
    trace.stats.sac.user5 = (
        ASTF_waveform.stats.sac.user5
    )  # Angle between ellipse major axis and strike

    # Sampling frequency and start time
    trace.stats.sampling_rate = ASTF_waveform.stats.sampling_rate  # Sampling frequency
    trace.stats.starttime = UTCDateTime(0)  # Assume waveform starts at time 0
