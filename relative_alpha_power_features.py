"""
relative_alpha_power_features.py - feature extraction script for the
Eyes-Open/Eyes-Closed-Alpha-Test.

Excpected input sampling rate at 250 Hz (config.SAMPLING_RATE_HZ).
GIPSA-training data (512 Hz) needs resampling ahead of calling this function
(check pretrain_gipsa.py)
"""

import numpy as np
from scipy.signal import welch

from config import ALPHA_BAND_HZ, BROADBAND_HZ, SAMPLING_RATE_HZ


def relative_alpha_power(window_samples, fs=SAMPLING_RATE_HZ,
                          alpha_band=ALPHA_BAND_HZ, broadband=BROADBAND_HZ):
    """ Calculates the per-channel log-transformed relative alpha power for each time window.

    window_samples: np.ndarray, shape (n_samples, n_channels) - already the 4 
    pre-selected alpha channels (P3, P4, O1, O2)

    Returns: np.ndarray, shape (n_channels,) -> single feature value per channel.
    No mean over all channels so that classifier can potentially weight channels individually.
    """
    n_samples = window_samples.shape[0]

    # nperseg is central parameter of the Welch method:

    # decides on how long each of the overlapping sub-segment within the 4s window is that 
    # is taken for the welch method to calculate the frequency power via FFT
    # we have 250 Hz fs -> so fs*2 = 500 Hz = 2s sub-segments - Welch takes 
    # mean over the overlapping windows and smoothes the decision additionally.

    # additionally: 250 Hz (fs) /500 Hz (nperseg) = 0.5 Hz  frequency resolution (defined through the nperseg Hz)
    # -> so FFT is computed in 0.5 Hz steps over the frequency spectrum
    # This follows the principal: the longer a time period that is observed for frequency computation: the higher 
    # the precision of the frequency in the lower frequency range (as lower frequencies need more time to form)
    # vice versa: worse frequency precision in the higher frequency ranges
    # here: only the 8-12 Hz range is relevant, so no need to account for changes in the frequency spectrum 
    # (otherwise consider using wavelets)

    # this means: we could take a smaller nperseg here -> it would reduce the sub-segment window size -> would 
    # allow for more smoothing of the frequency information within one window ! at the cost of frequency resolution !
    # (smaller nperseg => lower frequency resolution)
    nperseg = min(n_samples, int(fs * 2))     # using "min() just in case the window is smaller than fs*2

    freqs, psd = welch(
        window_samples, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        axis=0, detrend="constant",  # removes DC-offset per sub-segment
    )

    alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    broadband_mask = (freqs >= broadband[0]) & (freqs <= broadband[1])

    alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask], axis=0)
    broadband_power = np.trapezoid(psd[broadband_mask], freqs[broadband_mask], axis=0)

    relative_power = alpha_power / broadband_power
    return np.log(relative_power)