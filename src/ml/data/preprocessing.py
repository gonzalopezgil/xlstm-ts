# src/ml/data/preprocessing.py

import warnings

import pywt
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------
# Wavelet denoising
#
# Reference: https://doi.org/10.1002/for.3071
# (the idea of the wavelet denoising and some parameters are taken from this paper)
# -------------------------------------------------------------------------------------------

class NonCausalTransformWarning(UserWarning):
    """Warning raised when an offline transform can leak future observations."""


# Function for padding the data
def pad_data(data, pad_width, mode='edge'):
    return np.pad(data, pad_width, mode=mode)


def _as_float_array(data):
    return np.asarray(data, dtype=float).reshape(-1)


def offline_wavelet_denoising(data, wavelet='db4', level=1):
    """Denoise the whole series at once.

    This is suitable for offline signal smoothing and visual analysis only. It is
    not valid for live forecasting features because reconstructed historical
    values can depend on future observations.
    """
    warnings.warn(
        "offline_wavelet_denoising uses the full series and can leak future "
        "observations into historical values. Use wavelet_denoising or "
        "causal_wavelet_denoising for forecasting features.",
        NonCausalTransformWarning,
        stacklevel=2,
    )

    data = _as_float_array(data)

    # Padding with a width of 100
    padded_data = pad_data(data, pad_width=100, mode='edge')
    # Decompose signal using Wavelet Transform
    coeff = pywt.wavedec(padded_data, wavelet, mode="per", level=level)
    # Estimate the noise level
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    # Calculate the universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(padded_data)))
    # Apply soft thresholding to detail coefficients
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    # Set high-frequency coefficients to zero
    coeff[-level] = np.zeros_like(coeff[-level])
    # Reconstruct the denoised signal
    denoised_data = pywt.waverec(coeff, wavelet, mode='per')
    # Remove the padding
    denoised_data = denoised_data[100:-100]  # Adjust this if necessary

    # Handle edge effects
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]
    elif len(denoised_data) < len(data):
        denoised_data = np.pad(denoised_data, (0, len(data) - len(denoised_data)), 'edge')

    return denoised_data


def causal_wavelet_denoising(
    data,
    wavelet='db4',
    level=1,
    window=256,
    min_periods=64,
    pad_width=100,
):
    """Denoise each timestamp using only current and historical observations.

    The returned value at index ``t`` is computed from ``data[:t + 1]`` only.
    This makes the transform safe to use as an input feature for forecasting
    future values.
    """
    if level < 1:
        raise ValueError("level must be at least 1")
    if window < 1:
        raise ValueError("window must be at least 1")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")
    if pad_width < 0:
        raise ValueError("pad_width cannot be negative")

    data = _as_float_array(data)
    denoised = np.empty_like(data, dtype=float)
    wavelet_obj = pywt.Wavelet(wavelet)

    for end in range(len(data)):
        start = max(0, end - window + 1)
        history = data[start:end + 1]

        if len(history) < min_periods:
            denoised[end] = data[end]
            continue

        current_pad_width = min(pad_width, len(history) - 1)
        padded_history = pad_data(history, (current_pad_width, 0), mode='edge')
        max_level = pywt.dwt_max_level(len(padded_history), wavelet_obj.dec_len)
        current_level = min(level, max_level)

        if current_level < 1:
            denoised[end] = data[end]
            continue

        coeff = pywt.wavedec(
            padded_history,
            wavelet_obj,
            mode="symmetric",
            level=current_level,
        )
        sigma = np.median(np.abs(coeff[-1] - np.median(coeff[-1]))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(padded_history)))
        coeff[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeff[1:]]

        reconstructed = pywt.waverec(coeff, wavelet_obj, mode='symmetric')
        denoised[end] = reconstructed[:len(padded_history)][-1]

    return denoised


# Wavelet denoising function with parameterisation for wavelet type and decomposition level
def wavelet_denoising(data, wavelet='db4', level=1, window=256, min_periods=64, pad_width=100):
    return causal_wavelet_denoising(
        data,
        wavelet=wavelet,
        level=level,
        window=window,
        min_periods=min_periods,
        pad_width=pad_width,
    )

def plot_wavelet_denoising(df, stock):
    # Print some metrics for verification
    snr = 10 * np.log10(np.sum(df['Close_denoised'] ** 2) / np.sum(df['Noise'] ** 2))
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    # Plotting the results
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    loc = 'upper left'

    # Plot Original Signal
    axes[0].plot(df.index, df['Close'], label='Original Signal', color='blue')
    axes[0].legend(loc=loc)
    axes[0].grid(False)  # Disable grid

    # Plot Denoised Signal
    axes[1].plot(df.index, df['Close_denoised'], label='Denoised Signal', color='green')
    axes[1].legend(loc=loc)
    axes[1].grid(False)  # Disable grid

    # Plot Noise
    axes[2].plot(df.index, df['Noise'], label='Extracted Noise', color='red')
    axes[2].legend(loc=loc)
    axes[2].grid(False)  # Disable grid

    fig.suptitle(f"Wavelet Denoising for {stock}", fontsize=16)

    plt.tight_layout() 
    plt.show()

# -------------------------------------------------------------------------------------------
# Process dates
# -------------------------------------------------------------------------------------------

def process_dates(df):
    # Convert the Date column to time zone-naive datetime
    return df.tz_localize(None)
