import numpy as np
import scipy.signal as signal
import scipy.stats as stats

def process_BOLD(BOLD_signal, order, TR, cutoffLow, cutoffHigh):

    # Define the butter bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order=5):
        return signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
    # Use the butter bandpass filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    # Define the parameters for the filter
    fs = 1 / TR
    lowcut = cutoffLow / (fs / 2)
    highcut = cutoffHigh / (fs / 2)

    BOLD_mean = np.mean(BOLD_signal, axis=0)
    BOLD_mean = np.expand_dims(BOLD_mean, axis=0)
    ones_matrix = np.ones((BOLD_mean.shape[0], 1))
    BOLD_mean = ones_matrix @ BOLD_mean
    BOLD_regressed = BOLD_signal - BOLD_mean
    BOLD_butter = butter_bandpass_filter(BOLD_regressed, lowcut, highcut, fs, order)
    BOLD_z_score = stats.zscore(BOLD_butter)

    return BOLD_z_score
