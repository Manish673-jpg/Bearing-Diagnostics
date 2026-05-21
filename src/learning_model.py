import scipy.io
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import sklearn
from Signalverarbeitung import signal_length_matching,signals,bearing_data


def extract_features(sgn,window_size = 1024):
    features = []
    n_windows = len(sgn) // window_size
    trimmed = np.array(sgn[:n_windows * window_size])
    windows = trimmed.reshape(n_windows, window_size)

    for window in windows:
        rms = np.sqrt(np.mean(window**2))
        kurt = scipy.stats.kurtosis(window)
        crest = np.max(window)/rms
        skew = scipy.stats.skew(window)
        peak_to_peak = np.max(window)-np.min(window)
        features.append([rms, kurt,crest, skew, peak_to_peak])

    return np.array(features)

def build_dataset(sgn):
    values = []
    labels = []
    for i in range(len(sgn)):
        my_value = extract_features(sgn[i],window_size=1024)
        values.append(my_value)
        labels.append(np.full(len(values[i]),i))

    all_features = np.vstack(values)
    all_labels = np.concatenate(labels)
    df = pd.DataFrame(all_features,columns=['RMS', 'Kurtosis','Crest_factor','Skewness','Peak_to_Peak'])
    df['Label'] = all_labels

    return df



Signals = signal_length_matching(signals)
windowed_signal = extract_features(Signals[0],window_size=1024)

df = build_dataset(Signals)
