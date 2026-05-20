import scipy.io
import scipy.signal
from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


#importing mydata
OR = scipy.io.loadmat(r'C:\Users\manis\Documents\Masters\self project\Data\12k_Drive_End_OR007@6_0')
IR = scipy.io.loadmat(r'C:\Users\manis\Documents\Masters\self project\Data\12k_Drive_End_IR007@6_0')
Ball = scipy.io.loadmat(r'C:\Users\manis\Documents\Masters\self project\Data\12k_Drive_End_BA007@6_0')
NOR = scipy.io.loadmat(r'C:\Users\manis\Documents\Masters\self project\Data\normaldata')


#Signals & Frequencies
signals = [NOR['X097_DE_time'].ravel(),
           OR['X130_DE_time'].ravel(),
           IR['X278_DE_time'].ravel(),
           Ball['X282_DE_time'].ravel(),
           ]


bearing_data = {
    'pitch_diameter' : 38.5,
    'ball_diameter' : 7.94,
    'n' : 9,
    'phi' : 0,
    'shaft_speed' : 1797

}


def signal_length_matching(sgn):
    min_length = min(len(signal) for signal in sgn)

    for i in range(len(sgn)):
        sgn[i] = sgn[i][:min_length]

    return sgn

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
df.to_csv(r"C:\Users\manis\Documents\Masters\self project\Data\data_features\bearing_feature.csv", index=False)



df.boxplot(column='Kurtosis', by='Label')
plt.show()