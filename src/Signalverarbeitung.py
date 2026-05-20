import scipy.io
import scipy.signal
from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



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


##parameters
fs = 12000
n_samples = min(len(signal) for signal in signals)
time = np.linspace(0,n_samples/fs,n_samples)



###functions to match_signals length, tranform signals into frequency spectrum, and get the adjacent frequencies

def signal_length_matching(sgn):
    min_length = min(len(signal) for signal in sgn)

    for i in range(len(sgn)):
        sgn[i] = sgn[i][:min_length]

    return sgn


def window_signal(sgn):
    windowed = []

    for i in range(len(sgn)):
        window = np.hanning(len(sgn[i]))  # create Hann window
        windowed_signal = sgn[i] * window  # apply window
        windowed.append(windowed_signal)

    return windowed


def transform_signals(sgn):
    transformed_signals = []

    for i in range(len(sgn)):
        my_value = np.fft.rfft(sgn[i])
        my_value = my_value / len(sgn[i])
        my_value = np.abs(my_value)
        my_value = np.power(my_value, 2)
        transformed_signals.append(my_value)
    return transformed_signals


def get_frequencies(sgn, fs):
    frequencies = []

    for i in range(len(sgn)):
        my_value = np.fft.rfftfreq(len(sgn[i]), 1 / fs)
        frequencies.append(my_value)

    return frequencies


def apply_filter(sgn, sampling_rate):
    filtered_signals = []
    b, a = butter(4, [3000,5000] , btype='bandpass', fs=sampling_rate)

    for i in range(len(sgn)):
        filtered = filtfilt(b, a, sgn[i])
        filtered_signals.append(filtered)

    return filtered_signals


def char_frequencies(pitch_diameter, ball_diameter, n, phi, fr):
    fr = fr / 60
    bpfo = (n / 2) * fr * (1 - (ball_diameter / pitch_diameter) * np.cos(phi))  ##ballpass frequency, outer race
    bpfi = (n / 2) * fr * (1 + (ball_diameter / pitch_diameter) * np.cos(phi))  ##ballpass frequency, inner race

    bsf = (pitch_diameter / (2 * ball_diameter)) * (
                1 - (np.pow((ball_diameter / pitch_diameter) * np.cos(phi), 2)))  ##ball spin frequency

    ftf = (fr / 2) * (1 - (ball_diameter / pitch_diameter) * np.cos(phi))  ##fundamental train frequency
    my_features = {'BPFO':bpfo,
                   'BPFI':bpfi,
                   'BSF':bsf,
                   'FTF':ftf}

    return my_features


###hilbert+ envelope both is being done with this function
def apply_hilbert(sgn):
    hilbert_signals = []

    for i in range(len(sgn)):
        my_value = scipy.signal.hilbert(sgn[i])
        my_value = np.abs(my_value)
        my_value = np.power(my_value,2)
        hilbert_signals.append(my_value)

    return hilbert_signals

## main programm
Signals = signal_length_matching(signals)
filtered_signals = apply_filter(Signals,fs)
char_data = char_frequencies(bearing_data.get('pitch_diameter'),bearing_data.get('ball_diameter'),bearing_data.get('n'),bearing_data.get('phi'),bearing_data.get('shaft_speed') )

windowed_signal = window_signal(filtered_signals)
hilbert_signals = apply_hilbert(windowed_signal)
for i in range(len(hilbert_signals)):
    hilbert_signals[i] -= np.mean(hilbert_signals[i])

spectrum = transform_signals(hilbert_signals)
frequencies = get_frequencies(hilbert_signals,fs)


fig, axes = plt.subplots(4, 1, figsize=(16, 16))

for i in range(len(spectrum)):
    axes[i].plot(frequencies[i],spectrum[i])
    axes[i].set_xlim(0, 500)
    axes[i].set_ylabel('Amplitude')
    axes[i].set_xlabel('Frequency (Hz)')
    axes[i].axvline(x=char_data['BPFO'], linestyle='--', color='red', label='BPFO')
    axes[i].axvline(x=char_data['BPFI'], linestyle='--', color='green', label='BPFI')
    axes[i].axvline(x=char_data['BSF'], linestyle='--', color='orange', label='BSF')
    axes[i].axvline(x=char_data['FTF'], linestyle='--', color='black', label='FTF')
    axes[i].legend()

axes[0].set_title("Normal Data")
axes[1].set_title("outer ring Data")
axes[2].set_title("inner ring Data")
axes[3].set_title("ball Data")


plt.tight_layout()
plt.show()
