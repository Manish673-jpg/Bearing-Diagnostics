# Bearing Fault Detection using Signal Processing & Machine Learning

This is a self-initiated Masters project where I'm trying to detect and classify bearing faults from raw vibration signals. The idea is to go from raw sensor data all the way to a trained ML model that can automatically tell you what type of fault a bearing has — or if it's healthy.

I'm doing this to learn signal processing and machine learning together on a real engineering problem.

---

## The Data

I used the **CWRU (Case Western Reserve University) Bearing Dataset** — it's a publicly available benchmark dataset widely used in bearing fault detection research.

The data contains vibration signals recorded from a drive-end bearing at 12,000 Hz sampling rate, with 4 conditions:
- **Normal** — healthy bearing
- **Outer Race fault** — 0.007 inch fault diameter
- **Inner Race fault** — 0.007 inch fault diameter
- **Ball fault** — 0.007 inch fault diameter

All signals were recorded at 1797 RPM shaft speed.

---

## Phase 1 — Looking at the Raw Data

First thing I did was just plot the raw signals to see what they look like.

![Raw Signals](figures/inertial_raw_data.png)

You can see the outer ring signal has much higher amplitude than the rest — the fault is physically creating bigger vibrations. The normal signal looks the most "smooth" and bounded. But honestly, just looking at the raw signal doesn't tell you much about *what type* of fault it is.

I also zoomed into a small window (first 0.1 seconds) to see individual impacts:

![Zoomed Raw Data](figures/zoomed_raw_data.png)

In the outer ring plot you can clearly see **periodic bursts** — those are the impacts of the ball hitting the damaged spot on the outer race. Each burst happens at a regular interval. This is what we're trying to detect.

---

## Phase 2 — Characteristic Frequencies

Before doing any signal processing, I calculated the **characteristic frequencies** of the bearing from its geometry. These are the frequencies at which fault impacts would repeat if each component was damaged.

The value use:
- Pitch diameter: 38.5 mm
- Ball diameter: 7.94 mm  
- Number of balls: 9
- Contact angle: 0°
- Shaft speed: 1797 RPM

This gives:
| Frequency | Value | Meaning |
|---|---|---|
| BPFO | ~107 Hz | Ball pass frequency, outer race |
| BPFI | ~162 Hz | Ball pass frequency, inner race |
| BSF | — | Ball spin frequency |
| FTF | — | Fundamental train frequency |

These act like a **fingerprint** — if a bearing has an outer race fault, its vibration signal will contain energy repeating at exactly BPFO times per second.

---

## Phase 3 — Hilbert Envelope Analysis

This is the main signal processing step. The pipeline I built:

1. **Bandpass filter** (3000–5000 Hz) — isolates the resonance band where fault impacts are amplified
2. **Hann window** — reduces spectral leakage before FFT
3. **Hilbert transform** — extracts the envelope (the "outline" of the signal amplitude)
4. **Remove DC component** — subtracts the mean to eliminate the zero-frequency spike
5. **FFT of envelope** — converts to frequency domain to see what frequencies dominate

The result is called the **envelope spectrum**. If there's a fault, you should see a sharp peak at the corresponding characteristic frequency.

![Envelope Analysis](figures/envelope_analysis.png)

**What I found:**

- **Normal signal** — completely flat near BPFO, amplitude ~1e-11. No periodic fault, no peak. Exactly as expected.
- **Outer ring signal** — sharp, dominant peak sitting exactly on the BPFO line at ~107 Hz, with harmonics at 2× and 3× BPFO. This is a textbook outer race fault signature. Very clear result.
- **Inner ring signal** — some activity near BPFI but no clean sharp peak. Inner race faults are harder to detect because the fault rotates through the load zone, causing amplitude modulation that smears the peak.
- **Ball signal** — weak signal overall. Ball faults are generally the hardest to detect with basic envelope analysis.

I also plotted the FFT of the raw signals with the characteristic frequencies marked as reference:

![Indicating Frequencies](figures/indicating_frequencies.png)

---

## Phase 4 — Feature Extraction

After validating the signal processing, I extracted numerical features from the signals to use as input for ML models.

I split each signal into non-overlapping windows of 1024 samples and computed 5 features per window:

| Feature | Why |
|---|---|
| RMS | Overall vibration energy — faulty bearings vibrate harder |
| Kurtosis | Impulsiveness — fault impacts make the signal very spiky. Normal ~3, faulty can be 6–20+ |
| Crest Factor | Peak divided by RMS — another measure of impulsiveness |
| Skewness | Asymmetry of the signal distribution |
| Peak-to-Peak | Absolute vibration range |

This gave me a DataFrame with ~470 rows (windows) × 5 features + 1 label column, saved as `bearing_features.csv`.

---

## Phase 5 — Visualising the Features

**Kurtosis boxplot by fault type:**

![Kurtosis Boxplot](figures/box_plot_kurtosis.png)

Label 1 (Outer Ring) clearly has much higher kurtosis than the others. This is already telling us that kurtosis alone could separate the outer ring fault from normal. The other classes are more overlapping.

**RMS vs Kurtosis scatter plot:**

![RMS vs Kurtosis](figures/Rms vs Kurstosis plot.png)

This is the most interesting result of Phase 1. The outer ring cluster (red) is completely separated from everything else — high RMS and high kurtosis. The normal, inner ring, and ball signals are bunched together on the left with low RMS. This means the features are strong enough for a classifier to separate at least the outer ring fault very cleanly.

---

## What's Next — Month 2

Now that the feature DataFrame is ready, I'll move into ML classification:

- Train **Random Forest**, **SVM**, and **KNN** classifiers on the features
- Evaluate with confusion matrix and classification report
- See if the model can correctly classify all 4 fault types
- If inner ring and ball overlap is a problem, I'll explore adding more features

---

## Libraries Used

`numpy` `scipy` `matplotlib` `pandas` `scikit-learn`

---

## Dataset Source

Case Western Reserve University Bearing Data Center  
https://engineering.case.edu/bearingdatacenter
