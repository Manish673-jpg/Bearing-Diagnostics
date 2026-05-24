# Bearing Fault Detection — Signal Processing & ML

I started this as a self-initiated Masters project to learn signal processing and machine learning together on a real engineering problem. The goal was simple — take raw vibration signals from a bearing and figure out what fault it has, or if it's healthy. No fancy setup, just the CWRU dataset, Python, and building everything from scratch to actually understand what's happening.

**Dataset:** CWRU Bearing Dataset — 12,000 Hz, 1797 RPM
`0 = Normal` | `1 = Outer Race` | `2 = Inner Race` | `3 = Ball fault`

---

## Month 1 — Signal Processing

### Looking at the raw data first

First thing I did was just plot everything to see what I was working with.

<p align="center"><img src="figures/inertial_raw_data.png" width="320"/>
<img src="figures/zoomed_raw_data.png" width="320"/>
</p>

Outer ring signal is clearly different — much higher amplitude. But just looking at the raw signal doesn't really tell you what type of fault it is. So I zoomed in to the first 0.1 seconds:



The outer ring plot shows periodic bursts — those are the actual impacts of the ball hitting the damaged spot. That pattern is what the whole analysis is built around.

---

### Characteristic Frequencies

Before touching any signal processing, I calculated the frequencies at which fault impacts would repeat — based purely on bearing geometry. These act as a fingerprint.

| Frequency | Value | Meaning |
|---|---|---|
| BPFO | ~107 Hz | Ball pass frequency, outer race |
| BPFI | ~162 Hz | Ball pass frequency, inner race |
| BSF | — | Ball spin frequency |
| FTF | — | Fundamental train frequency |

<p align="center"><img src="figures/indicating_frequencies.png" width="700"/></p>

---

### Hilbert Envelope Analysis

This was the main signal processing step and honestly the most interesting part of Month 1. The pipeline:

**Bandpass filter (3000–5000 Hz) → Hann window → Hilbert transform → DC removal → FFT**

<p align="center"><img src="figures/envelope_analysis.png" width="700"/></p>

- **Normal** — flat. Nothing at BPFO. Exactly what a healthy bearing should look like.
- **Outer Ring** — sharp peak sitting exactly on the BPFO line, with harmonics at 2× and 3×. Textbook result.
- **Inner Ring** — some activity near BPFI but no clean peak. Inner race faults are harder because the fault rotates through the load zone.
- **Ball** — weakest signal overall. Ball faults are just hard to catch with basic envelope analysis.

---

### Feature Extraction

I split each signal into 1024-sample windows and computed 5 features per window:

| Feature | Why it matters |
|---|---|
| RMS | Overall energy — faulty bearings vibrate more |
| Kurtosis | Impulsiveness — fault impacts make signals spiky |
| Crest Factor | Peak/RMS ratio |
| Skewness | Signal asymmetry |
| Peak-to-Peak | Absolute vibration range |

Around 470 windows × 5 features saved to `bearing_features.csv`. Then I plotted to see if the features actually separate the classes:

<p align="center">
  <img src="figures/box_plot_kurtosis.png" width="320"/>
  <img src="figures/Rms_vs_Kurstosis_plot.png" width="320"/>
</p>

Outer Ring is completely isolated — high RMS, high kurtosis. Ball and Normal are bunched together on the left. That overlap became the main problem in Month 2.

---

## Month 2 — Machine Learning

Data split 80/20 with stratified sampling. StandardScaler fitted on training data only.

### Random Forest vs SVM (RBF kernel)

One interesting thing I found — GridSearchCV picked linear kernel as best based on cross validation, but RBF actually performed better on the test set. This confirmed the Ball/Normal boundary is non-linear, which makes sense given the scatter plot.

**Confusion Matrices:**
<p align="center">
  <img src="figures/confustion_matrix_randomclassifier.png" width="320"/>
  <img src="figures/confusion_matrix_svm(rbf).png" width="320"/>
</p>

**Feature Importance:**
<p align="cemter">
  <img src="figures/importance_random classifier.png" width="320"/>
  <img src="figures/Svm_feature_importance(rbf).png" width="320"/>
</p>

Both models agree — RMS and Peak-to-Peak matter most. Crest Factor and Skewness contribute almost nothing.

---

### Results

| Model | Test Accuracy | CV Mean | CV Std |
|---|---|---|---|
| Random Forest | 95.8% | 94.7% | 3.0% |
| SVM RBF | 95.8% | 93.6% | 2.8% |

Both hit 95.8% on the test set. The only consistent mistake: **Ball fault getting classified as Normal** — visible in the scatter plot from Month 1 and confirmed by both confusion matrices. Same accuracy, same problem. That tells me it's a feature problem, not a model problem.

Random Forest wins on cross validation mean — that's the final model.

---

### What I'd do differently / what's next
- Crest Factor and Skewness are weak — would remove them and test if accuracy holds
- Ball/Normal overlap needs better features to fix — maybe frequency domain features from the envelope spectrum directly
- Month 3: try feeding raw signals into a CNN instead of handcrafted features

---

## Project Structure
```
self project/
├── src/
│   ├── Signalverarbeitung.py   # Signal processing
│   ├── data_prep.py            # Loading, splitting, scaling
│   ├── learning_model.py       # Random Forest
│   └── svm_model.py            # SVM
├── figures/
├── Data/                       # .mat files (not uploaded — too large)
└── bearing_features.csv
```

## Libraries
`numpy` `scipy` `matplotlib` `pandas` `scikit-learn`

## Dataset
CWRU Bearing Data Center — https://engineering.case.edu/bearingdatacenter