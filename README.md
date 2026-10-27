# ICBHI Respiratory Disease Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ICBHI%202017-green.svg)](https://bhichallenge.med.auth.gr/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end classical machine learning pipeline for automated respiratory disease and adventitious sound detection using acoustic biomarkers from the **ICBHI 2017 Respiratory Sound Database**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Summary](#dataset-summary)
- [Key Exploratory Data Analysis (EDA) Findings](#key-exploratory-data-analysis-eda-findings)
- [Preprocessing & Segmentation Pipeline](#preprocessing--segmentation-pipeline)
- [Visualizations](#visualizations)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Overview

Auscultation of respiratory sounds using stethoscopes is a primary clinical method for diagnosing pulmonary disorders such as Chronic Obstructive Pulmonary Disease (COPD), Pneumonia, and Bronchiectasis. However, manual interpretation is subjective and operator-dependent.

This repository implements a **classical machine learning workflow** utilizing handcrafted digital signal processing (DSP) feature extraction:
1. **Signal Preprocessing**: Butterworth bandpass filtering (50–2,000 Hz) and standardized resampling to handle multi-hardware stethoscope data.
2. **Feature Engineering**: Time-frequency representation using Mel-Frequency Cepstral Coefficients (MFCCs), spectral descriptors (Centroid, Rolloff, Contrast, Flatness), and temporal dynamics (ZCR, RMS energy).
3. **Classification**: Benchmarking **Random Forest** (with cost-sensitive class balancing) against **Support Vector Machines (SVM)** and **Gradient Boosting (XGBoost)**.
4. **Validation**: Patient-independent evaluation following the official ICBHI challenge benchmark to prevent data leakage.

---

## Dataset Summary

The **ICBHI 2017 Challenge Database** consists of respiratory sound recordings acquired from 126 subjects across two clinical research centers (Portugal and Greece).

* **Subjects**: 126 patients (76 adults, 49 pediatric, 1 unrecorded)
* **Audio Recordings**: 920 `.wav` files (5.49 total hours)
* **Annotated Cycles**: 6,898 complete respiratory cycles labeled for crackles and wheezes
* **Hardware**: 4 distinct acquisition devices (AKG C417L microphone, 3M Littmann Classic II SE, 3M Littmann 3200, WelchAllyn Meditron)
* **Chest Locations**: 7 anatomical auscultation sites (Trachea, Anterior L/R, Posterior L/R, Lateral L/R)

---

## Key Exploratory Data Analysis (EDA) Findings

### 1. Patient Demographics & Diagnoses
| Disease Diagnosis | Patients ($N$) | Percentage (%) | Primary Age Group | Clinical Notes |
| :--- | :---: | :---: | :---: | :--- |
| **COPD** | 64 | 50.79% | 55 – 84 yrs | Continuous wheezes & crackles |
| **Healthy** | 26 | 20.63% | 0.5 – 60 yrs | Vesicular breath sounds |
| **URTI** | 14 | 11.11% | 1 – 15 yrs | Pediatric upper airway infections |
| **Bronchiectasis** | 7 | 5.56% | 50 – 78 yrs | Persistent coarse crackles |
| **Pneumonia** | 6 | 4.76% | 20 – 75 yrs | Mid/late inspiratory crackles |
| **Bronchiolitis** | 6 | 4.76% | 0.25 – 2 yrs | Exclusively infant cohort (<2 yrs) |
| **LRTI** | 2 | 1.59% | 3 – 58 yrs | Lower respiratory tract infection |
| **Asthma** | 1 | 0.79% | 70 yrs | Expiratory musical wheezing |

### 2. Adventitious Sound Classes (Cycle-Level)
| Class ID | Sound Category | Cycles ($N$) | Percentage (%) | Mean Duration (s) |
| :---: | :--- | :---: | :---: | :---: |
| **0** | **Normal** (No crackles, no wheezes) | 3,642 | 52.80% | $2.73 \pm 1.18$ |
| **1** | **Crackle Only** (Discontinuous) | 1,864 | 27.02% | $2.64 \pm 1.13$ |
| **2** | **Wheeze Only** (Continuous tonal) | 886 | 12.84% | $2.72 \pm 1.22$ |
| **3** | **Both** (Crackle + Wheeze) | 506 | 7.34% | $2.69 \pm 1.14$ |

### 3. Sampling Rate Heterogeneity
Audio recordings have varying native sampling frequencies:
* **44,100 Hz**: 824 files (89.57%)
* **4,000 Hz**: 90 files (9.78%)
* **10,000 Hz**: 6 files (0.65%)
* *Implication*: Resampling all audio to a uniform sampling rate (e.g., 4 kHz or 16 kHz) is required before feature extraction.

### 4. Official Benchmark Split (Patient-Independent)
* **Training Set**: 538 recordings (4,131 cycles) across 79 patients.
* **Testing Set**: 382 recordings (2,756 cycles) across 49 patients.
* *Zero patient leakage* ensures realistic clinical generalization.

---

## Preprocessing & Segmentation Pipeline

Raw stethoscope audio recordings present significant challenges: heterogeneous sampling rates (4 kHz to 44.1 kHz), heart sound contamination (<50 Hz), and varying cycle durations. The preprocessing pipeline standardizes the acoustic signals:

1. **Standardized Resampling (16 kHz)**:
   * Converted all recordings from native sampling frequencies to 16,000 Hz using high-quality sinc resampling (`soxr`) to establish a uniform time-frequency basis.
2. **Butterworth Bandpass Filter (50–2,000 Hz)**:
   * **Low Cutoff (50 Hz)**: Suppresses low-frequency cardiac sounds (S1, S2, murmurs), chest wall friction, and body motion rumble.
   * **High Cutoff (2,000 Hz)**: Filters out high-frequency sensor hiss, powerline harmonics, and environmental background noise.
   * **Implementation**: 4th-order Second-Order Sections (SOS) with bidirectional zero-phase digital filtering (`scipy.signal.sosfiltfilt`).
3. **Signal Normalization**:
   * Removed DC offset (mean demeaning) and scaled peak amplitude to 0.95 to eliminate numerical saturation and boundary clipping.
4. **Clinical Cycle Segmentation**:
   * Slices each recording into expert-annotated respiratory cycles, yielding **6,898 individual cycle `.wav` files** in `data/processed/segmented_cycles/` accompanied by a master metadata table `data/processed/segmented_cycles_metadata.csv`.

---

## Visualizations

All high-resolution figures are automatically generated by the pipelines and saved in `reports/figures/`:

| Disease & Demographics | Recording Acquisition Breakdown |
| :---: | :---: |
| ![Demographics](reports/figures/fig1_disease_and_demographics.png) | ![Acquisition](reports/figures/fig2_recording_acquisition_breakdown.png) |

| Sampling Rate Heterogeneity | Adventitious Sound Breakdown |
| :---: | :---: |
| ![Sampling](reports/figures/fig3_sampling_rate_heterogeneity.png) | ![Cycles](reports/figures/fig4_respiratory_cycles_adventitious_sounds.png) |

| Train/Test Benchmark Split | Spectrograms & Waveforms |
| :---: | :---: |
| ![Train/Test Split](reports/figures/fig5_train_test_distribution.png) | ![Spectrograms](reports/figures/fig6_audio_spectrogram_examples.png) |

| Preprocessing & Respiratory Cycle Segmentation |
| :---: |
| ![Preprocessing & Segmentation](reports/figures/fig7_preprocessing_and_segmentation.png) |

---

## Machine Learning Architecture

```
Raw Audio (.wav)
      │
      ▼
[Preprocessing] ──► Butterworth Bandpass (50–2000 Hz) + Resampling (16 kHz / 4 kHz)
      │
      ▼
[Segmentation]  ──► Cut into annotated respiratory cycles (start_time to end_time)
      │
      ▼
[Feature Extraction]
      ├── MFCCs (13–20 coefficients + Δ + ΔΔ)
      ├── Spectral Centroid, Spread, Rolloff (85%), Contrast, Flatness
      ├── Zero-Crossing Rate (ZCR) & RMS Energy
      └── Summary Statistics (Mean, Std, Skewness, Kurtosis)
      │
      ▼
[Class Balancing] ──► Cost-sensitive learning (`class_weight="balanced"`) or SMOTE
      │
      ▼
[Model Training & Evaluation]
      ├── Random Forest (Primary Model)
      ├── Support Vector Machine (SVM - RBF kernel)
      └── Gradient Boosting (XGBoost / LightGBM)
```

---

## Repository Structure

```
icbhi-respiratory-detection/
├── data/
│   ├── metadata/                       # Official challenge text files
│   │   ├── ICBHI_Challenge_demographic_information.txt
│   │   ├── ICBHI_Challenge_diagnosis.txt
│   │   ├── ICBHI_challenge_train_test.txt
│   │   └── filename_differences.txt
│   └── processed/                      # Extracted summary CSV tables & segmented audio
│       ├── patients_summary.csv
│       ├── recordings_summary.csv
│       ├── cycles_summary.csv
│       ├── segmented_cycles_metadata.csv # Master metadata for all 6,898 cycles
│       └── segmented_cycles/           # 6,898 clean 16 kHz cycle .wav files
├── notebooks/
│   ├── 01_ICBHI_Exploratory_Data_Analysis.ipynb     # Interactive EDA notebook
│   └── 02_Preprocessing_and_Segmentation.ipynb    # Interactive DSP & segmentation notebook
├── reports/
│   └── figures/                        # High-resolution (300 DPI) visualization plots
│       ├── fig1_disease_and_demographics.png
│       ├── fig2_recording_acquisition_breakdown.png
│       ├── fig3_sampling_rate_heterogeneity.png
│       ├── fig4_respiratory_cycles_adventitious_sounds.png
│       ├── fig5_train_test_distribution.png
│       ├── fig6_audio_spectrogram_examples.png
│       └── fig7_preprocessing_and_segmentation.png
├── scripts/
│   ├── run_eda.py                      # Standalone EDA pipeline runner
│   └── preprocess_and_segment.py       # Butterworth DSP filtering & cycle segmentation
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/Simplicity005/icbhi-respiratory-detection.git
cd icbhi-respiratory-detection
pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa soundfile soxr
```

### 2. Run the EDA Pipeline
To regenerate all summary tables and visualization figures:
```bash
python scripts/run_eda.py
```

### 3. Run Preprocessing & Cycle Segmentation
To perform sampling rate standardization (16 kHz), 4th-order zero-phase Butterworth bandpass filtering (50–2,000 Hz), peak normalization, and slice into 6,898 respiratory cycles:
```bash
python scripts/preprocess_and_segment.py
```

### 4. Open Interactive Notebooks
Launch Jupyter or open directly in VS Code:
```bash
jupyter notebook notebooks/01_ICBHI_Exploratory_Data_Analysis.ipynb
jupyter notebook notebooks/02_Preprocessing_and_Segmentation.ipynb
```

---

## Evaluation Metrics

Following official ICBHI challenge conventions, performance is measured using Sensitivity, Specificity, and the composite ICBHI Score to avoid skewed accuracy under class imbalance:

$$\text{Sensitivity } (S_e) = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(Detection of abnormal adventitious sounds)}$$

$$\text{Specificity } (S_p) = \frac{\text{TN}}{\text{TN} + \text{FP}} \quad \text{(Correct identification of normal breath)}$$

$$\text{ICBHI Score} = \frac{S_e + S_p}{2}$$

---

## References
* **ICBHI 2017 Challenge**: Rocha, B.M., et al. *"An open access database for the evaluation of respiratory sound classification algorithms."* Physiological Measurement 40.3 (2019): 035001. [Challenge Portal](https://bhichallenge.med.auth.gr/)
