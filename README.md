# ICBHI Respiratory Sound Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ICBHI%202017-green.svg)](https://bhichallenge.med.auth.gr/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end Machine Learning pipeline to automatically detect respiratory diseases and abnormal lung sounds (**Crackles** and **Wheezes**) from digital stethoscope recordings.

---

## 📌 Project at a Glance

* **The Problem**: Doctors use stethoscopes to listen for lung diseases like COPD and Pneumonia. However, human interpretation is subjective and depends heavily on physician experience and ambient room noise.
* **The Goal**: Build an automated machine learning system that listens to digital stethoscope audio, identifies individual breath cycles, and classifies whether the breathing sound is **Normal**, contains **Crackles**, contains **Wheezes**, or **Both**.
* **The Dataset**: Official **ICBHI 2017 Challenge Database** (126 patients, 920 recordings, 6,898 annotated breath cycles).
* **Current Status**: **80% Complete (Phase 4 Done)**. All 6,898 breath cycles preprocessed and converted into 307 acoustic features. Four classical machine learning classifiers (Random Forest, LightGBM, SVM, Logistic Regression) trained with strict 5-fold patient-aware cross-validation and fully benchmarked on the official ICBHI test cohort (2,756 cycles across 49 patients).

---

## 🧭 Project Roadmap & Status

```mermaid
flowchart LR
    P1["Phase 1: Data Exploration\n(DONE)"] --> P2["Phase 2: Audio Preprocessing\n(DONE)"]
    P2 --> P3["Phase 3: Feature Engineering\n(DONE)"]
    P3 --> P4["Phase 4: ML Model Benchmarking\n(DONE)"]
    P4 --> P5["Phase 5: Deep Learning & CNNs\n(NEXT UP)"]
    
    style P1 fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724
    style P2 fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724
    style P3 fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724
    style P4 fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724
    style P5 fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404
```

| Phase | Description | Status | Key Output |
| :--- | :--- | :---: | :--- |
| **Phase 1: Exploratory Data Analysis** | Analyze demographics, hardware, and sound distributions across 126 patients | `COMPLETED` | Summary tables (`.csv`) & Figures 1–6 |
| **Phase 2: Signal Preprocessing** | Standardize sampling rate (16 kHz), filter heart sounds (50–2,000 Hz bandpass), segment into breath cycles | `COMPLETED` | 6,898 segmented cycles & Figure 7 |
| **Phase 3: Feature Engineering** | Extract 307 acoustic features per cycle (MFCCs, pitch, spectral rolloff, loudness dynamics) | `COMPLETED` | `X_train.csv`, `X_test.csv` & Figures 8–10 |
| **Phase 4: ML Model Benchmarking** | Train & benchmark Random Forest, LightGBM, SVM, and Logistic Regression with patient-aware CV | `COMPLETED` | `models/*.joblib`, Results CSV & Figures 11–13 |
| **Phase 5: Deep Learning (Spectrogram CNNs)** | Train 2D CNNs / ResNet / AST models directly on Mel-Spectrograms | `NEXT UP` | Deep learning models & final report |

---

## 🏆 Model Benchmark Results (Phase 4)

All models were evaluated on the **official ICBHI test cohort** (2,756 respiratory cycles across 49 unseen patients) following official challenge guidelines:
* **Specificity ($S_p$)**: Accuracy on Normal breathing cycles ($Class = 0$).
* **Sensitivity ($S_e$)**: Average recall on abnormal disease cycles ($Class \in \{1, 2, 3\}$).
* **ICBHI Score**: Balanced clinical average: $\frac{S_p + S_e}{2}$.

### 4-Class Task (Normal vs. Crackle vs. Wheeze vs. Both)
| Model | ICBHI Score | Sensitivity ($S_e$) | Specificity ($S_p$) | Accuracy | Macro F1 | 5-Fold CV Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** | **42.85%** | 29.72% | **55.98%** | **47.71%** | **35.66%** | **51.41%** |
| 🥈 **Random Forest** | 39.78% | 31.43% | 48.13% | 44.85% | 33.80% | 48.28% |
| 🥉 **Support Vector Machine (RBF)** | 39.40% | 31.42% | 47.37% | 43.36% | 33.70% | 49.16% |
| 4️⃣ **Logistic Regression** | 36.92% | **39.33%** | 34.52% | 38.86% | 33.23% | 44.00% |

### 2-Class Binary Task (Normal vs. Adventitious / Abnormal)
| Model | ICBHI Score | Sensitivity ($S_e$) | Specificity ($S_p$) | Accuracy | Macro F1 | 5-Fold CV Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** | **55.70%** | 59.22% | 52.18% | 55.19% | 55.09% | 63.54% |
| 🥈 **Support Vector Machine (RBF)** | 55.52% | 54.80% | **56.24%** | **55.62%** | **55.28%** | **64.22%** |
| 🥉 **Random Forest** | 55.00% | 58.45% | 51.55% | 54.50% | 54.40% | 62.71% |
| 4️⃣ **Logistic Regression** | 54.67% | **64.83%** | 44.52% | 53.19% | 53.17% | 62.50% |

---

## 🩺 The 4 Lung Sound Classes (What Are We Detecting?)

Each respiratory cycle falls into one of four clinically defined categories:

```
                  ┌──────────────────────────────┐
                  │   Respiratory Cycle Audio    │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────┬───────┴───────┬───────────────┐
         ▼               ▼               ▼               ▼
     [Normal]        [Crackle]       [Wheeze]          [Both]
     (52.8%)          (27.0%)         (12.8%)          (7.3%)
  Smooth breathing    Short popping  High-pitched     Both crackles &
    without any       clicks (like    whistle from     wheezes in the
   abnormalities      bubble wrap)    narrow airways     same breath
```

1. 🟢 **Normal (52.8%)**: Standard vesicular breath sound with smooth, low-frequency airflow and no obstructions.
2. 🟠 **Crackle (27.0%)**: Discontinuous, explosive popping sounds lasting less than 20–100 ms (like peeling Velcro or popping bubble wrap). Caused by sudden opening of collapsed small airways in **Pneumonia** and **Bronchiectasis**.
3. 🔵 **Wheeze (12.8%)**: Continuous, high-pitched musical whistling lasting over 100 ms. Caused by vibrating narrowed airways in **COPD** and **Asthma**.
4. 🟣 **Both (7.3%)**: Severe breath cycles where crackles and wheezes occur together in the same breath.

---

## 🔬 How Feature Engineering Works (Phase 3)

Machine learning models cannot take raw audio waves directly. We convert each breath cycle into **307 numerical measurements** grouped into 4 intuitive categories:

```
307 Extracted Acoustic Features per Breath Cycle:
├── 1. Sound Texture / Timbre (MFCCs: 240 features)
│      Captures the unique frequency profile and "voice" of the breath sound over time.
│
├── 2. Pitch & Brightness (Spectral Descriptors: 38 features)
│      Measures if the sound is high-pitched (whistling wheeze) or low-pitched (normal breath).
│
├── 3. Loudness & Pops (Temporal Dynamics: 9 features)
│      Measures rapid clicks, zero-crossings, and energy spikes caused by explosive crackles.
│
└── 4. Musical Tones (Chroma STFT: 24 features)
       Detects musical harmonics and definite pitch (wheezes have musical harmonic structure).
```

### Ready-to-Train Benchmark Datasets ([`data/processed/`](data/processed/))
Following the official ICBHI challenge partition, the data is split into patient-independent training and testing sets so models are evaluated on unseen patients:

* **Training Set (`X_train.csv`, `y_train.csv`)**: 4,142 cycles across 79 patients.
* **Testing Set (`X_test.csv`, `y_test.csv`)**: 2,756 cycles across 49 patients.
* **Master Feature Table (`features_all.csv`)**: All 6,898 cycles $\times$ 327 columns (metadata + features).
* **Feature Catalogue (`feature_names.json`)**: Plain text dictionary listing all 307 feature names.

---

## 📊 Visualizations

All publication-ready plots (300 DPI) are automatically generated in [`reports/figures/`](reports/figures/):

### Phase 4: Model Benchmarking Figures
| Figure 11: Benchmark Metric Comparison | Figure 12: Test Set Confusion Matrices |
| :---: | :---: |
| ![Figure 11](reports/figures/fig11_model_benchmark_comparison.png) | ![Figure 12](reports/figures/fig12_confusion_matrices.png) |

| Figure 13: Top 20 Most Predictive Acoustic Biomarkers |
| :---: |
| ![Figure 13](reports/figures/fig13_feature_importances.png) |

### Phases 1–3: Data & Signal Processing Figures
| 1. Disease & Demographics | 2. Sound Class Distribution |
| :---: | :---: |
| ![Demographics](reports/figures/fig1_disease_and_demographics.png) | ![Cycles](reports/figures/fig4_respiratory_cycles_adventitious_sounds.png) |

| 3. Audio Filtering & Segmentation | 4. Benchmark Train / Test Split |
| :---: | :---: |
| ![Preprocessing](reports/figures/fig7_preprocessing_and_segmentation.png) | ![Train/Test Split](reports/figures/fig5_train_test_distribution.png) |

| 5. Acoustic Biomarker Boxplots | 6. Feature Correlation Heatmap |
| :---: | :---: |
| ![Biomarkers](reports/figures/fig8_acoustic_features_distribution.png) | ![Correlation](reports/figures/fig9_feature_correlation_matrix.png) |

| 7. 2D Cluster Visualizations (PCA & t-SNE) |
| :---: |
| ![Manifolds](reports/figures/fig10_pca_tsne_feature_space.png) |

---

## ⚡ Quick Start: How to Run

### 1. Setup Environment
Clone the repository and install the standard scientific Python packages:
```bash
git clone https://github.com/Simplicity005/icbhi-respiratory-detection.git
cd icbhi-respiratory-detection
pip install numpy pandas scipy scikit-learn lightgbm xgboost matplotlib seaborn librosa soundfile soxr
```

### 2. Run Data Exploration (Phase 1)
Generates patient demographic tables and Figures 1 to 6:
```bash
python scripts/run_eda.py
```

### 3. Run Preprocessing & Segmentation (Phase 2)
Filters raw audio (50–2,000 Hz) and segments continuous recordings into individual breath cycles:
```bash
python scripts/preprocess_and_segment.py
```

### 4. Run Feature Extraction (Phase 3)
Extracts all 307 acoustic features using parallel processing across CPU cores:
```bash
python scripts/extract_features.py --workers 8
```

### 5. Run Model Training & Benchmarking (Phase 4)
Trains Random Forest, LightGBM, SVM, and Logistic Regression with 5-fold patient-aware cross-validation and evaluates on the official test cohort:
```bash
python scripts/train_models.py
```

### 6. Open Interactive Notebooks
Explore the pipeline interactively in Jupyter or VS Code:
* [`notebooks/01_ICBHI_Exploratory_Data_Analysis.ipynb`](notebooks/01_ICBHI_Exploratory_Data_Analysis.ipynb): Interactive demographic & sound exploration.
* [`notebooks/02_Preprocessing_and_Segmentation.ipynb`](notebooks/02_Preprocessing_and_Segmentation.ipynb): Interactive Butterworth filtering and audio slicing.
* [`notebooks/03_Feature_Engineering.ipynb`](notebooks/03_Feature_Engineering.ipynb): Real ICBHI audio exploration, spectrograms, and PCA/t-SNE clusters.
* [`notebooks/04_Model_Training_and_Evaluation.ipynb`](notebooks/04_Model_Training_and_Evaluation.ipynb): Patient-aware cross-validation, classifier benchmarking, confusion matrices, and feature importance rankings.

---

## 📁 Repository Structure

```
icbhi-respiratory-detection/
├── data/
│   ├── metadata/                       # Official ICBHI challenge text files
│   └── processed/                      # Preprocessed datasets ready for ML
│       ├── X_train.csv                 # 4,142 train cycles × 307 features
│       ├── y_train.csv                 # Train labels (Normal, Crackle, Wheeze, Both)
│       ├── X_test.csv                  # 2,756 test cycles × 307 features
│       ├── y_test.csv                  # Test labels
│       ├── features_all.csv            # Master dataset (6,898 cycles × 327 columns)
│       ├── feature_names.json          # Dictionary of all 307 features
│       └── segmented_cycles_metadata.csv # Timing boundaries for each cycle
├── models/                             # Trained model weights & scalers
│   ├── lightgbm_4class.joblib          # Best 4-Class model (42.85% ICBHI Score)
│   ├── lightgbm_binary.joblib          # Best Binary model (55.70% ICBHI Score)
│   ├── random_forest_4class.joblib     # Random Forest 4-Class model
│   ├── random_forest_binary.joblib     # Random Forest Binary model
│   ├── svm_binary.joblib               # Support Vector Machine Binary model
│   └── scaler.joblib                   # Fitted StandardScaler
├── notebooks/
│   ├── 01_ICBHI_Exploratory_Data_Analysis.ipynb
│   ├── 02_Preprocessing_and_Segmentation.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Training_and_Evaluation.ipynb
├── reports/
│   ├── figures/                        # High-resolution (300 DPI) publication figures (Fig 1–13)
│   └── model_benchmark_results.csv     # Complete benchmark metrics across all models & tasks
├── scripts/
│   ├── run_eda.py                      # Phase 1 runner
│   ├── preprocess_and_segment.py       # Phase 2 runner
│   ├── extract_features.py             # Phase 3 runner
│   └── train_models.py                 # Phase 4 runner
├── .gitignore
└── README.md
```

---

## 📈 Evaluation Metrics (What We Optimize For)

In respiratory sound detection, accuracy alone is misleading because healthy sounds outnumber abnormal sounds. Following the official ICBHI challenge standard, we evaluate models using:

1. **Sensitivity ($S_e$)**: Ability to correctly detect abnormal breath sounds (Crackles and Wheezes).
$$\text{Sensitivity} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

2. **Specificity ($S_p$)**: Ability to correctly identify normal healthy breathing.
$$\text{Specificity} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}$$

3. **ICBHI Composite Score**: The balanced clinical average:
$$\text{ICBHI Score} = \frac{S_e + S_p}{2}$$

---

## 📚 References
* **ICBHI 2017 Challenge**: Rocha, B.M., et al. *"An open access database for the evaluation of respiratory sound classification algorithms."* Physiological Measurement 40.3 (2019): 035001. [Challenge Portal](https://bhichallenge.med.auth.gr/)
