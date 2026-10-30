"""
ICBHI 2017 Respiratory Sound Database - Acoustic Feature Engineering Pipeline.

Extracts 307 handcrafted acoustic biomarker features per respiratory cycle:
1. Mel-Frequency Cepstral Coefficients (MFCCs: 20 coeffs + Delta + Delta-Delta)
   - Aggregated via mean, standard deviation, skewness, and kurtosis across time frames (240 features).
2. Spectral Descriptors:
   - Spectral Centroid (mean, std, skew, kurtosis: 4 features)
   - Spectral Bandwidth / Spread (mean, std, skew, kurtosis: 4 features)
   - Spectral Rolloff at 85% and 95% (mean, std, skew, kurtosis: 8 features)
   - Spectral Flatness (mean, std, skew, kurtosis: 4 features)
   - Spectral Contrast across 7 octave sub-bands (mean, std: 14 features)
3. Temporal Dynamics:
   - Zero-Crossing Rate (ZCR) (mean, std, skew, kurtosis: 4 features)
   - Root-Mean-Square (RMS) Energy (mean, std, skew, kurtosis: 4 features)
   - Cycle Duration in seconds (1 feature)
4. Tonal & Pitch Biomarkers (Wheeze specific):
   - Chroma STFT across 12 pitch classes (mean, std: 24 features)

Exports:
- data/processed/features_all.csv (All 6,898 cycles with metadata and 307 features)
- data/processed/X_train.csv, X_test.csv (Official ICBHI patient-independent split)
- data/processed/y_train.csv, y_test.csv (4-class and binary targets)
- data/processed/metadata_train.csv, metadata_test.csv (Patient, hardware, and location metadata)
- data/processed/feature_names.json (Structured catalogue of feature names and families)
- High-resolution figures: fig8, fig9, fig10 in reports/figures/
"""

import os
import glob
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import soxr
    HAS_SOXR = True
except ImportError:
    HAS_SOXR = False

import librosa

# ---------------------------------------------------------------------
# Signal Processing & Filtering
# ---------------------------------------------------------------------

def design_butter_bandpass(lowcut=50.0, highcut=2000.0, fs=16000, order=4):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    return signal.butter(order, [low, high], btype='bandpass', output='sos')

def butter_bandpass_filter(data, sos):
    return signal.sosfiltfilt(sos, data)

def resample_signal(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    if HAS_SOXR:
        return soxr.resample(y, orig_sr, target_sr, quality='HQ')
    else:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type='soxr_hq')

def normalize_audio(audio, peak_target=0.95):
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * peak_target
    return audio.astype(np.float32)

# ---------------------------------------------------------------------
# Acoustic Feature Extraction for a Single Cycle
# ---------------------------------------------------------------------

def extract_cycle_features(y, sr=16000, n_mfcc=20, n_fft=1024, hop_length=512):
    """
    Extract 307 handcrafted acoustic features from a normalized respiratory cycle audio waveform.
    """
    y = y.astype(np.float32)
    # Ensure minimum sample length for STFT operations and delta filters (at least 5120 samples ~0.32s)
    min_len = max(n_fft, 5120)
    if len(y) < min_len:
        pad_width = min_len - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')

    feats = {}
    feats['cycle_duration_feat'] = float(len(y) / sr)

    # 1. Mel-Frequency Cepstral Coefficients (MFCCs) + Deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d_mfcc = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)

    for i in range(n_mfcc):
        row = np.nan_to_num(mfcc[i], nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'mfcc_{i:02d}_mean'] = float(np.mean(row))
        feats[f'mfcc_{i:02d}_std'] = float(np.std(row))
        feats[f'mfcc_{i:02d}_skew'] = float(stats.skew(row))
        feats[f'mfcc_{i:02d}_kurt'] = float(stats.kurtosis(row))

    for i in range(n_mfcc):
        row = np.nan_to_num(d_mfcc[i], nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'mfcc_delta_{i:02d}_mean'] = float(np.mean(row))
        feats[f'mfcc_delta_{i:02d}_std'] = float(np.std(row))
        feats[f'mfcc_delta_{i:02d}_skew'] = float(stats.skew(row))
        feats[f'mfcc_delta_{i:02d}_kurt'] = float(stats.kurtosis(row))

    for i in range(n_mfcc):
        row = np.nan_to_num(d2_mfcc[i], nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'mfcc_delta2_{i:02d}_mean'] = float(np.mean(row))
        feats[f'mfcc_delta2_{i:02d}_std'] = float(np.std(row))
        feats[f'mfcc_delta2_{i:02d}_skew'] = float(stats.skew(row))
        feats[f'mfcc_delta2_{i:02d}_kurt'] = float(stats.kurtosis(row))

    # Helper for 1D summary stats
    def add_summary_stats(arr, prefix):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'{prefix}_mean'] = float(np.mean(arr))
        feats[f'{prefix}_std'] = float(np.std(arr))
        feats[f'{prefix}_skew'] = float(stats.skew(arr, axis=None))
        feats[f'{prefix}_kurt'] = float(stats.kurtosis(arr, axis=None))

    # 2. Spectral Descriptors
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    ro85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop_length)
    ro95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, n_fft=n_fft, hop_length=hop_length)
    flat = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    cont = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    add_summary_stats(cent, 'spectral_centroid')
    add_summary_stats(bw, 'spectral_bandwidth')
    add_summary_stats(ro85, 'spectral_rolloff_85')
    add_summary_stats(ro95, 'spectral_rolloff_95')
    add_summary_stats(flat, 'spectral_flatness')

    # Spectral contrast has 7 octave sub-bands
    for b in range(cont.shape[0]):
        row = np.nan_to_num(cont[b], nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'spectral_contrast_b{b}_mean'] = float(np.mean(row))
        feats[f'spectral_contrast_b{b}_std'] = float(np.std(row))

    # 3. Temporal Dynamics
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    add_summary_stats(zcr, 'zcr')
    add_summary_stats(rms, 'rms_energy')

    # 4. Chroma STFT (12 semitones, fixed tuning to avoid estimator jitter)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, tuning=0.0)
    for c in range(chroma.shape[0]):
        row = np.nan_to_num(chroma[c], nan=0.0, posinf=0.0, neginf=0.0)
        feats[f'chroma_{c:02d}_mean'] = float(np.mean(row))
        feats[f'chroma_{c:02d}_std'] = float(np.std(row))

    return feats

# ---------------------------------------------------------------------
# Worker Function for Multiprocessing
# ---------------------------------------------------------------------

def process_recording_item(item):
    """
    Worker task: processes one recording file, filters audio, segments annotated cycles,
    extracts 307 features per cycle, and returns list of cycle records.
    """
    wav_path = item["wav_path"]
    txt_path = item["txt_path"]
    target_sr = item["target_sr"]
    lowcut = item["lowcut"]
    highcut = item["highcut"]
    order = item["order"]
    save_cycles_dir = item.get("save_cycles_dir", None)
    split = item["split"]
    patient_id = item["patient_id"]
    diagnosis = item["diagnosis"]
    age = item.get("age", np.nan)
    sex = item.get("sex", "NA")
    chest_loc = item["chest_loc"]
    equipment = item["equipment"]
    acq_mode = item["acq_mode"]

    fname = os.path.basename(wav_path)
    base_name = os.path.splitext(fname)[0]

    if not os.path.exists(txt_path):
        return []

    # Read annotations
    with open(txt_path, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]

    if not lines:
        return []

    try:
        raw_audio, orig_sr = sf.read(wav_path)
    except Exception as e:
        return []

    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1)

    # 1. Resample to standard target_sr (16 kHz)
    audio_resampled = resample_signal(raw_audio, orig_sr, target_sr)

    # 2. 4th-order zero-phase Butterworth bandpass filter (50–2,000 Hz)
    sos = design_butter_bandpass(lowcut, highcut, target_sr, order)
    audio_filtered = butter_bandpass_filter(audio_resampled, sos)

    # 3. Demean and peak normalize
    audio_norm = normalize_audio(audio_filtered, peak_target=0.95)

    records = []
    for cycle_idx, line in enumerate(lines):
        if len(line) < 4:
            continue
        try:
            c_start = float(line[0])
            c_end = float(line[1])
            c_crackle = int(line[2])
            c_wheeze = int(line[3])
        except ValueError:
            continue

        c_duration = c_end - c_start
        if c_duration <= 0:
            continue

        start_sample = max(0, int(round(c_start * target_sr)))
        end_sample = min(len(audio_norm), int(round(c_end * target_sr)))
        cycle_audio = audio_norm[start_sample:end_sample]

        if len(cycle_audio) == 0:
            continue

        # Class determination
        if c_crackle == 1 and c_wheeze == 1:
            class_label = "Both"
            class_id = 3
            is_adventitious = 1
        elif c_crackle == 1 and c_wheeze == 0:
            class_label = "Crackle"
            class_id = 1
            is_adventitious = 1
        elif c_crackle == 0 and c_wheeze == 1:
            class_label = "Wheeze"
            class_id = 2
            is_adventitious = 1
        else:
            class_label = "Normal"
            class_id = 0
            is_adventitious = 0

        cycle_id = f"{base_name}_c{cycle_idx:02d}_{class_label.lower()}"
        cycle_fname = f"{cycle_id}.wav"

        # Optionally save segmented wav file
        if save_cycles_dir:
            out_wav = os.path.join(save_cycles_dir, cycle_fname)
            sf.write(out_wav, cycle_audio, target_sr, subtype='PCM_16')

        # Extract handcrafted acoustic features
        feat_dict = extract_cycle_features(cycle_audio, sr=target_sr)

        record = {
            "cycle_id": cycle_id,
            "recording_name": base_name,
            "cycle_idx": cycle_idx,
            "cycle_start": c_start,
            "cycle_end": c_end,
            "cycle_duration": c_duration,
            "sample_count": len(cycle_audio),
            "crackle": c_crackle,
            "wheeze": c_wheeze,
            "class": class_label,
            "class_id": class_id,
            "is_adventitious": is_adventitious,
            "split": split,
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "age": age,
            "sex": sex,
            "chest_location": chest_loc,
            "equipment": equipment,
            "acquisition_mode": acq_mode,
        }
        # Merge features
        record.update(feat_dict)
        records.append(record)

    return records

# ---------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------

def generate_phase3_figures(df, feature_cols, figures_dir):
    """
    Generate publication-ready figures for Phase 3 (Feature Engineering):
    - fig8_acoustic_features_distribution.png: Boxplots across 4 sound classes
    - fig9_feature_correlation_matrix.png: Correlation heatmap of representative features
    - fig10_pca_tsne_feature_space.png: 2D PCA & t-SNE projections
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Figure 8: Acoustic Feature Distributions by Sound Class
    rep_features = [
        ("mfcc_00_mean", "MFCC 0 (Mean Amplitude)"),
        ("mfcc_01_mean", "MFCC 1 (Spectral Balance)"),
        ("spectral_centroid_mean", "Spectral Centroid Mean (Hz)"),
        ("spectral_rolloff_85_mean", "Spectral Rolloff 85% Mean (Hz)"),
        ("spectral_flatness_mean", "Spectral Flatness Mean"),
        ("rms_energy_mean", "RMS Energy Mean")
    ]
    # Filter features that exist
    rep_features = [(f, label) for f, label in rep_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    palette = {"Normal": "#2b8a3e", "Crackle": "#d9480f", "Wheeze": "#1971c2", "Both": "#7048e8"}

    for idx, (f_col, f_name) in enumerate(rep_features):
        ax = axes[idx]
        sns.boxplot(data=df, x="class", y=f_col, hue="class", order=["Normal", "Crackle", "Wheeze", "Both"],
                    palette=palette, ax=ax, width=0.5, fliersize=2, showcaps=True, legend=False)
        ax.set_title(f_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value")

    fig.suptitle("Acoustic Biomarker Feature Distributions across Respiratory Sound Classes (ICBHI 2017)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig8_path = os.path.join(figures_dir, "fig8_acoustic_features_distribution.png")
    fig.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 8: {fig8_path}", flush=True)

    # 2. Figure 9: Feature Correlation Matrix
    corr_candidates = [
        "cycle_duration_feat", "mfcc_00_mean", "mfcc_01_mean", "mfcc_02_mean",
        "mfcc_delta_00_mean", "mfcc_delta_01_mean",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "spectral_rolloff_85_mean", "spectral_flatness_mean",
        "spectral_contrast_b0_mean", "spectral_contrast_b3_mean",
        "zcr_mean", "rms_energy_mean", "chroma_00_mean", "chroma_05_mean"
    ]
    corr_cols = [c for c in corr_candidates if c in df.columns]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, annot=False)
    ax.set_title("Pairwise Pearson Correlation Matrix of Acoustic Biomarkers", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig9_path = os.path.join(figures_dir, "fig9_feature_correlation_matrix.png")
    fig.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 9: {fig9_path}", flush=True)

    # 3. Figure 10: 2D PCA & t-SNE Manifold Projections
    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    X_sub = df_sample[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X_sub)

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA Plot
    for cls_name, color in [("Normal", "#2b8a3e"), ("Crackle", "#d9480f"), ("Wheeze", "#1971c2"), ("Both", "#7048e8")]:
        mask_c = df_sample["class"] == cls_name
        axes[0].scatter(pca_coords[mask_c, 0], pca_coords[mask_c, 1],
                        c=color, label=f"{cls_name} (N={mask_c.sum()})", alpha=0.6, edgecolors='none', s=25)
    axes[0].set_title(f"PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend(loc="upper right", frameon=True)

    # t-SNE Plot
    for cls_name, color in [("Normal", "#2b8a3e"), ("Crackle", "#d9480f"), ("Wheeze", "#1971c2"), ("Both", "#7048e8")]:
        mask_c = df_sample["class"] == cls_name
        axes[1].scatter(tsne_coords[mask_c, 0], tsne_coords[mask_c, 1],
                        c=color, label=f"{cls_name}", alpha=0.6, edgecolors='none', s=25)
    axes[1].set_title("t-SNE Manifold Projection (Perplexity=30)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle("Unsupervised Manifold Representation of Acoustic Feature Space (ICBHI 2017)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig10_path = os.path.join(figures_dir, "fig10_pca_tsne_feature_space.png")
    fig.savefig(fig10_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 10: {fig10_path}")

# ---------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ICBHI Acoustic Feature Extraction Pipeline")
    parser.add_argument("--sr", type=int, default=16000, help="Standard sampling rate (default: 16000 Hz)")
    parser.add_argument("--lowcut", type=float, default=50.0, help="Bandpass low cutoff (default: 50.0 Hz)")
    parser.add_argument("--highcut", type=float, default=2000.0, help="Bandpass high cutoff (default: 2000.0 Hz)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order (default: 4)")
    parser.add_argument("--n_mfcc", type=int, default=20, help="Number of MFCC coefficients (default: 20)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--save_cycles_wav", action="store_true", help="Save segmented cycle .wav files to disk")
    args = parser.parse_args()

    t_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Resolve paths
    candidate_raw_dirs = [
        os.path.join(project_root, "ICBHI_final_database"),
        os.path.join(project_root, "..", "ICBHI_final_database", "ICBHI_final_database"),
        os.path.join(project_root, "..", "ICBHI_final_database"),
    ]
    raw_dir = next((d for d in candidate_raw_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.wav"))) > 0), None)
    if raw_dir is None:
        raise FileNotFoundError(f"Could not locate ICBHI raw database in: {candidate_raw_dirs}")

    metadata_dir = os.path.join(project_root, "data", "metadata")
    processed_dir = os.path.join(project_root, "data", "processed")
    figures_dir = os.path.join(project_root, "reports", "figures")
    cycles_dir = os.path.join(processed_dir, "segmented_cycles")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    if args.save_cycles_wav:
        os.makedirs(cycles_dir, exist_ok=True)

    print("=" * 75)
    print("      ICBHI 2017 ACOUSTIC FEATURE ENGINEERING & BENCHMARK PIPELINE")
    print("=" * 75)
    print(f"Project Root       : {project_root}")
    print(f"Raw Audio Dir      : {raw_dir}")
    print(f"Output Tables Dir  : {processed_dir}")
    print(f"Figures Dir        : {figures_dir}")
    print(f"Sampling Rate      : {args.sr} Hz")
    print(f"Filter Band        : {args.lowcut:.1f} - {args.highcut:.1f} Hz (Order {args.order} Butterworth SOS)")
    print(f"MFCC Coefficients  : {args.n_mfcc} (+ Delta + Delta-Delta)")
    print(f"Parallel Workers   : {args.workers}")
    print(f"Save Cycles WAV    : {args.save_cycles_wav}")
    print("=" * 75)

    # 1. Load metadata files
    print("\n[1/5] Loading metadata mappings...")
    demo_file = os.path.join(metadata_dir, "ICBHI_Challenge_demographic_information.txt")
    demo_df = pd.read_csv(demo_file, sep=r'\s+', header=None,
                          names=["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"],
                          na_values=['NA'])

    diag_file = os.path.join(metadata_dir, "ICBHI_Challenge_diagnosis.txt")
    diag_df = pd.read_csv(diag_file, sep=r'\s+', header=None, names=["patient_id", "diagnosis"])

    split_file = os.path.join(metadata_dir, "ICBHI_challenge_train_test.txt")
    split_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=["recording_name", "split"])
    split_map = dict(zip(split_df["recording_name"], split_df["split"]))

    patients_df = pd.merge(demo_df, diag_df, on="patient_id", how="outer")
    patient_meta_map = patients_df.set_index("patient_id").to_dict(orient="index")

    chest_loc_map = {
        'Tc': 'Trachea', 'Al': 'Anterior Left', 'Ar': 'Anterior Right',
        'Pl': 'Posterior Left', 'Pr': 'Posterior Right',
        'Ll': 'Lateral Left', 'Lr': 'Lateral Right'
    }
    mode_map = {'sc': 'Single-Channel', 'mc': 'Multi-Channel'}

    # 2. Build task list for all 920 recordings
    print("\n[2/5] Preparing work tasks for 920 recordings...")
    wav_files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    work_items = []

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        base_name = os.path.splitext(fname)[0]
        parts = base_name.split("_")
        if len(parts) != 5:
            continue

        patient_id = int(parts[0])
        rec_idx = parts[1]
        chest_code = parts[2]
        acq_code = parts[3]
        equipment = parts[4]

        # Official train/test split resolution
        split = split_map.get(base_name, "unknown")
        # In ICBHI challenge, patient 226 belongs to train cohort
        if split == "unknown" and patient_id == 226:
            split = "train"

        p_info = patient_meta_map.get(patient_id, {})
        diagnosis = p_info.get("diagnosis", "Unknown")
        age = p_info.get("age", np.nan)
        sex = p_info.get("sex", "NA")

        txt_path = os.path.join(raw_dir, f"{base_name}.txt")

        work_items.append({
            "wav_path": wav_path,
            "txt_path": txt_path,
            "target_sr": args.sr,
            "lowcut": args.lowcut,
            "highcut": args.highcut,
            "order": args.order,
            "save_cycles_dir": cycles_dir if args.save_cycles_wav else None,
            "split": split,
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "age": age,
            "sex": sex,
            "chest_loc": chest_loc_map.get(chest_code, chest_code),
            "equipment": equipment,
            "acq_mode": mode_map.get(acq_code, acq_code)
        })

    print(f"Total valid recordings queued: {len(work_items)}")

    # 3. Parallel Feature Extraction
    print(f"\n[3/5] Extracting 307 acoustic features across {len(work_items)} recordings (workers={args.workers})...")
    all_cycles = []
    t_feat_start = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {executor.submit(process_recording_item, item): idx for idx, item in enumerate(work_items)}
            completed = 0
            for future in as_completed(future_to_idx):
                completed += 1
                try:
                    res = future.result()
                    all_cycles.extend(res)
                except Exception as exc:
                    print(f"Error processing item: {exc}")

                if completed % 100 == 0 or completed == len(work_items):
                    elapsed = time.time() - t_feat_start
                    pct = (completed / len(work_items)) * 100
                    print(f"  Processed {completed:3d}/{len(work_items)} recordings ({pct:5.1f}%) | "
                          f"Cycles extracted: {len(all_cycles):5,d} | Elapsed: {elapsed:5.1f}s")
    else:
        for idx, item in enumerate(work_items, 1):
            res = process_recording_item(item)
            all_cycles.extend(res)
            if idx % 100 == 0 or idx == len(work_items):
                elapsed = time.time() - t_feat_start
                print(f"  Processed {idx:3d}/{len(work_items)} recordings | "
                      f"Cycles: {len(all_cycles):5,d} | Elapsed: {elapsed:5.1f}s")

    print(f"\nFeature extraction complete! Total cycles: {len(all_cycles):,d} in {time.time() - t_feat_start:.2f}s")

    # 4. Construct DataFrames and Save Structured Benchmark Splits
    print("\n[4/5] Constructing feature tables and structuring train/test benchmark splits...")
    df_all = pd.DataFrame(all_cycles)

    # Reorder columns: metadata first, then features
    metadata_cols = [
        "cycle_id", "recording_name", "cycle_idx", "cycle_start", "cycle_end",
        "cycle_duration", "sample_count", "crackle", "wheeze", "class", "class_id",
        "is_adventitious", "split", "patient_id", "diagnosis", "age", "sex",
        "chest_location", "equipment", "acquisition_mode"
    ]
    feature_cols = [col for col in df_all.columns if col not in metadata_cols]
    ordered_cols = metadata_cols + feature_cols
    df_all = df_all[ordered_cols]

    # Save full master feature table
    full_table_path = os.path.join(processed_dir, "features_all.csv")
    df_all.to_csv(full_table_path, index=False)
    print(f"Saved Master Feature Table : {full_table_path} ({df_all.shape[0]} rows x {df_all.shape[1]} cols)")

    # Split into train and test cohorts
    train_mask = df_all["split"] == "train"
    test_mask = df_all["split"] == "test"

    df_train = df_all[train_mask].reset_index(drop=True)
    df_test = df_all[test_mask].reset_index(drop=True)

    print(f"Train Cohort Cycles : {len(df_train):,d} (Patients: {df_train['patient_id'].nunique()})")
    print(f"Test Cohort Cycles  : {len(df_test):,d} (Patients: {df_test['patient_id'].nunique()})")

    # Save X and y
    X_train = df_train[feature_cols]
    y_train = df_train[["class_id", "class", "is_adventitious"]]
    meta_train = df_train[metadata_cols]

    X_test = df_test[feature_cols]
    y_test = df_test[["class_id", "class", "is_adventitious"]]
    meta_test = df_test[metadata_cols]

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    meta_train.to_csv(os.path.join(processed_dir, "metadata_train.csv"), index=False)

    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    meta_test.to_csv(os.path.join(processed_dir, "metadata_test.csv"), index=False)

    # Save feature names catalogue
    feature_catalogue = {
        "total_features": len(feature_cols),
        "feature_families": {
            "mfcc_stats": [c for c in feature_cols if c.startswith("mfcc_") and "delta" not in c],
            "mfcc_delta_stats": [c for c in feature_cols if c.startswith("mfcc_delta_")],
            "mfcc_delta2_stats": [c for c in feature_cols if c.startswith("mfcc_delta2_")],
            "spectral_descriptors": [c for c in feature_cols if c.startswith("spectral_")],
            "temporal_dynamics": [c for c in feature_cols if c.startswith("zcr") or c.startswith("rms") or c == "cycle_duration_feat"],
            "chroma_tonality": [c for c in feature_cols if c.startswith("chroma_")]
        },
        "all_feature_names": feature_cols
    }
    with open(os.path.join(processed_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_catalogue, f, indent=2)

    print(f"Saved X_train, y_train, metadata_train ({X_train.shape[0]} x {X_train.shape[1]})")
    print(f"Saved X_test, y_test, metadata_test   ({X_test.shape[0]} x {X_test.shape[1]})")
    print(f"Saved feature_names.json ({len(feature_cols)} features categorized)")

    # 5. Generate Phase 3 Verification Figures
    print("\n[5/5] Generating publication-ready feature engineering figures...")
    generate_phase3_figures(df_all, feature_cols, figures_dir)

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"  PHASE 3 FEATURE ENGINEERING COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 75)

if __name__ == "__main__":
    main()
