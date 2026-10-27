"""
ICBHI 2017 Respiratory Sound Database - Preprocessing & Segmentation Pipeline.

Performs:
1. Signal Resampling to standardized rate (default: 16 kHz).
2. Butterworth Bandpass Filtering (50 Hz - 2,000 Hz, zero-phase SOS) to eliminate
   heart sounds (<50 Hz) and high-frequency transducer noise (>2,000 Hz).
3. DC-bias removal and peak normalization.
4. Respiratory Cycle Segmentation based on official expert annotations.
5. Storage of clean cycle audio files (.wav) and master metadata CSV.
6. Generation of publication-ready verification figures.
"""

import os
import glob
import argparse
import time
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import soxr
    HAS_SOXR = True
except ImportError:
    import librosa
    HAS_SOXR = False

# -------------------------------------------------------------
# Configuration & Filtering Functions
# -------------------------------------------------------------

def design_butter_bandpass(lowcut=50.0, highcut=2000.0, fs=16000, order=4):
    """
    Design a digital Butterworth bandpass filter using Second-Order Sections (SOS).
    Using SOS prevents numerical instability at high orders or low frequencies.
    """
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    # Ensure high cutoff is strictly below Nyquist
    high = min(0.999, highcut / nyq)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    return sos

def butter_bandpass_filter(data, sos):
    """
    Apply zero-phase digital filtering to avoid phase distortion.
    """
    return signal.sosfiltfilt(sos, data)

def resample_signal(y, orig_sr, target_sr):
    """
    High-quality resampling using soxr (C library) or librosa fallback.
    """
    if orig_sr == target_sr:
        return y
    if HAS_SOXR:
        return soxr.resample(y, orig_sr, target_sr, quality='HQ')
    else:
        import librosa
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type='soxr_hq')

def normalize_audio(audio, peak_target=0.95):
    """
    Remove DC offset and normalize amplitude to peak_target to prevent clipping.
    """
    # Demean to eliminate DC bias
    audio = audio - np.mean(audio)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * peak_target
    return audio.astype(np.float32)

# -------------------------------------------------------------
# Main Preprocessing & Segmentation Pipeline
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ICBHI Audio Preprocessing & Segmentation Pipeline")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate (default: 16000 Hz)")
    parser.add_argument("--lowcut", type=float, default=50.0, help="Highpass cutoff frequency (default: 50.0 Hz)")
    parser.add_argument("--highcut", type=float, default=2000.0, help="Lowpass cutoff frequency (default: 2000.0 Hz)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order (default: 4)")
    parser.add_argument("--output_cycles_dir", type=str, default=None, help="Directory to save segmented cycle .wav files")
    parser.add_argument("--no_save_wav", action="store_true", help="Skip writing individual cycle .wav files to disk")
    args = parser.parse_args()

    start_time_all = time.time()
    
    # Base paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Locate raw audio database
    candidate_raw_dirs = [
        os.path.join(project_root, "ICBHI_final_database"),
        os.path.join(project_root, "..", "ICBHI_final_database", "ICBHI_final_database"),
        os.path.join(project_root, "..", "ICBHI_final_database"),
    ]
    raw_data_dir = next((d for d in candidate_raw_dirs if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*.wav"))) > 0), None)
    if raw_data_dir is None:
        raise FileNotFoundError(f"Could not locate ICBHI_final_database in candidates: {candidate_raw_dirs}")
    
    metadata_dir = os.path.join(project_root, "data", "metadata")
    processed_dir = os.path.join(project_root, "data", "processed")
    figures_dir = os.path.join(project_root, "reports", "figures")
    
    if args.output_cycles_dir is None:
        cycles_dir = os.path.join(processed_dir, "segmented_cycles")
    else:
        cycles_dir = os.path.abspath(args.output_cycles_dir)
        
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    if not args.no_save_wav:
        os.makedirs(cycles_dir, exist_ok=True)

    print("=" * 75)
    print("      ICBHI 2017 RESPIRATORY PREPROCESSING & SEGMENTATION PIPELINE")
    print("=" * 75)
    print(f"Project Root       : {project_root}")
    print(f"Raw Data Directory : {raw_data_dir}")
    print(f"Cycles Output Dir  : {cycles_dir}")
    print(f"Target Sample Rate : {args.sr} Hz")
    print(f"Bandpass Filter    : {args.lowcut:.1f} Hz - {args.highcut:.1f} Hz (Order {args.order} Butterworth SOS)")
    print("=" * 75)

    # 1. Load patient metadata & official splits
    print("\n[1/5] Loading metadata mappings...")
    demo_file = os.path.join(metadata_dir, "ICBHI_Challenge_demographic_information.txt")
    demo_cols = ["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"]
    demo_df = pd.read_csv(demo_file, sep=r'\s+', header=None, names=demo_cols, na_values=['NA'])
    
    diag_file = os.path.join(metadata_dir, "ICBHI_Challenge_diagnosis.txt")
    diag_df = pd.read_csv(diag_file, sep=r'\s+', header=None, names=["patient_id", "diagnosis"])
    
    split_file = os.path.join(metadata_dir, "ICBHI_challenge_train_test.txt")
    split_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=["recording_name", "split"])
    split_map = dict(zip(split_df["recording_name"], split_df["split"]))
    
    patients_df = pd.merge(demo_df, diag_df, on="patient_id", how="outer")
    patient_meta_map = patients_df.set_index("patient_id").to_dict(orient="index")

    chest_loc_map = {
        'Tc': 'Trachea',
        'Al': 'Anterior Left',
        'Ar': 'Anterior Right',
        'Pl': 'Posterior Left',
        'Pr': 'Posterior Right',
        'Ll': 'Lateral Left',
        'Lr': 'Lateral Right'
    }
    mode_map = {'sc': 'Single-Channel', 'mc': 'Multi-Channel'}

    # 2. Design the Bandpass Filter
    sos = design_butter_bandpass(args.lowcut, args.highcut, args.sr, args.order)

    # 3. Process all recordings & segment cycles
    wav_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.wav")))
    n_recordings = len(wav_files)
    print(f"\n[2/5] Processing {n_recordings} recordings & extracting annotated cycles...")

    segmented_records = []
    sample_for_plot = None  # To capture one clean example for visualization

    t_start_loop = time.time()
    for idx, wav_path in enumerate(wav_files, 1):
        fname = os.path.basename(wav_path)
        base_name = os.path.splitext(fname)[0]
        
        parts = base_name.split("_")
        if len(parts) != 5:
            continue
            
        patient_id = int(parts[0])
        rec_idx = parts[1]
        chest_code = parts[2]
        mode_code = parts[3]
        equipment = parts[4]
        
        split = split_map.get(base_name, "unknown")
        pat_info = patient_meta_map.get(patient_id, {})
        diagnosis = pat_info.get("diagnosis", "Unknown")
        age = pat_info.get("age", np.nan)
        sex = pat_info.get("sex", "Unknown")
        
        # Read raw audio
        raw_audio, orig_sr = sf.read(wav_path)
        
        # Ensure single-channel mono
        if raw_audio.ndim > 1:
            raw_audio = np.mean(raw_audio, axis=1)
            
        # Resample to target_sr
        audio_resampled = resample_signal(raw_audio, orig_sr, args.sr)
        
        # Apply Zero-Phase Butterworth Bandpass Filter
        audio_filtered = butter_bandpass_filter(audio_resampled, sos)
        
        # Normalize
        audio_norm = normalize_audio(audio_filtered, peak_target=0.95)
        
        # Parse annotation file for cycle timings
        txt_path = os.path.join(raw_data_dir, f"{base_name}.txt")
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, 'r') as f:
            lines = [l.strip().split() for l in f if l.strip()]
            
        # Check if we should save this recording as our visualization exemplar
        if sample_for_plot is None and equipment == "Meditron" and len(lines) >= 6:
            # Check if this file has both crackles and wheezes for visual richness
            has_both = any(len(l) >= 4 and int(l[2]) == 1 and int(l[3]) == 1 for l in lines)
            if has_both or idx == 10:
                sample_for_plot = {
                    "raw_audio": raw_audio,
                    "orig_sr": orig_sr,
                    "filtered_audio": audio_norm,
                    "target_sr": args.sr,
                    "base_name": base_name,
                    "diagnosis": diagnosis,
                    "equipment": equipment,
                    "chest_location": chest_loc_map.get(chest_code, chest_code),
                    "annotation_lines": lines
                }

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
                
            # Compute sample boundaries
            start_sample = max(0, int(round(c_start * args.sr)))
            end_sample = min(len(audio_norm), int(round(c_end * args.sr)))
            
            cycle_audio = audio_norm[start_sample:end_sample]
            if len(cycle_audio) == 0:
                continue
                
            # Class label
            if c_crackle == 1 and c_wheeze == 1:
                class_label = "Both"
                class_id = 3
            elif c_crackle == 1 and c_wheeze == 0:
                class_label = "Crackle"
                class_id = 1
            elif c_crackle == 0 and c_wheeze == 1:
                class_label = "Wheeze"
                class_id = 2
            else:
                class_label = "Normal"
                class_id = 0
                
            cycle_id = f"{base_name}_c{cycle_idx:02d}_{class_label.lower()}"
            cycle_fname = f"{cycle_id}.wav"
            cycle_rel_path = os.path.join("segmented_cycles", cycle_fname)
            cycle_full_path = os.path.join(cycles_dir, cycle_fname)
            
            # Save segmented cycle audio
            if not args.no_save_wav:
                sf.write(cycle_full_path, cycle_audio, args.sr, subtype='PCM_16')
                
            segmented_records.append({
                "cycle_id": cycle_id,
                "recording_name": base_name,
                "cycle_idx": cycle_idx,
                "cycle_file": cycle_fname,
                "cycle_rel_path": cycle_rel_path,
                "cycle_start": c_start,
                "cycle_end": c_end,
                "cycle_duration": c_duration,
                "sample_count": len(cycle_audio),
                "sample_rate": args.sr,
                "crackle": c_crackle,
                "wheeze": c_wheeze,
                "class": class_label,
                "class_id": class_id,
                "split": split,
                "patient_id": patient_id,
                "diagnosis": diagnosis,
                "age": age,
                "sex": sex,
                "chest_location_code": chest_code,
                "chest_location": chest_loc_map.get(chest_code, chest_code),
                "acquisition_mode": mode_map.get(mode_code, mode_code),
                "equipment": equipment
            })
            
        if idx % 100 == 0 or idx == n_recordings:
            elapsed = time.time() - t_start_loop
            print(f"  Processed {idx:3d}/{n_recordings} files ({len(segmented_records):5d} cycles segmented) - {elapsed:.1f}s elapsed")

    cycles_df = pd.DataFrame(segmented_records)
    
    # Save master metadata table
    print("\n[3/5] Saving master metadata table...")
    metadata_out_path = os.path.join(processed_dir, "segmented_cycles_metadata.csv")
    cycles_df.to_csv(metadata_out_path, index=False)
    print(f"Saved cycle metadata table ({len(cycles_df)} rows) to:\n  -> {metadata_out_path}")

    # 4. Generate Preprocessing & Segmentation Figure
    print("\n[4/5] Generating publication-quality verification figure...")
    if sample_for_plot is None:
        # Fallback to first recording
        raw_audio, orig_sr = sf.read(wav_files[0])
        sample_for_plot = {
            "raw_audio": raw_audio if raw_audio.ndim == 1 else raw_audio[:, 0],
            "orig_sr": orig_sr,
            "filtered_audio": normalize_audio(butter_bandpass_filter(resample_signal(raw_audio, orig_sr, args.sr), sos)),
            "target_sr": args.sr,
            "base_name": os.path.splitext(os.path.basename(wav_files[0]))[0],
            "diagnosis": "COPD",
            "equipment": "Meditron",
            "chest_location": "Trachea",
            "annotation_lines": []
        }

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25)

    raw_y = sample_for_plot["raw_audio"]
    raw_sr = sample_for_plot["orig_sr"]
    filt_y = sample_for_plot["filtered_audio"]
    filt_sr = sample_for_plot["target_sr"]
    rec_title = f"{sample_for_plot['base_name']} ({sample_for_plot['diagnosis']}, {sample_for_plot['equipment']}, {sample_for_plot['chest_location']})"

    # Window of 5 seconds for waveform comparison
    t_max = min(5.0, len(raw_y) / raw_sr, len(filt_y) / filt_sr)
    raw_sub = raw_y[:int(t_max * raw_sr)]
    filt_sub = filt_y[:int(t_max * filt_sr)]
    t_raw = np.linspace(0, len(raw_sub)/raw_sr, len(raw_sub))
    t_filt = np.linspace(0, len(filt_sub)/filt_sr, len(filt_sub))

    # Subplot 1: Raw Waveform
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_raw, raw_sub, color='#4575b4', lw=0.8, alpha=0.9)
    ax1.set_title(f"A1: Raw Audio Signal [Original SR: {raw_sr} Hz] (First 5s)\n{rec_title}", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Raw Amplitude", fontsize=10)
    ax1.set_xlabel("Time (seconds)", fontsize=10)

    # Subplot 2: Filtered & Normalized Waveform
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_filt, filt_sub, color='#d73027', lw=0.8, alpha=0.9)
    ax2.set_title(f"A2: Preprocessed Signal [Resampled to {filt_sr} Hz + Bandpass 50–2000 Hz + Peak Norm]\n{rec_title}", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Normalized Amplitude", fontsize=10)
    ax2.set_xlabel("Time (seconds)", fontsize=10)

    # Subplot 3: Power Spectral Density (PSD) Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    f_raw, psd_raw = signal.welch(raw_y, raw_sr, nperseg=min(2048, len(raw_y)))
    f_filt, psd_filt = signal.welch(filt_y, filt_sr, nperseg=min(2048, len(filt_y)))
    
    ax3.semilogy(f_raw[f_raw <= 4000], psd_raw[f_raw <= 4000], label=f'Raw Signal ({raw_sr} Hz)', color='#4575b4', lw=1.2, alpha=0.7)
    ax3.semilogy(f_filt[f_filt <= 4000], psd_filt[f_filt <= 4000], label=f'Bandpassed ({args.lowcut:.0f}–{args.highcut:.0f} Hz)', color='#d73027', lw=1.5)
    ax3.axvspan(0, args.lowcut, color='gray', alpha=0.2, label='Low Attenuation (<50 Hz Heart/Motion)')
    ax3.axvspan(args.highcut, 4000, color='gray', alpha=0.15, label='High Attenuation (>2000 Hz Transducer Noise)')
    ax3.set_title("B: Power Spectral Density (PSD) - Noise Rejection Verification", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel("Power Spectral Density (V²/Hz)", fontsize=10)
    ax3.set_xlim(0, 3000)
    ax3.legend(loc='upper right', fontsize=8.5)

    # Subplot 4: Filter Magnitude & Phase Frequency Response (Bode Plot)
    ax4 = fig.add_subplot(gs[1, 1])
    w, h = signal.sosfreqz(sos, worN=2048, fs=filt_sr)
    ax4.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-5)), color='#2b8a3e', lw=2)
    ax4.axvline(args.lowcut, color='black', linestyle='--', alpha=0.6, label=f'Low Cutoff: {args.lowcut:.0f} Hz')
    ax4.axvline(args.highcut, color='black', linestyle=':', alpha=0.6, label=f'High Cutoff: {args.highcut:.0f} Hz')
    ax4.axhline(-3, color='red', linestyle='--', alpha=0.5, label='-3 dB Half-Power Points')
    ax4.set_title(f"C: Butterworth Bandpass Filter Frequency Response (Order {args.order} SOS)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Frequency (Hz)", fontsize=10)
    ax4.set_ylabel("Magnitude (dB)", fontsize=10)
    ax4.set_xlim(0, 3500)
    ax4.set_ylim(-60, 5)
    ax4.legend(loc='lower right', fontsize=8.5)

    # Subplot 5: Full Audio Recording with Segmented Cycle Annotations
    ax5 = fig.add_subplot(gs[2, :])
    t_full = np.linspace(0, len(filt_y)/filt_sr, len(filt_y))
    ax5.plot(t_full, filt_y, color='#555555', lw=0.6, alpha=0.65, label='Preprocessed Audio')
    
    color_map = {
        'Normal': '#2b8a3e',
        'Crackle': '#e67700',
        'Wheeze': '#d6336c',
        'Both': '#7048e8'
    }

    # Overlay annotated cycle spans
    shown_classes = set()
    for line in sample_for_plot["annotation_lines"]:
        if len(line) < 4:
            continue
        try:
            cs = float(line[0])
            ce = float(line[1])
            ck = int(line[2])
            wz = int(line[3])
        except ValueError:
            continue
            
        if ck == 1 and wz == 1:
            lbl = "Both"
        elif ck == 1:
            lbl = "Crackle"
        elif wz == 1:
            lbl = "Wheeze"
        else:
            lbl = "Normal"
            
        color = color_map[lbl]
        first_time = lbl not in shown_classes
        ax5.axvspan(cs, ce, color=color, alpha=0.25, label=f'Cycle: {lbl}' if first_time else None)
        shown_classes.add(lbl)
        
        # Add label tag at cycle center
        c_mid = (cs + ce) / 2.0
        ax5.text(c_mid, 0.85, lbl, ha='center', va='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7, edgecolor='none', textcolor='white') if hasattr(dict, 'dummy') else dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color))

    ax5.set_title(f"D: Complete Recording Auscultation with Annotated Respiratory Cycle Segments\nRecording: {sample_for_plot['base_name']} | Patient Diagnosis: {sample_for_plot['diagnosis']}", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Recording Time (seconds)", fontsize=10)
    ax5.set_ylabel("Normalized Amplitude", fontsize=10)
    ax5.set_xlim(0, len(filt_y)/filt_sr)
    ax5.set_ylim(-1.05, 1.05)
    ax5.legend(loc='lower right', fontsize=9)

    fig7_path = os.path.join(figures_dir, "fig7_preprocessing_and_segmentation.png")
    plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved publication verification figure to:\n  -> {fig7_path}")

    # 5. Print Summary Statistics
    total_time = time.time() - start_time_all
    print("\n" + "=" * 75)
    print("               PREPROCESSING & SEGMENTATION COMPLETED")
    print("=" * 75)
    print(f"Total Audio Recordings Processed : {n_recordings}")
    print(f"Total Respiratory Cycles Saved   : {len(cycles_df)}")
    print(f"Pipeline Execution Time          : {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nSegmented Cycles Class Distribution:")
    cls_counts = cycles_df["class"].value_counts()
    for cls_name, count in cls_counts.items():
        pct = (count / len(cycles_df)) * 100
        print(f"  - {cls_name:8s}: {count:5d} ({pct:5.2f}%)")

    print("\nBenchmark Split Breakdown (Cycles):")
    split_counts = cycles_df["split"].value_counts()
    for sp_name, count in split_counts.items():
        pct = (count / len(cycles_df)) * 100
        print(f"  - {sp_name.capitalize():6s} Set : {count:5d} ({pct:5.2f}%)")

    print("\nCycle Duration Statistics:")
    print(f"  - Mean Duration   : {cycles_df['cycle_duration'].mean():.2f}s (± {cycles_df['cycle_duration'].std():.2f}s)")
    print(f"  - Median Duration : {cycles_df['cycle_duration'].median():.2f}s")
    print(f"  - Min / Max       : {cycles_df['cycle_duration'].min():.2f}s / {cycles_df['cycle_duration'].max():.2f}s")
    
    if not args.no_save_wav:
        disk_files = glob.glob(os.path.join(cycles_dir, "*.wav"))
        total_size_mb = sum(os.path.getsize(f) for f in disk_files) / (1024 * 1024)
        print(f"\nStorage on Disk:")
        print(f"  - Saved WAV Files : {len(disk_files)} in {cycles_dir}")
        print(f"  - Total Size      : {total_size_mb:.2f} MB")
    print("=" * 75)

if __name__ == "__main__":
    main()
