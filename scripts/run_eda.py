"""
Comprehensive Exploratory Data Analysis (EDA) for ICBHI 2017 Respiratory Sound Database.
Extracts patient demographics, diagnosis, recording metadata, respiratory cycles,
adventitious sound distributions, and generates publication-quality figures.
"""

import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Set style for publication-ready figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "ICBHI_final_database", "ICBHI_final_database"))
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data dir: {RAW_DATA_DIR}")
print(f"Metadata dir: {METADATA_DIR}")

# -------------------------------------------------------------
# 1. LOAD OFFICIAL METADATA
# -------------------------------------------------------------
print("\n[1/6] Loading metadata files...")

# Demographics: Participant ID, Age, Sex, Adult BMI (kg/m2), Child Weight (kg), Child Height (cm)
demo_file = os.path.join(METADATA_DIR, "ICBHI_Challenge_demographic_information.txt")
demo_cols = ["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"]
demographics_df = pd.read_csv(demo_file, sep=r'\s+', header=None, names=demo_cols, na_values=['NA'])

# Diagnosis: Patient ID, Diagnosis
diag_file = os.path.join(METADATA_DIR, "ICBHI_Challenge_diagnosis.txt")
diagnosis_df = pd.read_csv(diag_file, sep=r'\s+', header=None, names=["patient_id", "diagnosis"])

# Train/Test Split: Recording Name, Split (train/test)
split_file = os.path.join(METADATA_DIR, "ICBHI_challenge_train_test.txt")
train_test_df = pd.read_csv(split_file, sep=r'\s+', header=None, names=["recording_name", "split"])

# Merge Patient Metadata
patients_df = pd.merge(demographics_df, diagnosis_df, on="patient_id", how="outer")
print(f"Loaded {len(patients_df)} patient demographic & diagnosis records.")

# -------------------------------------------------------------
# 2. PARSE RECORDING FILES & AUDIO CHARACTERISTICS
# -------------------------------------------------------------
print("\n[2/6] Parsing 920 audio recordings and annotations...")

wav_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.wav")))
print(f"Found {len(wav_files)} .wav files in raw data directory.")

recordings_list = []
cycles_list = []

chest_loc_map = {
    'Tc': 'Trachea',
    'Al': 'Anterior Left',
    'Ar': 'Anterior Right',
    'Pl': 'Posterior Left',
    'Pr': 'Posterior Right',
    'Ll': 'Lateral Left',
    'Lr': 'Lateral Right'
}

mode_map = {
    'sc': 'Single-Channel',
    'mc': 'Multi-Channel'
}

# Pre-map train/test split
split_map = dict(zip(train_test_df["recording_name"], train_test_df["split"]))

for wav_path in wav_files:
    fname = os.path.basename(wav_path)
    base_name = os.path.splitext(fname)[0]
    
    parts = base_name.split("_")
    if len(parts) != 5:
        continue
        
    patient_id = int(parts[0])
    rec_idx = parts[1]
    chest_loc = parts[2]
    acq_mode = parts[3]
    equipment = parts[4]
    
    # Read audio metadata quickly using soundfile.info
    info = sf.info(wav_path)
    sr = info.samplerate
    duration = info.duration
    channels = info.channels
    frames = info.frames
    subtype = info.subtype
    
    split = split_map.get(base_name, "unknown")
    
    # Parse corresponding annotation file
    txt_path = os.path.join(RAW_DATA_DIR, f"{base_name}.txt")
    n_cycles = 0
    n_crackles = 0
    n_wheezes = 0
    n_both = 0
    n_normal = 0
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = [l.strip().split() for l in f if l.strip()]
            
        for line in lines:
            if len(line) >= 4:
                try:
                    c_start = float(line[0])
                    c_end = float(line[1])
                    c_crackle = int(line[2])
                    c_wheeze = int(line[3])
                except ValueError:
                    continue
                    
                c_dur = c_end - c_start
                n_cycles += 1
                
                if c_crackle == 1 and c_wheeze == 1:
                    c_class = "Both"
                    n_both += 1
                elif c_crackle == 1 and c_wheeze == 0:
                    c_class = "Crackle"
                    n_crackles += 1
                elif c_crackle == 0 and c_wheeze == 1:
                    c_class = "Wheeze"
                    n_wheezes += 1
                else:
                    c_class = "Normal"
                    n_normal += 1
                    
                cycles_list.append({
                    "recording_name": base_name,
                    "patient_id": patient_id,
                    "cycle_start": c_start,
                    "cycle_end": c_end,
                    "cycle_duration": c_dur,
                    "crackle": c_crackle,
                    "wheeze": c_wheeze,
                    "class": c_class,
                    "split": split,
                    "chest_location": chest_loc_map.get(chest_loc, chest_loc),
                    "equipment": equipment
                })
                
    recordings_list.append({
        "recording_name": base_name,
        "patient_id": patient_id,
        "rec_idx": rec_idx,
        "chest_location_code": chest_loc,
        "chest_location": chest_loc_map.get(chest_loc, chest_loc),
        "acq_mode_code": acq_mode,
        "acq_mode": mode_map.get(acq_mode, acq_mode),
        "equipment": equipment,
        "sample_rate": sr,
        "duration_sec": duration,
        "channels": channels,
        "frames": frames,
        "subtype": subtype,
        "split": split,
        "num_cycles": n_cycles,
        "normal_cycles": n_normal,
        "crackle_cycles": n_crackles,
        "wheeze_cycles": n_wheezes,
        "both_cycles": n_both
    })

recordings_df = pd.DataFrame(recordings_list)
cycles_df = pd.DataFrame(cycles_list)

# Merge patient diagnosis into recordings and cycles
recordings_df = pd.merge(recordings_df, patients_df[["patient_id", "diagnosis", "age", "sex"]], on="patient_id", how="left")
cycles_df = pd.merge(cycles_df, patients_df[["patient_id", "diagnosis", "age", "sex"]], on="patient_id", how="left")

# Save processed CSVs
patients_df.to_csv(os.path.join(PROCESSED_DIR, "patients_summary.csv"), index=False)
recordings_df.to_csv(os.path.join(PROCESSED_DIR, "recordings_summary.csv"), index=False)
cycles_df.to_csv(os.path.join(PROCESSED_DIR, "cycles_summary.csv"), index=False)

print(f"Processed {len(recordings_df)} recordings and {len(cycles_df)} respiratory cycles.")
print(f"Saved CSV summaries to: {PROCESSED_DIR}")

# -------------------------------------------------------------
# 3. PRINT COMPREHENSIVE DATASET STATISTICS
# -------------------------------------------------------------
print("\n" + "="*70)
print("             ICBHI 2017 DATASET EXPLORATORY STATISTICS")
print("="*70)

print("\n--- 1. PATIENT DEMOGRAPHICS & DIAGNOSIS ---")
print(f"Total Unique Patients: {len(patients_df)}")
print("\nDiagnosis Distribution (Patients):")
diag_counts = patients_df["diagnosis"].value_counts()
for diag, cnt in diag_counts.items():
    pct = (cnt / len(patients_df)) * 100
    print(f"  {diag:16s}: {cnt:3d} ({pct:5.2f}%)")

print("\nGender Distribution:")
sex_counts = patients_df["sex"].value_counts()
for s, cnt in sex_counts.items():
    pct = (cnt / len(patients_df)) * 100
    print(f"  {s:16s}: {cnt:3d} ({pct:5.2f}%)")

adults = patients_df[patients_df["age"] >= 18]
pediatric = patients_df[patients_df["age"] < 18]
print(f"\nAge Summary:")
print(f"  Overall Age Mean ± SD: {patients_df['age'].mean():.2f} ± {patients_df['age'].std():.2f} (Range: {patients_df['age'].min()} - {patients_df['age'].max()})")
print(f"  Adults (>=18 yrs)     : {len(adults)} patients (Mean age: {adults['age'].mean():.1f})")
print(f"  Pediatric (<18 yrs)   : {len(pediatric)} patients (Mean age: {pediatric['age'].mean():.1f})")

print("\n--- 2. RECORDING LEVEL METADATA ---")
print(f"Total Audio Recordings: {len(recordings_df)}")
total_audio_hours = recordings_df["duration_sec"].sum() / 3600
print(f"Total Audio Duration  : {total_audio_hours:.2f} hours ({recordings_df['duration_sec'].sum():.1f} seconds)")
print(f"Recording Duration    : Mean={recordings_df['duration_sec'].mean():.2f}s, Median={recordings_df['duration_sec'].median():.2f}s, Min={recordings_df['duration_sec'].min():.2f}s, Max={recordings_df['duration_sec'].max():.2f}s")

print("\nSampling Rates Breakdown:")
for sr, cnt in recordings_df["sample_rate"].value_counts().items():
    pct = (cnt / len(recordings_df)) * 100
    print(f"  {sr:7d} Hz : {cnt:3d} recordings ({pct:5.2f}%)")

print("\nRecording Equipment Distribution:")
for eq, cnt in recordings_df["equipment"].value_counts().items():
    pct = (cnt / len(recordings_df)) * 100
    print(f"  {eq:14s}: {cnt:3d} recordings ({pct:5.2f}%)")

print("\nChest Anatomical Locations:")
for loc, cnt in recordings_df["chest_location"].value_counts().items():
    pct = (cnt / len(recordings_df)) * 100
    print(f"  {loc:16s}: {cnt:3d} recordings ({pct:5.2f}%)")

print("\nAcquisition Mode:")
for m, cnt in recordings_df["acq_mode"].value_counts().items():
    pct = (cnt / len(recordings_df)) * 100
    print(f"  {m:16s}: {cnt:3d} recordings ({pct:5.2f}%)")

print("\n--- 3. RESPIRATORY CYCLES & ADVENTITIOUS SOUNDS ---")
print(f"Total Annotated Respiratory Cycles: {len(cycles_df)}")
print(f"Cycles per Recording (Mean ± SD) : {recordings_df['num_cycles'].mean():.2f} ± {recordings_df['num_cycles'].std():.2f}")
print(f"Cycle Duration (Mean ± SD)       : {cycles_df['cycle_duration'].mean():.2f} ± {cycles_df['cycle_duration'].std():.2f}s (Median: {cycles_df['cycle_duration'].median():.2f}s)")

print("\n4-Class Adventitious Sound Distribution:")
class_counts = cycles_df["class"].value_counts()
for cls, cnt in class_counts.items():
    pct = (cnt / len(cycles_df)) * 100
    print(f"  {cls:8s}: {cnt:5d} ({pct:5.2f}%)")

print("\nMarginal Adventitious Sound Counts:")
total_crackles = cycles_df['crackle'].sum()
total_wheezes = cycles_df['wheeze'].sum()
both_crackles_wheezes = len(cycles_df[(cycles_df['crackle']==1) & (cycles_df['wheeze']==1)])
print(f"  Total Cycles with Crackles: {total_crackles} ({total_crackles/len(cycles_df)*100:.2f}%)")
print(f"  Total Cycles with Wheezes : {total_wheezes} ({total_wheezes/len(cycles_df)*100:.2f}%)")
print(f"  Cycles with BOTH          : {both_crackles_wheezes} ({both_crackles_wheezes/len(cycles_df)*100:.2f}%)")

print("\n--- 4. TRAIN / TEST SPLIT COMPOSITION ---")
train_recs = recordings_df[recordings_df["split"]=="train"]
test_recs = recordings_df[recordings_df["split"]=="test"]
train_pats = set(train_recs["patient_id"])
test_pats = set(test_recs["patient_id"])
overlap_pats = train_pats.intersection(test_pats)

print(f"Train Recordings: {len(train_recs)} | Patients: {len(train_pats)}")
print(f"Test Recordings : {len(test_recs)} | Patients: {len(test_pats)}")
print(f"Patient Overlap Between Train & Test: {len(overlap_pats)} (0 means patient-independent split!)")

print("\nTrain vs Test 4-Class Breakdown (Cycles):")
train_cycles = cycles_df[cycles_df["split"]=="train"]["class"].value_counts()
test_cycles = cycles_df[cycles_df["split"]=="test"]["class"].value_counts()
split_comp = pd.DataFrame({"Train Cycles": train_cycles, "Test Cycles": test_cycles})
split_comp["Train %"] = (split_comp["Train Cycles"] / split_comp["Train Cycles"].sum()) * 100
split_comp["Test %"] = (split_comp["Test Cycles"] / split_comp["Test Cycles"].sum()) * 100
print(split_comp.to_string())

print("="*70)

# -------------------------------------------------------------
# 4. GENERATE PUBLICATION-QUALITY FIGURES
# -------------------------------------------------------------
print("\n[4/6] Generating visualization figures...")

# Set cohesive modern color palette
palette_disease = sns.color_palette("Set2", len(diag_counts))
palette_classes = ["#2b5c8f", "#e26d5c", "#f4a261", "#e76f51"]

# -------------------------------------------------------------
# Figure 1: Patient Diagnosis & Demographics
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1A: Diagnosis Bar Plot
sns.countplot(
    data=patients_df, 
    y="diagnosis", 
    order=patients_df["diagnosis"].value_counts().index, 
    palette="viridis", 
    ax=axes[0],
    hue="diagnosis",
    legend=False
)
axes[0].set_title("A: Patient Diagnosis Distribution (N=126)", fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel("Number of Patients", fontsize=11)
axes[0].set_ylabel("Diagnosis", fontsize=11)
for p in axes[0].patches:
    width = p.get_width()
    axes[0].annotate(f'{int(width)} ({width/126*100:.1f}%)',
                     (width + 1, p.get_y() + p.get_height() / 2.),
                     va='center', fontsize=9.5)

# 1B: Age Distribution by Diagnosis
sns.boxplot(
    data=patients_df, 
    x="diagnosis", 
    y="age", 
    palette="tab10", 
    ax=axes[1],
    hue="diagnosis",
    legend=False
)
axes[1].set_title("B: Age by Respiratory Diagnosis", fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel("Diagnosis", fontsize=11)
axes[1].set_ylabel("Age (Years)", fontsize=11)
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(18, color='gray', linestyle='--', alpha=0.7, label='Adult/Pediatric Cutoff (18y)')
axes[1].legend(loc='upper right', fontsize=8.5)

# 1C: Gender Distribution by Diagnosis
gender_diag = pd.crosstab(patients_df["diagnosis"], patients_df["sex"])
gender_diag.plot(kind="barh", stacked=True, color=["#f08080", "#4682b4"], ax=axes[2])
axes[2].set_title("C: Sex Distribution Across Diagnoses", fontsize=13, fontweight='bold', pad=12)
axes[2].set_xlabel("Number of Patients", fontsize=11)
axes[2].set_ylabel("")
axes[2].legend(title="Sex", fontsize=10)

plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig1_disease_and_demographics.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Recording Acquisition & Equipment Breakdown
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# 2A: Chest Location
loc_order = recordings_df["chest_location"].value_counts().index
sns.countplot(
    data=recordings_df, 
    y="chest_location", 
    order=loc_order, 
    palette="mako", 
    ax=axes[0, 0],
    hue="chest_location",
    legend=False
)
axes[0, 0].set_title("A: Auscultation Chest Locations (N=920)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Number of Audio Recordings", fontsize=10)
axes[0, 0].set_ylabel("Chest Location", fontsize=10)
for p in axes[0, 0].patches:
    width = p.get_width()
    axes[0, 0].annotate(f'{int(width)} ({width/920*100:.1f}%)',
                        (width + 5, p.get_y() + p.get_height() / 2.),
                        va='center', fontsize=9)

# 2B: Recording Equipment
eq_order = recordings_df["equipment"].value_counts().index
sns.countplot(
    data=recordings_df, 
    x="equipment", 
    order=eq_order, 
    palette="crest", 
    ax=axes[0, 1],
    hue="equipment",
    legend=False
)
axes[0, 1].set_title("B: Recording Equipment / Stethoscope Type", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Equipment", fontsize=10)
axes[0, 1].set_ylabel("Number of Recordings", fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=20)
for p in axes[0, 1].patches:
    h = p.get_height()
    axes[0, 1].annotate(f'{int(h)} ({h/920*100:.1f}%)',
                        (p.get_x() + p.get_width() / 2., h + 6),
                        ha='center', fontsize=9)

# 2C: Recording Duration Distribution
sns.histplot(
    data=recordings_df, 
    x="duration_sec", 
    bins=30, 
    kde=True, 
    color="#2a9d8f", 
    ax=axes[1, 0]
)
axes[1, 0].set_title("C: Audio Recording Duration Distribution", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Duration (seconds)", fontsize=10)
axes[1, 0].set_ylabel("Count of Recordings", fontsize=10)
axes[1, 0].axvline(recordings_df["duration_sec"].median(), color='red', linestyle='--', label=f'Median: {recordings_df["duration_sec"].median():.1f}s')
axes[1, 0].legend()

# 2D: Acquisition Mode by Equipment
sns.countplot(
    data=recordings_df, 
    x="equipment", 
    hue="acq_mode", 
    palette="Set1", 
    ax=axes[1, 1]
)
axes[1, 1].set_title("D: Acquisition Mode (Single vs Multi-Channel) by Equipment", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Equipment", fontsize=10)
axes[1, 1].set_ylabel("Count", fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig2_recording_acquisition_breakdown.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Sampling Rate Heterogeneity
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sr_counts = recordings_df["sample_rate"].value_counts().sort_index()
colors = ["#4575b4", "#74add1", "#fdae61", "#f46d43"]
axes[0].pie(sr_counts, labels=[f"{k} Hz" for k in sr_counts.index], autopct='%1.1f%%',
            startangle=140, colors=colors, explode=[0.05]*len(sr_counts),
            textprops={'fontsize': 11})
axes[0].set_title("A: Distribution of Sampling Rates Across 920 Recordings", fontsize=12, fontweight='bold')

sns.countplot(data=recordings_df, x="equipment", hue="sample_rate", palette="Spectral", ax=axes[1])
axes[1].set_title("B: Native Sampling Rates by Recording Device", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Equipment", fontsize=10)
axes[1].set_ylabel("Recording Count", fontsize=10)
axes[1].legend(title="Sampling Rate (Hz)", fontsize=9.5)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "fig3_sampling_rate_heterogeneity.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}")

# -------------------------------------------------------------
# Figure 4: Respiratory Cycles & Adventitious Sounds
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# 4A: 4-Class Cycle Distribution
class_order = ["Normal", "Crackle", "Wheeze", "Both"]
class_colors = ["#2b8a3e", "#e67700", "#d6336c", "#7048e8"]
sns.barplot(
    x=class_counts[class_order].index, 
    y=class_counts[class_order].values, 
    palette=class_colors, 
    ax=axes[0, 0],
    hue=class_counts[class_order].index,
    legend=False
)
axes[0, 0].set_title(f"A: Adventitious Sound Class Imbalance (N={len(cycles_df)} Cycles)", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Acoustic Class", fontsize=10)
axes[0, 0].set_ylabel("Number of Cycles", fontsize=10)
for p in axes[0, 0].patches:
    h = p.get_height()
    axes[0, 0].annotate(f'{int(h)}\n({h/len(cycles_df)*100:.1f}%)',
                        (p.get_x() + p.get_width() / 2., h + 40),
                        ha='center', fontsize=9.5)

# 4B: Cycle Duration Distribution by Class
sns.boxplot(
    data=cycles_df, 
    x="class", 
    y="cycle_duration", 
    order=class_order, 
    palette=class_colors, 
    ax=axes[0, 1],
    hue="class",
    legend=False,
    showfliers=False
)
axes[0, 1].set_title("B: Respiratory Cycle Duration by Sound Class", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Acoustic Class", fontsize=10)
axes[0, 1].set_ylabel("Cycle Duration (seconds)", fontsize=10)
axes[0, 1].axhline(cycles_df["cycle_duration"].median(), color="red", linestyle="--", label=f"Median: {cycles_df['cycle_duration'].median():.2f}s")
axes[0, 1].legend()

# 4C: Sound Class Breakdown by Disease Diagnosis
diag_class_ct = pd.crosstab(cycles_df["diagnosis"], cycles_df["class"], normalize="index") * 100
diag_class_ct = diag_class_ct.reindex(columns=class_order)
diag_class_ct.plot(
    kind="bar", 
    stacked=True, 
    color=class_colors, 
    ax=axes[1, 0]
)
axes[1, 0].set_title("C: Relative Sound Class Proportions per Diagnosis (%)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Diagnosis", fontsize=10)
axes[1, 0].set_ylabel("Percentage of Cycles (%)", fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].legend(title="Class", loc="upper right")

# 4D: Sound Class Breakdown by Chest Location
loc_class_ct = pd.crosstab(cycles_df["chest_location"], cycles_df["class"], normalize="index") * 100
loc_class_ct = loc_class_ct.reindex(columns=class_order)
loc_class_ct.plot(
    kind="bar", 
    stacked=True, 
    color=class_colors, 
    ax=axes[1, 1]
)
axes[1, 1].set_title("D: Relative Sound Class Proportions by Chest Location (%)", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Chest Location", fontsize=10)
axes[1, 1].set_ylabel("Percentage of Cycles (%)", fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=30)
axes[1, 1].legend(title="Class", loc="upper right")

plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "fig4_respiratory_cycles_adventitious_sounds.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Saved: {fig4_path}")

# -------------------------------------------------------------
# Figure 5: Official Train / Test Benchmark Split
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5A: Split Count of Recordings & Cycles
split_summary = pd.DataFrame({
    'Train': [len(train_recs), len(cycles_df[cycles_df['split']=='train'])],
    'Test': [len(test_recs), len(cycles_df[cycles_df['split']=='test'])]
}, index=['Recordings', 'Cycles'])
split_summary.plot(kind='bar', color=['#3a86ff', '#ff006e'], ax=axes[0])
axes[0].set_title("A: Train vs Test Set Volumes", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Count", fontsize=10)
axes[0].tick_params(axis='x', rotation=0)
for p in axes[0].patches:
    h = p.get_height()
    axes[0].annotate(f'{int(h)}',
                     (p.get_x() + p.get_width() / 2., h + 30),
                     ha='center', fontsize=9.5)

# 5B: Diagnosis Distribution in Train vs Test
diag_split = pd.crosstab(patients_df["diagnosis"], recordings_df.groupby("patient_id")["split"].first())
diag_split.plot(kind="bar", color=['#ff006e', '#3a86ff'], ax=axes[1])
axes[1].set_title("B: Patient Diagnosis Split (Patient-Independent)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Diagnosis", fontsize=10)
axes[1].set_ylabel("Number of Patients", fontsize=10)
axes[1].tick_params(axis='x', rotation=35)
axes[1].legend(["Test", "Train"])

# 5C: 4-Class Sound Proportions in Train vs Test
split_class_pct = pd.crosstab(cycles_df["class"], cycles_df["split"], normalize="columns") * 100
split_class_pct = split_class_pct.reindex(class_order)
split_class_pct.plot(kind="bar", color=['#ff006e', '#3a86ff'], ax=axes[2])
axes[2].set_title("C: Sound Class Consistency Between Train & Test (%)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Class", fontsize=10)
axes[2].set_ylabel("Percentage of Split (%)", fontsize=10)
axes[2].tick_params(axis='x', rotation=0)
axes[2].legend(["Test", "Train"])

plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "fig5_train_test_distribution.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Saved: {fig5_path}")

# -------------------------------------------------------------
# Figure 6: Waveforms and Mel-Spectrograms of Sound Classes
# -------------------------------------------------------------
print("\n[5/6] Extracting representative audio cycle spectrograms...")

# Select 1 representative cycle for each class
sample_cycles = {}
for cls in ["Normal", "Crackle", "Wheeze", "Both"]:
    sub = cycles_df[(cycles_df["class"] == cls) & (cycles_df["cycle_duration"] >= 1.5) & (cycles_df["cycle_duration"] <= 3.5)]
    sample_cycles[cls] = sub.iloc[0]

fig, axes = plt.subplots(4, 2, figsize=(16, 12))

for idx, (cls, row) in enumerate(sample_cycles.items()):
    rec_name = row["recording_name"]
    wav_path = os.path.join(RAW_DATA_DIR, f"{rec_name}.wav")
    c_start = row["cycle_start"]
    c_end = row["cycle_end"]
    
    # Load audio chunk
    y, sr = librosa.load(wav_path, sr=None, offset=c_start, duration=c_end - c_start)
    
    # Plot Waveform
    time_axis = np.linspace(0, len(y)/sr, len(y))
    axes[idx, 0].plot(time_axis, y, color=class_colors[idx], linewidth=0.8)
    axes[idx, 0].set_title(f"Class: {cls} | File: {rec_name} ({row['diagnosis']}) | {row['equipment']}", fontsize=11, fontweight='bold')
    axes[idx, 0].set_ylabel("Amplitude", fontsize=9.5)
    if idx == 3:
        axes[idx, 0].set_xlabel("Time (seconds)", fontsize=9.5)
        
    # Plot Mel-Spectrogram
    n_mels = 64
    n_fft = min(1024, len(y))
    hop_length = n_fft // 4
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=min(2000, sr//2))
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=min(2000, sr//2), ax=axes[idx, 1], cmap='magma')
    fig.colorbar(img, ax=axes[idx, 1], format='%+2.0f dB')
    axes[idx, 1].set_title(f"Mel-Spectrogram ({cls}) - Freq Range [0, {min(2000, sr//2)} Hz]", fontsize=11, fontweight='bold')
    axes[idx, 1].set_ylabel("Hz", fontsize=9.5)
    if idx == 3:
        axes[idx, 1].set_xlabel("Time (seconds)", fontsize=9.5)

plt.tight_layout()
fig6_path = os.path.join(FIGURES_DIR, "fig6_audio_spectrogram_examples.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"Saved: {fig6_path}")

print("\n[6/6] EDA complete! All figures and data summaries generated successfully.")
