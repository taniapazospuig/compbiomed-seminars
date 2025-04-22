import pickle
import numpy as np
import pandas as pd
import scipy.signal as sp
from scipy.interpolate import interp1d
from collections import Counter
import matplotlib.pyplot as plt
import random

# === STEP 1: Load the all_points dataset ===
with open(r"data/all_points_may_2024-001.pkl", "rb") as f:
    all_points = pickle.load(f)

# === STEP 2: Define ECG leads we expect ===
ecq_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# === STEP 3: Preprocessing function (resample + filter) ===
def preprocess_ecg_simple(ecg_signals, fs=1000, target_fs=250, high=0.5, low=100.0):
    timepoints = ecg_signals.shape[0]
    new_timepoints = int(timepoints * target_fs / fs)
    ecg_resampled = np.zeros((new_timepoints, ecg_signals.shape[1]))
    for lead in range(ecg_signals.shape[1]):
        f = interp1d(np.arange(timepoints), ecg_signals[:, lead])
        ecg_resampled[:, lead] = f(np.linspace(0, timepoints - 1, new_timepoints))

    # Apply bandpass filter
    b_high, a_high = sp.butter(2, high / (target_fs / 2), btype='high')
    b_low, a_low = sp.butter(2, low / (target_fs / 2), btype='low')
    ecg_filtered = sp.filtfilt(b_high, a_high, ecg_resampled, axis=0)
    ecg_filtered = sp.filtfilt(b_low, a_low, ecg_filtered, axis=0)

    return ecg_filtered

# === STEP 4: Load SOO mapping from Excel (Hoja1 + Hoja2) ===
label_map_hoja1 = pd.read_excel(r"data/labels_FontiersUnsupervised.xlsx", sheet_name="Hoja1")
label_map_hoja2 = pd.read_excel(r"data/labels_FontiersUnsupervised.xlsx", sheet_name="Hoja2")
soo_to_chamber = {**dict(zip(label_map_hoja1["SOO"], label_map_hoja1["SOO_Chamber"])),
                  **dict(zip(label_map_hoja2["SOO"], label_map_hoja2["SOO_chamber"]))}

# === STEP 5: Normalize chamber classification (Left / Right / Other) ===
def normalize_chamber(label):
    if isinstance(label, str):
        l = label.upper()
        if "RVOT" in l or "RIGHT" in l or "TRICUS" in l:
            return "Right"
        elif "LVOT" in l or "LEFT" in l or "MITRAL" in l or "AORT" in l:
            return "Left"
    return "OTHER"

# === STEP 6: Loop through all patients, extract ECGs ===
X, y = [], []

for patient_id, patient_data in all_points.items():
    structures = patient_data.get("Structures")
    if not isinstance(structures, dict):
        continue

    raw_soo_list = patient_data.get("SOO", [])
    raw_soo = raw_soo_list[0] if isinstance(raw_soo_list, list) and raw_soo_list else None
    mapped_chamber = normalize_chamber(soo_to_chamber.get(raw_soo, "OTHER"))

    for structure_name, donor_dict in structures.items():
        for donor_id, ecg_dict in donor_dict.items():
            if isinstance(ecg_dict, dict) and all(lead in ecg_dict for lead in ecq_leads):
                try:
                    ecg_matrix = np.array([ecg_dict[lead] for lead in ecq_leads]).T
                    if ecg_matrix.ndim == 2 and ecg_matrix.shape[0] > 0:
                        ecg_clean = preprocess_ecg_simple(ecg_matrix)
                        X.append(ecg_clean)
                        y.append(mapped_chamber)
                except Exception as e:
                    print(f"⚠️ Failed to process ECG from {donor_id} in {patient_id}/{structure_name}: {e}")

# === STEP 7: Convert and count classes ===
X = np.stack(X)
y_labels = [normalize_chamber(label) for label in y]
print("\n📊 Class distribution BEFORE augmentation:")
counts = Counter(y_labels)
for label, count in counts.items():
    print(f"{label}: {count}")
# === STEP 8: Data augmentation for Left class ===
def augment_ecg(ecg, noise_level=0.01, shift_range=10):
    augmented = []
    # 1. Gaussian noise
    augmented.append(ecg + np.random.normal(0, noise_level, ecg.shape))
    # 2. Time shift
    augmented.append(np.roll(ecg, random.randint(-shift_range, shift_range), axis=0))
    # 3. Right-side lead dropout (simulate partial dropout of right-side leads)
    right_leads = ['V1', 'V2', 'V3']
    lead_indices = [ecq_leads.index(lead) for lead in right_leads if lead in ecq_leads]
    dropout = ecg.copy()
    # Randomly drop one of the right leads only
    if lead_indices:
        drop_idx = random.choice(lead_indices)
        dropout[:, drop_idx] = 0
    augmented.append(dropout)
    return augmented

X_aug, y_aug = [], []
for xi, yi in zip(X, y_labels):
    X_aug.append(xi)
    y_aug.append(yi)
    if yi == "Left":
        for aug in augment_ecg(xi):
            X_aug.append(aug)
            y_aug.append(yi)

# Final formatting
X_aug = np.stack(X_aug)
y_final = np.array([0 if label == "Left" else 1 for label in y_aug if label in ["Left", "Right"]])
X_final = np.array([x for x, label in zip(X_aug, y_aug) if label in ["Left", "Right"]])

print("\n✅ Augmentation complete. Final dataset shape:", X_final.shape)
print("Left (0):", sum(y_final == 0))
print("Right (1):", sum(y_final == 1))
