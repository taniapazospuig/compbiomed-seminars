import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# === STEP 1: Load ECG Data ===
with open(r"C:\Users\Usuario\Documents\CompBioMed\compbiomed-seminars\CompBioMed25_Seminars\full_data_corrected_2024.pkl", "rb") as f:
    data = pickle.load(f)

# === STEP 2: Identify ECG Leads and Build Tensor ===
ecg_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
signal_length = len(data['I'][0])
multi_lead_ecgs = np.zeros((181, len(ecg_leads), signal_length))
for i, lead in enumerate(ecg_leads):
    for j in range(181):
        multi_lead_ecgs[j, i, :] = data[lead][j]

# === STEP 3A: Preprocess all ECG signals to [samples x leads] at 250Hz ===

import scipy.signal as sp
from scipy.interpolate import interp1d

def preprocess_ecg_signal(ecg_signals, fs=1000, target_fs=250, high=0.5, low=100.0):
    """
    Preprocess a multi-lead ECG signal [samples, leads]:
    - Resample to target_fs
    - Bandpass filter
    """
    timepoints = ecg_signals.shape[0]
    new_timepoints = int(timepoints * target_fs / fs)
    ecg_resampled = np.zeros((new_timepoints, ecg_signals.shape[1]))
    for lead in range(ecg_signals.shape[1]):
        f = interp1d(np.arange(timepoints), ecg_signals[:, lead])
        ecg_resampled[:, lead] = f(np.linspace(0, timepoints - 1, new_timepoints))

    # Bandpass filter
    b_high, a_high = sp.butter(2, high / (target_fs / 2), btype='high')
    b_low, a_low = sp.butter(2, low / (target_fs / 2), btype='low')
    ecg_filtered = sp.filtfilt(b_high, a_high, ecg_resampled, axis=0)
    ecg_filtered = sp.filtfilt(b_low, a_low, ecg_filtered, axis=0)

    return ecg_filtered

# Apply to all ECGs
preprocessed_ecgs = []
for i in range(multi_lead_ecgs.shape[0]):
    signal_raw = multi_lead_ecgs[i].T  # shape [samples, leads]
    processed = preprocess_ecg_signal(signal_raw)
    preprocessed_ecgs.append(processed)

# Stack into 3D array if possible
try:
    preprocessed_ecgs = np.stack(preprocessed_ecgs)
    print("✅ All signals successfully preprocessed to shape:", preprocessed_ecgs.shape)
except:
    print("⚠️ Signals have different lengths. Stored as a list.")



# === STEP 3B: Load Hoja 1 Mapping ===
label_map_hoja1 = pd.read_excel(
    r"C:\Users\Usuario\Documents\CompBioMed\compbiomed-seminars\CompBioMed25_Seminars\labels_FontiersUnsupervised.xlsx",
    sheet_name="Hoja1"
)
soo_to_chamber_hoja1 = dict(zip(label_map_hoja1["SOO"], label_map_hoja1["SOO_Chamber"]))

# === STEP 4: Initial Mapping from Hoja1 ===
simplified_chambers = []
for entry in data["SOO"]:
    if isinstance(entry, str) and entry in soo_to_chamber_hoja1:
        simplified_chambers.append(soo_to_chamber_hoja1[entry])
    else:
        simplified_chambers.append("OTHER")

# === STEP 5: Apply Hoja 2 to remaining OTHER cases ===
label_map_hoja2 = pd.read_excel(
    r"C:\Users\Usuario\Documents\CompBioMed\compbiomed-seminars\CompBioMed25_Seminars\labels_FontiersUnsupervised.xlsx",
    sheet_name="Hoja2"
)
soo_to_chamber_hoja2 = dict(zip(label_map_hoja2["SOO"], label_map_hoja2["SOO_chamber"]))

# Update entries marked as "OTHER" using Hoja2
for i, entry in enumerate(data["SOO"]):
    if simplified_chambers[i] == "OTHER" and isinstance(entry, str) and entry in soo_to_chamber_hoja2:
        simplified_chambers[i] = soo_to_chamber_hoja2[entry]

# === STEP 6: Normalize to Left / Right Ventricle ===
def normalize_chamber(label):
    if isinstance(label, str):
        l = label.upper()
        if "RVOT" in l or "RIGHT" in l or "TRICUS" in l:
            return "Right"
        elif "LVOT" in l or "LEFT" in l or "MITRAL" in l or "AORT" in l:
            return "Left"
    return "OTHER"

final_chambers_normalized = [normalize_chamber(c) for c in simplified_chambers]

# === STEP 7: Plot Final Left vs Right ===
final_counts = Counter([c for c in final_chambers_normalized if c in ["Left", "Right"]])
excluded_final = final_chambers_normalized.count("OTHER")

plt.bar(final_counts.keys(), final_counts.values(), color=["tab:red", "tab:blue"])
plt.title("Final Mapping: Left vs Right Ventricle")
plt.xlabel("Chamber")
plt.ylabel("Number of Patients")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# === STEP 8A: Data Augmentation for "Left" Class ===
import random

def augment_ecg(ecg, noise_level=0.01, shift_range=10):
    augmented_versions = []

    # 1. Add Gaussian noise
    noise = ecg + np.random.normal(0, noise_level, ecg.shape)
    augmented_versions.append(noise)

    # 2. Time shift (circular roll)
    shift = random.randint(-shift_range, shift_range)
    shifted = np.roll(ecg, shift, axis=0)
    augmented_versions.append(shifted)

    return augmented_versions

# Separate original Left and Right class indices
left_indices = [i for i, label in enumerate(final_chambers_normalized) if label == "Left"]
right_indices = [i for i, label in enumerate(final_chambers_normalized) if label == "Right"]

# Create augmented dataset
X_augmented = []
y_augmented = []

# Add all Right (label 1)
for idx in right_indices:
    X_augmented.append(preprocessed_ecgs[idx])
    y_augmented.append(1)

# Add original and 2x augmented Left (label 0)
for idx in left_indices:
    original = preprocessed_ecgs[idx]
    X_augmented.append(original)
    y_augmented.append(0)
    for aug in augment_ecg(original):
        X_augmented.append(aug)
        y_augmented.append(0)

# Convert to arrays
X_augmented = np.stack(X_augmented)
y_augmented = np.array(y_augmented)

# Report balance
print("\n✅ Data Augmentation Complete")
print("X_augmented shape:", X_augmented.shape)
print("Left (0):", np.sum(y_augmented == 0))
print("Right (1):", np.sum(y_augmented == 1))

# === STEP 8B: Visualize Augmented vs Original ECGs ===
import matplotlib.pyplot as plt

# Plotting function
def plot_ecg_comparison(original, augmented_list, lead=0, sample_rate=250):
    time = np.arange(original.shape[0]) / sample_rate
    plt.figure(figsize=(12, 6))

    # Original
    plt.plot(time, original[:, lead], label="Original", linewidth=2)

    # Augmented
    for i, aug in enumerate(augmented_list):
        plt.plot(time, aug[:, lead], label=f"Augmented {i+1}", linestyle='--')

    plt.title(f"ECG Lead {ecg_leads[lead]} — Original vs Augmented")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Select an example Left-class patient
left_example_idx = left_indices[0]
original_signal = preprocessed_ecgs[left_example_idx]
augmented_signals = augment_ecg(original_signal)

# Plot comparison for lead V2 (or any lead you prefer)
plot_ecg_comparison(original_signal, augmented_signals, lead=6)  # V2 is index 6

# === STEP 9: Show Unclassified SOO Values ===
print("\n Unclassified Patients (OTHER):", excluded_final)
print("Unclassified SOO values:")
unclassified_soo = sorted(set(data["SOO"][i] for i, c in enumerate(final_chambers_normalized) if c == "OTHER" and isinstance(data["SOO"][i], str)))
for val in unclassified_soo:
    print("-", val)

# === STEP 10: Save the processed data ===
# === X_augmented: 3D array of shape (n_samples, time_steps, leads)
# Each element is a full multi-lead ECG signal, preprocessed to 250Hz
# - shape: [# samples, 625 timepoints, 12 leads]
# - already includes augmented versions for Left-class

# === y_augmented: 1D array of shape (n_samples,)
# - Labels: 0 = Left Ventricle, 1 = Right Ventricle

# Save both to a compressed .npz file
'''
np.savez_compressed(
    "ecg_data_balanced_250Hz.npz",
    X=X_augmented,
    y=y_augmented
)

print("✅ Saved as 'ecg_data_balanced_250Hz.npz'")
'''