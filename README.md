# 🫀 ECG Origin Classification in Ventricular Arrhythmias  
### Final Seminar Project — Computational Models in Biomedicine (CompBioMed)  
**Universitat Pompeu Fabra (UPF), Barcelona**


Project Summary: 
Our project presents a complete machine learning pipeline for classifying the anatomical origin of ventricular arrhythmias using multilead ECG recordings and clinical metadata. The goal was to computationally distinguish between  **Right vs Left ventricular outflow tract origins**, and more specifically, between **RCC vs RVOTSEPTUM** cases.

Our work combines signal preprocessing, deep learning-based segmentation, morphological feature extraction, metadata integration, feature selection, and supervised learning — with a focus on model interpretability and class imbalance handling.

## 🧠 Methodology
1. **Dataset before preprocesing:**:  
   - We had 12-lead ECGs from real patients stored in structured `.pkl` format  
   - We also had some clinical features: Sex, Age, BMI, Height, Weight, comorbidities, PVC transition types, etc.  
   - There were 2 classification tasks:
     - **Left vs Right ventricular chamber**
     - **RCC vs RVOTSEPTUM** 

2. **Signal Processing & Preprocessing pf the big and small datasets**:
   - Signals  were resampled (1000Hz → 250Hz) and filtered (bandpass: 0.5–100Hz)
   - We did R-peak alignment applied using Lead II
   - Finally we obtained: Aligned ECG windows 

3. **Segmentation & Feature Extraction**:
   - To segment ECG waves and extract morphological features, we relied on an ensemble of pretrained models built in PyTorch:
         - R/S amplitudes  
         - QRS duration  
         - T-wave polarity  
   - Features computed per lead → total of 60+ morphology descriptors

4. **Metadata Fusion between feature extraction and selection**:
   - We merged each ECG sample with its patient metadata via `(PatientID, SampleID, Structure)`
   - Some categorical features (Sex, Smoker, PVC transition) were one-hot encoded (binarized). 

5. **Feature Selection**:
   - We did some comparison between **ANOVA F-test** and **Mutual Information** methods
   - We selected the top 30 features per strategy
   - The interpretation of these features was based on SHAP values

6. **Classification Models that worked best**:
   - Left vs Right: Random Forest, XGBoost, Logistic Regression  
   - RCC vs RVOTSEPTUM: Logistic Regression, SVM  
   - Evaluation: Accuracy, F1-score, Confusion Matrices, SHAP explanations

7. **Class Imbalance Strategy, done before model evaluation**:
   - Applied undersampling on dominant class (the Right one) to improve macro recall
   - The final balanced datasets improved interpretability and fairness

## 📊 Key Results

| Task                  | Best Model       | Accuracy | Highlights                              |
|-----------------------|------------------|----------|------------------------------------------|
| Left vs Right         | Random Forest    | ~76%     | Complex, nonlinear ECG patterns captured |
| RCC vs RVOTSEPTUM     | Logistic Regression | ~72%     | Simpler boundary, strong linear features |

- **SHAP analysis** confirmed us  the relevance of some specific ECG leads (V2–V5, AVF) and clinical data like: Sex, Height, PVC transitions, etc. 
- Mutual Information (MI) selected more nonlinear/continuous features, while ANOVA favored stable categorical variables

## 👩‍🔬 Authors & Course Info

This project was completed as part of the **CompBioMed (Computational Models in Biomedicine)** course at **Universitat Pompeu Fabra (UPF), Barcelona** during the Spring 2025 term.

Our project developed by: **[Iabel Expósito, Tania Pazos, Madeleine Fairey and Xavier Miret]**
