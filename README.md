# ECG-Based Classification of Ventricular Arrhythmia Origin

**Final Project – Data Science and Computational Models in Biomedicine (CompBioMed), Universitat Pompeu Fabra (2025)**  

---

## Overview
This project developed a complete machine learning pipeline to classify the anatomical origin of ventricular arrhythmias using 12-lead ECG recordings and clinical metadata.  
Two classification tasks were addressed:  
1. Left vs Right ventricular outflow tract origin.  
2. Right Coronary Cusp (RCC) vs RVOT Septum origin for fine-grained right-sided classification.

---

## Key Contributions
- Performed data preprocessing, including resampling, bandpass filtering, R-peak alignment, and standardization of anatomical labels.  
- Extracted morphological ECG features such as wave amplitudes, QRS duration, and T-wave polarity using pretrained segmentation models.  
- Combined ECG-derived features with patient metadata for a richer feature space.  
- Compared ANOVA F-test and Mutual Information methods for feature selection.  
- Evaluated seven classifiers: Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, k-NN, and MLP.  
- Applied SHAP analysis to interpret predictions and relate feature importance to clinical reasoning.  
- Addressed class imbalance using patient-level undersampling of the majority class.

---

## Results
- Left vs Right: Random Forest with ANOVA-selected features achieved 0.86 accuracy and 0.68 macro recall.  
- RCC vs RVOT Septum: Logistic Regression with ANOVA-selected features achieved 0.63 accuracy and 0.81 macro recall.  

---

## Repository Contents
- [seminars-report.pdf](./seminars-report.pdf)
  Full methodology, experiments, and results.  
- [big_dataset_processing.ipynb](./big_dataset_processing.ipynb)
  Data preprocessing, model training, and SHAP explainability.
