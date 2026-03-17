# Healthcare ML Tutorial: ICU Mortality Prediction

## Overview

This tutorial builds a complete machine learning pipeline to predict **in-hospital mortality for ICU patients** using structured clinical data. It covers the full workflow from data extraction through model evaluation and interpretability.

## Data Source

- **MIMIC-III Clinical Database** (v1.4) accessed via **Google BigQuery**
- Dataset: `physionet-data.mimiciii_clinical` (structured data) and `physionet-data.mimiciii_notes` (discharge summaries)
- 5 tables used: `patients`, `admissions`, `icustays`, `chartevents`, `labevents`
- After merging and filtering to adult patients: **53,330 ICU stays** from **38,511 patients**
- Prediction window: **first 24 hours** of ICU admission only
- Mortality rate: **12.3%**

> **Access Requirement:** MIMIC-III is a restricted dataset. You must complete CITI training and sign the PhysioNet Data Use Agreement before accessing the data. See [PhysioNet](https://physionet.org/content/mimiciii/1.4/) for details.

## What's Inside

| Section | Topic | Description |
|---------|-------|-------------|
| 1 | Setup & Environment | Package installation, random seeds, GPU check |
| 2 | Data Loading | BigQuery authentication, query 5 MIMIC-III tables, merge into one DataFrame |
| 3 | Preprocessing | Drop high-missing features, clip outliers, impute missing values, normalize |
| 4 | Feature Engineering | One-hot encode categoricals, build feature matrix, stratified 70/15/15 split |
| 5 | Traditional ML | Logistic Regression, Random Forest, XGBoost with stratified 5-fold CV |
| 6 | Deep Learning | PyTorch feedforward neural network (2 hidden layers) with early stopping |
| 7 | Evaluation | ROC curves, PR curves, metrics comparison table (AUROC, AUPRC, F1, etc.) |
| 8 | Interpretability | SHAP summary plot, force plot, tree-based feature importance |
| 9 | Bonus: Multimodal | ClinicalBERT discharge summary embeddings + structured features |
| 10 | Conclusion | Key takeaways, limitations, references |

## Models & Results

| Model | CV AUROC |
|-------|----------|
| Logistic Regression | 0.8340 ± 0.005 |
| Random Forest | 0.8413 ± 0.006 |
| XGBoost | 0.8223 ± 0.006 |
| Neural Network | Early stopping at epoch 60, val loss 0.2767 |

## How to Run

1. Open `Healthcare_ML_Tutorial_ICU_Mortality.ipynb` in **Google Colab**
2. Set your GCP project ID in Section 2: `PROJECT_ID = "your-project-id"`
3. Run all cells sequentially (Section 9 is optional)
