# MIMIC-III: In-Hospital Mortality Prediction

Applying Machine Learning and Deep Learning to Electronic Health Records (EHR) for predicting in-hospital mortality using the MIMIC-III Clinical Database.

## Downstream Task

**In-Hospital Mortality Prediction** — Given a patient's first 24 hours of ICU data, predict whether they will survive their hospital stay.

## Dataset

- **Source:** MIMIC-III Clinical Database via Google BigQuery (`physionet-data.mimiciii_clinical`)
- **Cohort:** 47,114 adult ICU patients (age >= 18)
- **Mortality Rate:** 11.0% (5,190 died / 41,924 survived)
- **Features:** 71 total (demographics + vital signs + lab values, aggregated as mean/min/max/std from first 24h)

## Methods

### Machine Learning
| Model | Accuracy | F1-Score | AUROC | AUPRC |
|-------|----------|----------|-------|-------|
| **XGBoost** | **0.8509** | **0.5180** | **0.8946** | **0.6211** |
| XGBoost (Tuned) | 0.8202 | 0.4891 | 0.8911 | 0.6075 |
| Random Forest | 0.8703 | 0.5174 | 0.8739 | 0.5756 |
| RF (Tuned) | 0.8703 | 0.5174 | 0.8739 | 0.5756 |
| SVM (Linear) | 0.8898 | 0.0000 | 0.8448 | 0.4683 |
| Logistic Regression | 0.7843 | 0.4239 | 0.8418 | 0.4678 |

### Deep Learning
| Model | Accuracy | F1-Score | AUROC | AUPRC |
|-------|----------|----------|-------|-------|
| LSTM (temporal vitals) | 0.7404 | 0.3710 | 0.8003 | 0.4681 |

- **LSTM Architecture:** 24 time steps x 8 vitals, 64 hidden units, 19K parameters
- **Bonus:** GridSearchCV hyperparameter tuning on RF and XGBoost

## Key Findings

1. **XGBoost is the top performer** (AUROC=0.895) — captures non-linear feature interactions
2. **Clinical instability predicts mortality** — vital sign variability (std), lactate, and kidney markers are top features
3. **Class imbalance is critical** — AUPRC is more informative than accuracy for this 11% mortality rate
4. **LSTM shows promise** — AUROC=0.80 using only temporal vitals without demographics or labs

## Repository Structure

```
Healthcare ML/DL/
├── MIMIC_III_Mortality_Prediction.ipynb    # Clean template notebook
├── MIMIC_III_Mortality_Prediction_Slides.pdf  # Slide deck with visualizations
└── README.md                               # This file
```

## Requirements

- Google Colab (or Python 3.8+)
- Google Cloud project with BigQuery access to MIMIC-III
- PhysioNet credentialed access to MIMIC-III

### Python Packages
```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost
torch (PyTorch)
google-cloud-bigquery, pandas-gbq
```

## How to Run

1. Open the notebook in Google Colab
2. Set your GCP `PROJECT_ID` in the authentication cell
3. Run all cells sequentially
4. BigQuery handles all data loading — no CSV downloads needed

## Visualizations

The notebook includes:
- PCA and t-SNE dimensionality reduction plots
- ROC and Precision-Recall curves for all models
- Random Forest feature importance
- LSTM training loss and validation AUROC curves
- Confusion matrices for top 3 models
- Final model comparison bar charts
