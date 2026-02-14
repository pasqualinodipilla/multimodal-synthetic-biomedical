# Multimodal Synthetic Biomedical Data

This project studies the generation and evaluation of synthetic biomedical data across multiple modalities.

The objective is to assess whether synthetic data can preserve:
- Statistical similarity to real data (fidelity)
- Predictive performance (utility)
- Privacy properties

The project is structured as a research-oriented evaluation framework.

---

## Project Structure

multimodal-synthetic-biomedical/

│  
├── clinical_longitudinal/      # Parkinson longitudinal pipeline  
│   ├── data_processing.py  
│   ├── baseline_gru.py  
│   └── config.py  
│  
├── mri_roi/                    # MRI ROI tabular pipeline  
├── histopathology_tabular/     # Histopathology tabular pipeline  
├── shared_metrics/             # Fidelity, utility and privacy metrics  
├── notebooks/                  # Exploratory analysis  
├── data/                       # Local datasets (not tracked)  
├── requirements.txt  
└── README.md  

---

## Current Implementation

### 1. Clinical Longitudinal Pipeline (Parkinson Dataset)

Implemented components:

- Preprocessing of longitudinal patient data
- Construction of fixed-length temporal sequences
- Train/test split at patient level
- GRU baseline model for regression on `motor_UPDRS`
- Evaluation using Mean Squared Error (MSE)

The GRU baseline trained on real data provides a reference performance.

---

### 2. Evaluation Framework

The project includes three evaluation dimensions:

### Fidelity Evaluation

Measures statistical similarity between real and synthetic data:
- Feature-wise statistical comparison
- Distributional distance metrics
- Correlation structure comparison

---

### Utility Evaluation

- Train predictive model on synthetic data
- Test on real data
- Compare performance with real-data baseline

If performance remains close to the real baseline, synthetic data is considered useful.

---

### Privacy Evaluation

- Nearest-neighbor distance (synthetic vs real)
- Memorization risk inspection

---

## What Remains to Be Completed

- Generation of synthetic longitudinal sequences
- Full fidelity evaluation between real and synthetic datasets
- Utility evaluation (train on synthetic, test on real)
- Privacy analysis on generated data
- Final structured report of results

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Notes

- Raw datasets are stored locally and are not included in the repository.
- The repository focuses on evaluation methodology rather than data distribution.
