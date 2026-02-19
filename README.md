# Synthetic Generation and Evaluation of Longitudinal Clinical Data

This project focuses on synthetic generation and evaluation of longitudinal clinical time-series data using the Parkinson motor_UPDRS dataset.

The objective is to evaluate whether synthetic sequences preserve:

- Statistical fidelity to real longitudinal data
- Predictive utility (train on synthetic, test on real)
- Privacy properties (avoid memorization of real patients)

At the current stage, development is restricted to longitudinal clinical data only.

---

## Project Scope

Although the repository was initially designed for multimodal biomedical data, the current active development focuses exclusively on:

- Longitudinal clinical pipeline (Parkinson dataset)

Other modalities such as MRI ROI and histopathology tabular pipelines are not actively developed.

---

## Dataset

Parkinson longitudinal dataset with:

- Patient-level time-series
- Fixed-length windows (T6 sequences)
- Target: motor_UPDRS at time T6

Preprocessing includes:

- Patient-level train/test split (20% test patients)
- Per-patient chronological ordering using 'age' as time proxy
- Sliding-window construction of fixed-lenght sequences (seq_len=6)
- Target definition: motor_UPDRS at the last timestep of each window
- Feature normalization with StandardScaler (fit on train only)

---

## Repository Structure

```
multimodal-synthetic-biomedical/
│
├── clinical_longitudinal/
│   ├── data_processing.py        # Preprocessing pipeline
│   ├── baseline_gru.py           # GRU regression baseline (real data)
│   ├── ts_cvae.py                # Time-Series Conditional VAE model
│   ├── train_cvae.py             # Training + synthetic generation
│
├── shared_metrics/               # Fidelity / utility / privacy metrics
├── data/                         # Local datasets (not tracked)
├── notebooks/                    # Exploratory analysis
├── requirements.txt
└── README.md
```

---

## Current Implementation

### 1. Real Data Baseline – GRU

A GRU regression model is trained on real longitudinal data to predict motor_UPDRS at T6.

Purpose:
- Establish a realistic upper-bound performance
- Provide a reference for utility evaluation

Evaluation metric:
- Mean Squared Error (MSE)

---

### 2. Generative Model – TS-CVAE

A Time-Series Conditional Variational Autoencoder (TS-CVAE) is implemented to model longitudinal sequences.

The model:

- Encodes full temporal sequences
- Conditions on the normalized target (motor_UPDRS)
- Learns a latent representation of sequence dynamics
- Generates new synthetic sequences conditioned on sampled targets

The training script:

- Trains the generative model
- Saves the best checkpoint
- Generates a synthetic dataset
- Stores synthetic sequences and corresponding targets

---

## Evaluation Framework

The evaluation is structured across three dimensions:

### Fidelity Evaluation

Assesses statistical similarity between real and synthetic data:

- Feature-wise statistical comparison
- Distributional distance metrics
- Correlation structure comparison
- KS test per feature
- Wasserstein distance
- Correlation matrix distance

---

### Utility Evaluation

Assesses whether synthetic data preserves predictive structure:

1. Train GRU on synthetic data
2. Test on real test set
3. Compare performance with real-data baseline

If performance is close to the real baseline, synthetic data is considered useful.

---

### Privacy Evaluation

Assesses whether synthetic data memorizes real patients:

- Nearest-neighbor distance (synthetic vs real)
- Memorization risk inspection

The goal is to verify that generated sequences are not copies of real individuals.

---

## Current Status

Completed:

- Longitudinal preprocessing pipeline
- Real-data GRU baseline
- TS-CVAE implementation
- Synthetic sequence generation

In Progress:

- Full fidelity evaluation
- Utility evaluation (train on synthetic, test on real)
- Privacy analysis

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python -m clinical_longitudinal.data_processing
```

Train baseline:

```bash
python -m clinical_longitudinal.baseline_gru
```

Train TS-CVAE and generate synthetic data:

```bash
python -m clinical_longitudinal.train_cvae
```
