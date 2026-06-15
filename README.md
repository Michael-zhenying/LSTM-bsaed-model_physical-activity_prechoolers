# LSTM-bsaed-model_physical-activity_prechoolers
Deep Learning (Bi-LSTM) frameworks for classifying preschoolers' physical activity and sedentary behaviour from thigh-worn Fibion and wrist-worn ActiGraph raw accelerometer data.

# Multi-Sensor Deep Learning Framework for Preschooler Physical Activity Classification

This repository contains the finalised, production-ready deep learning models, pre-fitted preprocessing vectors, and deployment script templates from the manuscript titled: **Comparing Deep Learning and Cut-Points for Preschooler Physical Activity and Sedentary Behaviour Classification from Thigh, Wrist, and Multi-Site Accelerometers**. 

The pipeline leverages a Bidirectional Long Short-Term Memory (Bi-LSTM) network architecture optimised to bypass manual, hand-crafted feature engineering. It directly interprets continuous, raw triaxial acceleration patterns to classify preschooler movement into Sedentary Behavior (SB), Light Physical Activity (LPA), or Moderate-to-Vigorous Physical Activity (MVPA).

---

## Repository Architecture

```text
├── README.md                 # User documentation and guide
├── template_inference.py     # Clean template for deploying models to new datasets
├── models/                   # Finalized trained neural network graphs
│   ├── final_model_combined.keras   # Multi-site configuration (Wrist + Thigh)
│   ├── final_model_actigraph.keras  # Single-site Wrist configuration
│   └── final_model_fibion.keras     # Single-site Thigh configuration
└── scalers/                  # Pre-fitted StandardScaler calibration files
    ├── scaler_combined.pkl
    ├── scaler_actigraph.pkl
    └── scaler_fibion.pkl
