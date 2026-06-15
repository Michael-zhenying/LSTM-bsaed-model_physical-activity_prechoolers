# LSTM-bsaed-model_physical-activity_prechoolers
Deep Learning (Bi-LSTM) frameworks for classifying preschoolers' physical activity and sedentary behaviour from thigh-worn Fibion and wrist-worn ActiGraph raw accelerometer data.

Markdown# Multi-Sensor Deep Learning Framework for Preschooler Physical Activity Classification

This repository contains the finalized, production-ready deep learning models, pre-fitted preprocessing vectors, and deployment script templates from the manuscript titled: **Comparing Deep Learning and Cut-Points for Preschooler Physical Activity and Sedentary Behaviour Classification from Thigh, Wrist, and Multi-Site Accelerometers**. 

The pipeline leverages a Bidirectional Long Short-Term Memory (Bi-LSTM) network architecture optimized to bypass manual, hand-crafted feature engineering. It directly interprets continuous, raw triaxial acceleration patterns to classify preschooler movement intensities into Sedentary Behavior (SB), Light Physical Activity (LPA), or Moderate-to-Vigorous Physical Activity (MVPA).

---

## Repository Architecture
``` text
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
```

## Environmental Requirements
To ensure execution compatibility and prevent graph errors, deploy this framework within a Python environment matching the following primary version footprints:
*    Python: v3.12.4
*    TensorFlow: v2.15.0
*    scikit-learn: v1.4+numpy: v1.26+
*    pandas: v2.2+
Install dependencies easily via terminal command line:
pip install tensorflow==2.15.0 scikit-learn numpy pandas

## Model & Preprocessing Pipeline Specifics
Our validation protocols yielded a highly rigid structural blueprint. For proper deployment on external datasets, input vectors must strictly replicate these preprocessing laws:
1. **Sampling Alignment**: Sensor feeds must be mapped to a uniform 30Hz sampling rate. Thigh data collected at different native sample rates (e.g., Fibion at 12.5Hz) must be scaled up to 30Hz using a Nearest-Neighbor interpolation rule rather than linear smoothing to preserve temporal burst signatures.
2. **Window Segmentation**: Continuous signal logs must be chopped into sequential 3-second windows (exactly 90 frames x n hannels).
3. **Signal Calibration**: Before pushing input window matrices into the network, raw acceleration values must pass through the corresponding pre-fitted StandardScaler loaded from the scalers/ path.

**Model Layer Matrix Dimensions**
*    Combined Loop: Shape (90, 6) -> ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 'x', 'y', 'z']
*    ActiGraph Only Loop: Shape (90, 3) -> ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']
*    Fibion Only Loop: Shape (90, 3) -> ['x', 'y', 'z']

## Open Science License
This framework, its weights, and template parameters are distributed under the MIT License. You are free to copy, modify, and distribute these pipelines in future research tracking childhood movement behaviors, provided proper academic citation is credited to the parent manuscript.
