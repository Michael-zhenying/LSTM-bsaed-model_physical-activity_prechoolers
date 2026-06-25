# LSTM-based Model for Preschooler Physical Activity Classification

This repository contains fixed-setting Bidirectional Long Short-Term Memory (Bi-LSTM) models for classifying preschoolers' physical activity and sedentary behaviour from thigh-worn Fibion and wrist-worn ActiGraph accelerometer data.

The models accompany the manuscript:

**Comparing Deep Learning and Cut-Points Methods for Preschooler Physical Activity and Sedentary Behaviour Classification from Thigh, Wrist, and Multi-Site Accelerometers**

The models classify each window into three activity classes:

- **SB**: sedentary behaviour
- **LPA**: light-intensity physical activity
- **MVPA**: moderate-to-vigorous physical activity

---

## Repository structure

```text
├── README.md
├── template_inference.py
├── models/
│   ├── final_model_combined.keras
│   ├── final_model_actigraph.keras
│   └── final_model_fibion.keras
└── scalers/
    ├── scaler_combined.pkl
    ├── scaler_actigraph.pkl
    └── scaler_fibion.pkl
```

---

## Model configurations

The final deployment models use a fixed model setting. No formal window-size search or hyperparameter tuning is required for inference.

| Configuration | Wear site / data stream | Input shape | Required columns, in order |
|---|---|---:|---|
| `actigraph` | Wrist ActiGraph acceleration | `(90, 3)` | `Accelerometer X`, `Accelerometer Y`, `Accelerometer Z` |
| `fibion` | Thigh Fibion acceleration + inclination | `(90, 6)` | `x`, `y`, `z`, `angle/x/scalar`, `angle/y/scalar`, `angle/z/scalar` |
| `combined` | Wrist + thigh acceleration/inclination | `(90, 9)` | `Accelerometer X`, `Accelerometer Y`, `Accelerometer Z`, `x`, `y`, `z`, `angle/x/scalar`, `angle/y/scalar`, `angle/z/scalar` |

---

## Fixed model setting

The final models use the following fixed setting:

- **Sampling rate**: 30 Hz
- **Window length**: 3 seconds
- **Samples per window**: 90
- **Window overlap**: none
- **Stride**: 3 seconds
- **Metadata**: not used in the final deployment model
- **Architecture**: Bi-LSTM + dropout + dense layer + softmax output
- **Bi-LSTM units**: 64
- **Dropout**: 0.2
- **Dense units**: 32
- **Optimizer during training**: Adam, learning rate = 0.001
- **Training loss**: categorical cross-entropy
- **Output classes**: SB, LPA, MVPA

The final application models were trained using all available participants after validation. The saved scaler files must be used during deployment; do **not** refit a new scaler on deployment data.

---

## Data preprocessing requirements

Before using the inference script, new sensor data should be prepared to match the training pipeline.

1. **Time alignment**
   - Data should be aligned to a uniform **30 Hz** sampling rate.
   - Fibion data collected at a different native sampling rate should be resampled/aligned to 30 Hz before inference.
   - The validation pipeline used nearest-neighbour alignment for Fibion streams when aligning with ActiGraph timestamps.

2. **Column names**
   - The input CSV must contain the exact column names required by the selected configuration.
   - Column order is handled by the script, but column names must match.

3. **Scaling**
   - The script applies the saved `StandardScaler` from the `scalers/` directory.
   - Do not fit a new scaler on external/deployment data.

4. **Windowing**
   - The script segments continuous data into non-overlapping 3-s windows.
   - Any remaining rows shorter than a full 3-s window are ignored.

---

## Environment

A typical environment is:

```bash
pip install tensorflow scikit-learn numpy pandas
```

The models were developed using TensorFlow/Keras and scikit-learn. For best compatibility, use a TensorFlow/Keras version close to the version used to save the `.keras` model files.

---

## Running inference

Example for the combined wrist + thigh model:

```bash
python template_inference.py \
  --config combined \
  --input_csv path/to/new_child_30hz_aligned.csv \
  --output_csv outputs/new_child_predictions_combined.csv
```

Example for the thigh-only Fibion model:

```bash
python template_inference.py \
  --config fibion \
  --input_csv path/to/new_child_30hz_aligned.csv \
  --output_csv outputs/new_child_predictions_fibion.csv
```

Example for the wrist-only ActiGraph model:

```bash
python template_inference.py \
  --config actigraph \
  --input_csv path/to/new_child_30hz_aligned.csv \
  --output_csv outputs/new_child_predictions_actigraph.csv
```

The output CSV contains one row per 3-s window, including:

- `window_id`
- `start_row`
- `end_row_exclusive`
- `predicted_class`
- `predicted_activity`
- `prob_SB`
- `prob_LPA`
- `prob_MVPA`

---

## Important notes

- The inference script assumes that each input CSV represents one continuous, already-aligned time series.
- The script does not perform raw `.gt3x` extraction, sensor synchronisation, or Fibion cloud export processing.
- For external datasets, ensure that wear-site placement and preprocessing are as close as possible to the validation protocol.
- The final deployment models do not use age, sex, height, or weight metadata.

---

## License

This framework, trained weights, scaling parameters, and template inference scripts are distributed under the MIT License. Please cite the parent manuscript when using these models in future research.
