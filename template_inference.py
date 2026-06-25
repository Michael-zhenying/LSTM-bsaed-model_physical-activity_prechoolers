"""
Template inference script for fixed-setting Bi-LSTM preschool activity models.

This script applies the final deployment models from:
"Comparing Deep Learning and Cut-Points Methods for Preschooler Physical
Activity and Sedentary Behaviour Classification from Thigh, Wrist, and
Multi-Site Accelerometers".

Final model setting:
- Fixed 3-s non-overlapping windows at 30 Hz (90 samples/window)
- No demographic metadata input
- StandardScaler fitted during final model training; do NOT refit on new data
- Activity classes: SB, LPA, MVPA

Expected feature columns:
- actigraph: wrist acceleration only, 3 channels
- fibion: thigh acceleration + thigh inclination, 6 channels
- combined: wrist acceleration + thigh acceleration + thigh inclination, 9 channels
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# -----------------------------------------------------------------------------
# Fixed deployment configuration
# -----------------------------------------------------------------------------
SAMPLING_RATE = 30
WINDOW_SECONDS = 3
WINDOW_SIZE = WINDOW_SECONDS * SAMPLING_RATE  # 90 samples
STRIDE = WINDOW_SIZE  # non-overlapping windows
ACTIVITY_NAMES = ["SB", "LPA", "MVPA"]

ACTIGRAPH_COLS = [
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
]

FIBION_ACCEL_COLS = ["x", "y", "z"]
FIBION_INCLINATION_COLS = [
    "angle/x/scalar",
    "angle/y/scalar",
    "angle/z/scalar",
]

FEATURE_COLUMNS: Dict[str, List[str]] = {
    "actigraph": ACTIGRAPH_COLS,
    "fibion": FIBION_ACCEL_COLS + FIBION_INCLINATION_COLS,
    "combined": ACTIGRAPH_COLS + FIBION_ACCEL_COLS + FIBION_INCLINATION_COLS,
}


# -----------------------------------------------------------------------------
# Loading and validation utilities
# -----------------------------------------------------------------------------
def get_feature_columns(config_name: str) -> List[str]:
    """Return ordered feature columns for the selected model configuration."""
    config_name = config_name.lower()
    if config_name not in FEATURE_COLUMNS:
        raise ValueError(
            f"Unknown config_name='{config_name}'. Expected one of: "
            f"{list(FEATURE_COLUMNS)}"
        )
    return FEATURE_COLUMNS[config_name]


def load_deployment_pipeline(
    config_name: str,
    models_dir: str | Path = "models",
    scalers_dir: str | Path = "scalers",
):
    """Load the trained Keras model and its corresponding pre-fitted scaler."""
    config_name = config_name.lower()
    model_path = Path(models_dir) / f"final_model_{config_name}.keras"
    scaler_path = Path(scalers_dir) / f"scaler_{config_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    # compile=False avoids requiring the original training loss/optimizer state for inference.
    model = load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Raise a clear error if any model input columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "The input CSV is missing required columns for this configuration:\n"
            f"{missing}\n\nAvailable columns are:\n{list(df.columns)}"
        )


# -----------------------------------------------------------------------------
# Preprocessing and inference
# -----------------------------------------------------------------------------
def create_non_overlapping_windows(
    data_2d: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Segment a continuous feature matrix into non-overlapping windows.

    Parameters
    ----------
    data_2d:
        Array with shape (n_samples, n_channels), already scaled.
    window_size:
        Number of samples per window. For the final model, 90 samples = 3 s x 30 Hz.
    stride:
        Window stride. For the final model, stride equals window_size.

    Returns
    -------
    X:
        Array with shape (n_windows, window_size, n_channels), dtype float32.
    window_index:
        DataFrame containing start/end row indices for each window.
    """
    n_samples = data_2d.shape[0]
    windows = []
    rows = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data_2d[start:end])
        rows.append({"window_id": len(rows), "start_row": start, "end_row_exclusive": end})

    if not windows:
        raise ValueError(
            f"Not enough rows to form one {WINDOW_SECONDS}-s window. "
            f"Need at least {window_size} rows, but found {n_samples}."
        )

    X = np.asarray(windows, dtype=np.float32)
    window_index = pd.DataFrame(rows)
    return X, window_index


def preprocess_new_data(
    raw_csv_path: str | Path,
    config_name: str,
    scaler,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load a new participant/session CSV, apply the saved scaler, and create 3-s windows.

    Important: the input CSV should already be time-aligned to 30 Hz. For Fibion data
    collected at 12.5 Hz, resample/alignment should be performed before using this script.
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {raw_csv_path}")

    feature_cols = get_feature_columns(config_name)
    df = pd.read_csv(raw_csv_path)
    validate_required_columns(df, feature_cols)

    # Keep only model features in the exact order used in training.
    feature_df = df[feature_cols].copy()

    # Convert to numeric and stop if missing/non-numeric values remain.
    for col in feature_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    if feature_df.isna().any().any():
        bad_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values detected in model input columns. "
            f"Please clean/interpolate these columns before inference: {bad_cols}"
        )

    # Use the pre-fitted training scaler. Do not refit on deployment data.
    scaled = scaler.transform(feature_df).astype(np.float32)
    X_new, window_index = create_non_overlapping_windows(scaled)
    return X_new, window_index


def predict_activity(
    raw_csv_path: str | Path,
    config_name: str,
    models_dir: str | Path = "models",
    scalers_dir: str | Path = "scalers",
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run inference and return one row per 3-s window."""
    config_name = config_name.lower()
    model, scaler = load_deployment_pipeline(config_name, models_dir, scalers_dir)
    X_new, window_index = preprocess_new_data(raw_csv_path, config_name, scaler)

    expected_timesteps = model.input_shape[1]
    expected_channels = model.input_shape[2]
    if X_new.shape[1] != expected_timesteps or X_new.shape[2] != expected_channels:
        raise ValueError(
            "Model input shape mismatch. "
            f"Prepared data shape is {X_new.shape[1:]}, but model expects "
            f"({expected_timesteps}, {expected_channels})."
        )

    probabilities = model.predict(X_new, verbose=0)
    predicted_class = np.argmax(probabilities, axis=1)

    results = window_index.copy()
    results["config_name"] = config_name
    results["window_seconds"] = WINDOW_SECONDS
    results["predicted_class"] = predicted_class
    results["predicted_activity"] = [ACTIVITY_NAMES[i] for i in predicted_class]

    for idx, activity in enumerate(ACTIVITY_NAMES):
        results[f"prob_{activity}"] = probabilities[:, idx]

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv_path, index=False)

    return results


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply fixed 3-s Bi-LSTM activity classifiers to new accelerometer data."
    )
    parser.add_argument(
        "--config",
        default="combined",
        choices=["combined", "actigraph", "fibion"],
        help="Model configuration to use.",
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to a 30-Hz time-aligned input CSV for one participant/session.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save window-level predictions.",
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Directory containing final_model_<config>.keras files.",
    )
    parser.add_argument(
        "--scalers_dir",
        default="scalers",
        help="Directory containing scaler_<config>.pkl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cols = get_feature_columns(args.config)
    print(f"Configuration: {args.config}")
    print(f"Expected columns ({len(cols)}): {cols}")
    print(f"Window setting: {WINDOW_SECONDS} s, {WINDOW_SIZE} samples, no overlap")

    results = predict_activity(
        raw_csv_path=args.input_csv,
        config_name=args.config,
        models_dir=args.models_dir,
        scalers_dir=args.scalers_dir,
        output_csv_path=args.output_csv,
    )

    print(f"Inference complete. Windows predicted: {len(results)}")
    print(results["predicted_activity"].value_counts().to_string())
    if args.output_csv:
        print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
