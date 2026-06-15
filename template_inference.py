import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. User Configurations
CONFIG_NAME = "combined"  # Options: "combined", "actigraph", "fibion"
SAMPLING_RATE = 30
WINDOW_SIZE = 3 * SAMPLING_RATE  # 3-second windows (90 steps)
ACTIVITY_NAMES = ["SB", "LPA", "MVPA"]

def load_deployment_pipeline(config_name):
    """Loads the serialized model graph and its associated scaling vector."""
    model = load_model(f'models/final_model_{config_name}.keras')
    with open(f'scalers/scaler_{config_name}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def preprocess_new_data(raw_csv_path, cols_to_use, scaler):
    """
    TEMPLATE: Place your data loading and alignment steps here.
    Note: Thigh data must be resampled to 30Hz using Nearest-Neighbor alignment.
    """
    # Load raw accelerometer streams
    df = pd.read_csv(raw_csv_path)

    # Apply the pre-fitted training scaler to the new data
    df[cols_to_use] = scaler.transform(df[cols_to_use])

    # Segment continuous data into 3-second (90 steps) windows
    windows_X = []
    # Loop logic goes here to slice data into shapes of: (window_size, n_channels)

    return np.array(windows_X)

def main():
    print(f"Initializing Inference Engine for Setup: {CONFIG_NAME.upper()}")

    # Load pipeline components
    model, scaler = load_deployment_pipeline(CONFIG_NAME)

    # Define columns based on configuration
    if CONFIG_NAME == "actigraph":
        cols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']
    elif CONFIG_NAME == "fibion":
        cols = ['x', 'y', 'z']
    else:
        cols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 'x', 'y', 'z']

    print(f"Pipeline successfully loaded. Ready to receive files matching columns: {cols}")

    # --- Inference Execution Example ---
    # X_new = preprocess_new_data("path_to_new_child_data.csv", cols, scaler)
    # probabilities = model.predict(X_new)
    # predicted_classes = np.argmax(probabilities, axis=1)
    # predicted_activities = [ACTIVITY_NAMES[idx] for idx in predicted_classes]

if __name__ == "__main__":
    main()
