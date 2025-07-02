import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

def get_data_loaders(
    df: pd.DataFrame,
    batch_size: int,
    scaler_dir: str,
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 123
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data into training, validation, and test sets; normalize numeric features;
    save the fitted scaler to `scaler_dir`; and return PyTorch DataLoaders.

    Parameters:
        df : pd.DataFrame
            DataFrame containing features and a 'pm2.5' target column.
        batch_size : int
            Batch size for DataLoaders.
        scaler_dir : str
            Directory where the fitted StandardScaler will be saved.
        val_split : float
            Fraction of data to reserve for validation (default 0.2).
        test_split : float
            Fraction of data to reserve for test (default 0.1).
        random_seed : int
            Random seed for reproducibility (default 123).

    Returns
        train_loader, val_loader, test_loader : tuple[DataLoader]
            DataLoaders for the training, validation, and test sets.
    """
    # Check size of test_split and valsplit
    assert 0 <= test_split < 1, "test_split must be in [0,1)"
    assert 0 <= val_split < 1, "val_split must be in [0,1)"
    assert val_split + test_split < 1, "val_split + test_split must be less than 1"

    df = df.copy()

    # Separate features and target
    label_col = 'PM2.5'
    if label_col not in df.columns:
        raise KeyError(f"Expected target column '{label_col}' in DataFrame")
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])
    y = df[label_col].values

    # 3way split: first test, then val from remaining
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_split,
        random_state=random_seed,
        shuffle=False
    )
    rel_val = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=rel_val,
        random_state=random_seed,
        shuffle=False
    )

    # Fitting scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Saving the scaler
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, 'standard_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info("Saved StandardScaler to %s", scaler_path)

    # logtransforming the data, since it's skewed. (long right tail)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    # Helper to wrap into TensorDataset
    def to_tensor_dataset(X_arr, y_arr):
        return TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32)
        )

    train_ds = to_tensor_dataset(X_train_scaled, y_train_log)
    val_ds = to_tensor_dataset(X_val_scaled, y_val_log)
    test_ds = to_tensor_dataset(X_test_scaled, y_test_log)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
