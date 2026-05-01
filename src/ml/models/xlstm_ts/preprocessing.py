# src/ml/xlstm_ts/preprocessing.py

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from ml.constants import SEQ_LENGTH_XLSTM
from ml.utils.visualisation import plot_data_split

# -------------------------------------------------------------------------------------------
# Normalise data
# -------------------------------------------------------------------------------------------

def normalise_data_xlstm(data):
  """Scale data with a newly fitted MinMaxScaler.

  Use this only with training data. For train/validation/test workflows, prefer
  normalise_split_data_xlstm so the scaler is fitted on the training split only.
  """
  scaler = MinMaxScaler(feature_range=(0, 1))
  return scaler.fit_transform(data.reshape(-1, 1)), scaler

def inverse_normalise_data_xlstm(data, scaler):
  if hasattr(data, "detach"):
    data = data.detach().cpu().numpy()
  return scaler.inverse_transform(np.asarray(data).reshape(-1, 1))

def _to_numpy(data):
  if hasattr(data, "detach"):
    return data.detach().cpu().numpy()
  return np.asarray(data)

def _scale_like(data, scaler):
  original_shape = data.shape
  scaled = scaler.transform(_to_numpy(data).reshape(-1, 1)).reshape(original_shape)

  if hasattr(data, "detach"):
    return torch.from_numpy(scaled).float().to(data.device)

  return scaled

def fit_scaler_on_train_xlstm(train_x, train_y=None):
  """Fit a MinMaxScaler using only training split values."""
  values = [_to_numpy(train_x).reshape(-1, 1)]
  if train_y is not None:
    values.append(_to_numpy(train_y).reshape(-1, 1))

  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler.fit(np.vstack(values))
  return scaler

def normalise_split_data_xlstm(train_x, train_y, val_x, val_y, test_x, test_y, scaler=None):
  """Scale xLSTM train/validation/test splits without fitting on future data."""
  if scaler is None:
    scaler = fit_scaler_on_train_xlstm(train_x, train_y)

  return (
    _scale_like(train_x, scaler),
    _scale_like(train_y, scaler),
    _scale_like(val_x, scaler),
    _scale_like(val_y, scaler),
    _scale_like(test_x, scaler),
    _scale_like(test_y, scaler),
    scaler,
  )

def fit_feature_target_scalers_on_train_xlstm(train_x, train_y):
  """Fit separate feature and target scalers using training split values only."""
  x_scaler = MinMaxScaler(feature_range=(0, 1))
  y_scaler = MinMaxScaler(feature_range=(0, 1))
  x_scaler.fit(_to_numpy(train_x).reshape(-1, 1))
  y_scaler.fit(_to_numpy(train_y).reshape(-1, 1))
  return x_scaler, y_scaler

def normalise_feature_target_split_data_xlstm(
  train_x,
  train_y,
  val_x,
  val_y,
  test_x,
  test_y,
  x_scaler=None,
  y_scaler=None,
):
  """Scale feature sequences and targets with separate train-only scalers."""
  if x_scaler is None or y_scaler is None:
    x_scaler, y_scaler = fit_feature_target_scalers_on_train_xlstm(train_x, train_y)

  return (
    _scale_like(train_x, x_scaler),
    _scale_like(train_y, y_scaler),
    _scale_like(val_x, x_scaler),
    _scale_like(val_y, y_scaler),
    _scale_like(test_x, x_scaler),
    _scale_like(test_y, y_scaler),
    x_scaler,
    y_scaler,
  )

# -------------------------------------------------------------------------------------------
# Create sequences
# -------------------------------------------------------------------------------------------

# Function to create sequences
def create_sequences(data, dates):
    xs, ys, date_list = [], [], []

    for i in range(len(data) - SEQ_LENGTH_XLSTM):
        x = data[i:i + SEQ_LENGTH_XLSTM]
        y = data[i + SEQ_LENGTH_XLSTM]
        date = dates[i + SEQ_LENGTH_XLSTM]
        xs.append(x)
        ys.append(y)
        date_list.append(date)

    X = np.array(xs)
    y = np.array(ys)
    dates = pd.Series(date_list)

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    return X, y, dates

def create_feature_target_sequences(features, target, dates):
    xs, ys, date_list = [], [], []
    features = np.asarray(features)
    target = np.asarray(target)

    if len(features) != len(target) or len(features) != len(dates):
        raise ValueError("features, target, and dates must have the same length")

    for i in range(len(features) - SEQ_LENGTH_XLSTM):
        x = features[i:i + SEQ_LENGTH_XLSTM]
        y = target[i + SEQ_LENGTH_XLSTM]
        date = dates[i + SEQ_LENGTH_XLSTM]
        xs.append(x)
        ys.append(y)
        date_list.append(date)

    X = np.array(xs)
    y = np.array(ys)
    dates = pd.Series(date_list)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    return X, y, dates

# -------------------------------------------------------------------------------------------
# Train, Validation and Test split
# -------------------------------------------------------------------------------------------

def _default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def _move_to_device(data, device):
    if hasattr(data, "to"):
        return data.to(device)
    return data

def _split_data(x, y, dates, set, train_end_date, val_end_date, device=None):
    if set == 'train':
        mask = (dates < train_end_date)
    elif set == 'val':
        mask = (dates >= train_end_date) & (dates < val_end_date)
    elif set == 'test':
        mask = (dates >= val_end_date)
    else:
        raise ValueError("Invalid set name. Must be 'train', 'val', or 'test'.")

    device = device or _default_device()
    x_splitted = _move_to_device(x[mask], device)
    y_splitted = _move_to_device(y[mask], device)

    print(f"{set} X shape: {x_splitted.shape}")
    print(f"{set} y shape: {y_splitted.shape}")

    return x_splitted, y_splitted, dates[mask]

def split_train_val_test_xlstm(x, y, dates, train_end_date, val_end_date, scaler=None, stock=None, device=None):
    train_x, train_y, train_dates = _split_data(x, y, dates, 'train', train_end_date, val_end_date, device)
    val_x, val_y, val_dates = _split_data(x, y, dates, 'val', train_end_date, val_end_date, device)
    test_x, test_y, test_dates = _split_data(x, y, dates, 'test', train_end_date, val_end_date, device)

    if scaler is not None and stock is not None:
        plot_data_split(train_dates.to_numpy(), inverse_normalise_data_xlstm(train_y, scaler), val_dates.to_numpy(), inverse_normalise_data_xlstm(val_y, scaler), test_dates.to_numpy(), inverse_normalise_data_xlstm(test_y, scaler), stock)

    return train_x, train_y, train_dates, val_x, val_y, val_dates, test_x, test_y, test_dates
