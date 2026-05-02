#!/usr/bin/env python3
"""Run one xLSTM-TS leakage-safe raw-vs-causal-denoised comparison."""

import argparse
import builtins
import datetime as dt
import inspect
import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", default="sp500_daily")
    parser.add_argument("--stock", default="S&P 500")
    parser.add_argument("--train-end-date", default="2021-01-01")
    parser.add_argument("--val-end-date", default="2022-07-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/content")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def quiet_epoch_prints():
    real_print = builtins.print

    def filtered_print(*args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        if msg.startswith("Epoch [") and "Reducing learning rate" not in msg:
            return
        real_print(*args, **kwargs)

    builtins.print = filtered_print


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_torch_include_paths_compat():
    """Keep xlstm==1.0.3 importable on Colab runtimes with newer PyTorch."""
    import torch.utils.cpp_extension as cpp_extension

    original_include_paths = cpp_extension.include_paths
    signature = inspect.signature(original_include_paths)
    if "cuda" in signature.parameters:
        return

    def include_paths_compat(*args, cuda=False, **kwargs):
        if "device_type" in signature.parameters and not args and "device_type" not in kwargs:
            kwargs["device_type"] = "cuda" if cuda else "cpu"
        return original_include_paths(*args, **kwargs)

    cpp_extension.include_paths = include_paths_compat


def ensure_xlstm_runtime_dependencies():
    missing_packages = []
    apply_torch_include_paths_compat()

    try:
        import xlstm  # noqa: F401
    except ModuleNotFoundError:
        missing_packages.append("xlstm==1.0.3")

    try:
        import torchinfo  # noqa: F401
    except ModuleNotFoundError:
        missing_packages.append("torchinfo==1.8.0")

    try:
        import ninja  # noqa: F401
    except ModuleNotFoundError:
        missing_packages.append("ninja==1.11.1.1")

    try:
        import pywt  # noqa: F401
    except ModuleNotFoundError:
        missing_packages.append("PyWavelets==1.6.0")

    if missing_packages:
        print(
            "Installing missing xLSTM runtime dependencies: "
            + ", ".join(missing_packages),
            flush=True,
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_packages])
        apply_torch_include_paths_compat()


def main():
    args = parse_args()

    os.chdir(REPO_ROOT)
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    ensure_xlstm_runtime_dependencies()
    if args.epochs is not None:
        os.environ["XLSTM_TS_NUM_EPOCHS"] = str(args.epochs)
    if args.patience is not None:
        os.environ["XLSTM_TS_PATIENCE"] = str(args.patience)
    if args.batch_size is not None:
        os.environ["XLSTM_TS_BATCH_SIZE"] = str(args.batch_size)
    device = resolve_device(args.device)
    os.environ["XLSTM_TS_DEVICE"] = device
    if device != "cuda":
        os.environ.setdefault("XLSTM_SLSTM_BACKEND", "vanilla")

    print(f"Using xLSTM-TS device: {device}", flush=True)

    plt.show = lambda *_, **__: None
    quiet_epoch_prints()
    set_seed(args.seed)

    from ml.data.preprocessing import process_dates, wavelet_denoising
    from ml.models.xlstm_ts.preprocessing import (
        create_feature_target_sequences,
        normalise_feature_target_split_data_xlstm,
        split_train_val_test_xlstm,
    )
    import ml.models.xlstm_ts.logic as xlstm_logic
    from ml.models.xlstm_ts.logic import run_xlstm_ts

    xlstm_logic.visualise = lambda *_, **__: None

    train_end_date = dt.datetime.strptime(args.train_end_date, "%Y-%m-%d")
    val_end_date = dt.datetime.strptime(args.val_end_date, "%Y-%m-%d")

    file_path = REPO_ROOT / "data" / "datasets" / f"{args.file_name}.csv"
    df = pd.read_csv(file_path, header=0, index_col="Date")
    date_index = pd.Index(df.index.astype(str)).str.slice(0, 19)
    df.index = pd.to_datetime(date_index)
    df["Close_denoised"] = wavelet_denoising(df["Close"])
    df["Noise"] = df["Close"] - df["Close_denoised"]
    df = process_dates(df)

    X, y, dates = create_feature_target_sequences(
        df["Close"].values.reshape(-1, 1),
        df["Close"].values.reshape(-1, 1),
        df.index,
    )
    X_denoised, y_denoised, dates_denoised = create_feature_target_sequences(
        df["Close_denoised"].values.reshape(-1, 1),
        df["Close"].values.reshape(-1, 1),
        df.index,
    )

    train_X, train_y, _, val_X, val_y, _, test_X, test_y, test_dates = split_train_val_test_xlstm(
        X,
        y,
        dates,
        train_end_date,
        val_end_date,
    )
    train_X, train_y, val_X, val_y, test_X, test_y, _, scaler = normalise_feature_target_split_data_xlstm(
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
    )

    (
        train_X_denoised,
        train_y_denoised,
        _,
        val_X_denoised,
        val_y_denoised,
        _,
        test_X_denoised,
        test_y_denoised,
        test_dates_denoised,
    ) = split_train_val_test_xlstm(
        X_denoised,
        y_denoised,
        dates_denoised,
        train_end_date,
        val_end_date,
    )
    (
        train_X_denoised,
        train_y_denoised,
        val_X_denoised,
        val_y_denoised,
        test_X_denoised,
        test_y_denoised,
        _,
        scaler_denoised,
    ) = normalise_feature_target_split_data_xlstm(
        train_X_denoised,
        train_y_denoised,
        val_X_denoised,
        val_y_denoised,
        test_X_denoised,
        test_y_denoised,
    )

    print("Running xLSTM-TS original raw pipeline...")
    _, metrics = run_xlstm_ts(
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        scaler,
        args.stock,
        "Original",
        test_dates,
    )

    print("Running xLSTM-TS causal-denoised-feature pipeline with raw target...")
    _, metrics_denoised = run_xlstm_ts(
        train_X_denoised,
        train_y_denoised,
        val_X_denoised,
        val_y_denoised,
        test_X_denoised,
        test_y_denoised,
        scaler_denoised,
        args.stock,
        "Causal Denoised Features",
        test_dates_denoised,
    )

    summary = pd.DataFrame.from_dict(
        {
            "Original": metrics,
            "Causal Denoised Features": metrics_denoised,
        },
        orient="index",
    )
    cols = [
        "MAE",
        "RMSE",
        "RMSSE",
        "MAPE",
        "R2",
        "Train Accuracy",
        "Validation Accuracy",
        "Test Accuracy",
        "Recall",
        "Precision (Rise)",
        "Precision (Fall)",
        "F1 Score",
    ]
    result_key = f"XLSTM_ONLY_{args.file_name.upper()}_RESULTS"
    result_text = f"\n{result_key}\n" + summary[cols].round(2).to_string()
    result_csv = summary[cols].to_csv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"xlstm_only_{args.file_name}_results.csv").write_text(result_csv)
    (output_dir / f"xlstm_only_{args.file_name}_results.txt").write_text(
        result_text + f"\n\n{result_key}_CSV\n" + result_csv
    )

    print(result_text)
    print(f"\n{result_key}_CSV")
    print(result_csv)
    old_references = {
        "sp500_daily": (
            "Old cached daily xLSTM references: raw Test Accuracy 49.47%, "
            "leaky offline-denoised Test Accuracy 66.22%."
        ),
        "sp500_hourly": (
            "Old cached hourly xLSTM references: raw Test Accuracy 50.11%, "
            "leaky offline-denoised Test Accuracy 68.05%."
        ),
    }
    if args.file_name in old_references:
        print("\n" + old_references[args.file_name])


if __name__ == "__main__":
    main()
