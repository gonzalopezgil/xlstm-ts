# Corrected Result Artifacts

This directory contains small committed result summaries for leakage-safe reruns.

`xlstm_daily_leakage_safe_results.csv` records the daily S&P 500 xLSTM-TS rerun
performed on Google Colab T4 from branch `fix/leakage-safe-denoising` with:

```bash
python -u scripts/run_xlstm_daily_only.py --epochs 20 --patience 5 --batch-size 32 --output-dir /content
```

`xlstm_hourly_leakage_safe_results.csv` records the hourly S&P 500 xLSTM-TS rerun
performed locally on MPS from branch `fix/leakage-safe-denoising` after the
Colab runtime disconnected before producing a CSV:

```bash
python -u scripts/run_xlstm_daily_only.py \
  --file-name sp500_hourly \
  --train-end-date 2023-07-01 \
  --val-end-date 2024-01-01 \
  --epochs 20 \
  --patience 5 \
  --batch-size 32 \
  --output-dir /tmp/xlstm-hourly-final
```

The legacy offline-denoised reference rows are included only for comparison and
must not be cited as live forecasting performance.
