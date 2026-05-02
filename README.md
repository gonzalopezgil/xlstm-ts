# 🚀 xLSTM-TS: Extended Long-Short Term Memory for Time Series

![Project Logo](./assets/logo.png)

**Authors:** Gonzalo López Gil, Paul Duhamel-Sebline, Andrew McCarren  
*Published in: [An Evaluation of Deep Learning Models for Stock Market Trend Prediction](https://arxiv.org/abs/2408.12408)*

This repository contains the implementation of the **xLSTM-TS model**, a time series-optimised adaptation of the Extended Long Short-Term Memory (xLSTM) architecture proposed by Beck et al. (2024). The xLSTM-TS model modifies the xLSTM framework to make it suitable for time series forecasting. The architecture includes the xLSTM-TS implementation and leakage-safe preprocessing utilities, including causal wavelet denoising. While designed for versatility in time series forecasting, the model has been applied to short-term Stock Market Trend Prediction as a key use case.

In addition to xLSTM-TS, this repository features implementations of several state-of-the-art forecasting models for benchmarking purposes, such as TCN, N-BEATS, TFT, N-HiTS, and TiDE. These models have been evaluated alongside xLSTM-TS in our study. This repository provides datasets and code for the complete workflow, from preprocessing, to model training, and evaluation, along with detailed comparisons of accuracy and trend prediction capabilities.

This is the **official repository** for the paper *"An Evaluation of Deep Learning Models for Stock Market Trend Prediction."*

## Important correction: denoising leakage

[Issue #2](https://github.com/gonzalopezgil/xlstm-ts/issues/2) correctly identified that the original denoised stock-prediction workflow applied wavelet denoising as a full-series offline transform before chronological evaluation. That allowed future observations to influence historical denoised values. The previously reported denoised metrics, cached notebook outputs, and generated prediction CSVs should therefore be treated as offline smoothing results, not valid live forecasting performance.

The source code now makes `wavelet_denoising()` causal by default: the value produced at time `t` is computed only from observations up to and including `t`. The legacy full-series transform is still available as `offline_wavelet_denoising()` for visual/offline analysis, but it emits a warning and should not be used for forecasting features or targets.

The daily S&P 500 xLSTM-TS rerun on this branch does **not** reproduce the old denoised gain. A bounded Colab T4 rerun (`scripts/run_xlstm_daily_only.py --epochs 20 --patience 5 --batch-size 32`) produced:

| Pipeline | Test Accuracy | F1 Score | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Old cached raw xLSTM-TS | 49.47% | 51.78% | 38.25 | 48.22 |
| Current raw xLSTM-TS | 49.20% | 51.40% | 51.65 | 64.39 |
| Old cached offline-denoised xLSTM-TS | 66.22% | 68.33% | 55.84 | 67.83 |
| Current causal-denoised-feature xLSTM-TS | 47.87% | 49.74% | 74.63 | 90.62 |

The corrected result supports the architecture/preprocessing cleanup, but it retracts the original claim that wavelet denoising materially improves live stock-direction prediction.

```bibtex
@misc{gil2024evaluationdeeplearningmodels,
      title={An Evaluation of Deep Learning Models for Stock Market Trend Prediction}, 
      author={Gonzalo Lopez Gil and Paul Duhamel-Sebline and Andrew McCarren},
      year={2024},
      eprint={2408.12408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.12408}, 
}
```

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [📄 Abstract](#-abstract)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📊 Dataset](#-dataset)
- [🔧 Preprocessing](#-preprocessing)
- [🧠 Models](#-models)
- [📈 Results](#-results)
- [🤝 Contributions](#-contributions)
- [📚 References](#-references)

## ✨ Key Features

- **xLSTM-TS Implementation**: An adaptation of the Extended LSTM architecture for time series applications.
- **Causal Wavelet Denoising**: Leakage-safe discrete wavelet denoising for forecasting features. The legacy full-series smoother is explicit and warned.
- **Benchmark Models**: Includes leading deep learning architectures for comparison, such as TCN, N-BEATS, TFT, N-HiTS and TiDE.
- **Comprehensive Evaluation**: Includes metrics such as Accuracy, F1 Score, MAE, RMSE, RMSSE, and MASE.
- **Interactive Notebooks**: Experiment with pre-defined setups or customise parameters to explore your own datasets.
- **Extensible Codebase**: Use the modular code in the src/ml folder for your own projects.

## 📄 Abstract

The stock market is a fundamental component of financial systems, reflecting economic health, providing investment opportunities, and influencing global dynamics. Accurate stock market predictions can lead to significant gains and promote better investment decisions. However, predicting stock market trends is challenging due to their non-linear and stochastic nature.

This study investigates the efficacy of advanced deep learning models for short-term trend forecasting using daily and hourly closing prices from the S&P 500 index and the Brazilian ETF EWZ. The models explored include Temporal Convolutional Networks (TCN), Neural Basis Expansion Analysis for Time Series Forecasting (N-BEATS), Temporal Fusion Transformers (TFT), Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS), and Time-series Dense Encoder (TiDE). Furthermore, we introduce the Extended Long Short-Term Memory for Time Series (xLSTM-TS) model, an xLSTM adaptation optimised for time series prediction.

The original experiments applied wavelet denoising to smooth the signal and reduce minor fluctuations. Those denoised results are now deprecated for live forecasting because the preprocessing was not causal. The xLSTM-TS implementation remains available, but performance claims should be based on raw inputs or on reruns using the corrected causal transform and train-only scaling workflow.

By leveraging advanced deep learning models and effective data preprocessing techniques, this research provides valuable insights into the application of machine learning for market movement forecasting, highlighting both the potential and the challenges involved.

## ⚙️ Installation

To set up the environment and run the code, follow the steps below.

### 🔧 System Requirements

This project requires CUDA for GPU acceleration. Ensure that you have a GPU with a compatible CUDA version installed to accelerate model training and inference. Refer to CUDA Toolkit Documentation for installation instructions.

CUDA is required for GPU acceleration. The project was developed and tested in the following environment:
- **Python**: Version 3.10.12 (in Google Colab).
- **GPU**: NVIDIA T4 GPU (available in Google Colab). This GPU supports CUDA.

If you use Google Colab, ensure that you enable GPU runtime in your notebook settings.

### 🛠️ Clone the Repository

First, clone this repository to your local machine or Google Colab environment:

```bash
git clone https://github.com/gonzalopezgil/xlstm-ts.git
```

Navigate to the repository folder:

```bash
cd xlstm-ts
```

### 📦 Install Dependencies

Install the required Python dependencies:

```bash
pip install -r requirements.txt --quiet
```

### 🗂️ Add the `src` Directory to the Python Path

The project uses a modular structure, and the `src` folder contains all the core code. To ensure the code runs smoothly, you need to add the `src` directory to the Python path. This is necessary for importing utilities and model code into your scripts or notebooks.

Run the following Python code to add the `src` directory to the path:

```python
import sys
import os

# Get the current working directory
current_dir = os.getcwd()

# Construct the path to the 'src' directory
src_path = os.path.join(current_dir, 'src')

# Add the 'src' directory to the Python path
if src_path not in sys.path:
    sys.path.append(src_path)

# Now you can import modules from the 'src' directory
from ml.utils.imports import *  # Example import
```

### 🔑 Optional: Retrieve Additional Hourly Data

If you need hourly stock data beyond 2 years (due to API limitations), you can use the **Tiingo API** to fetch it. Follow these steps:

1. Create a Tiingo account at [Tiingo API](https://api.tiingo.com).
2. Access your API Token at [API Token Account Page](https://api.tiingo.com/account/api/token).
3. Set the token in your environment:

```python
import os
os.environ['TIINGO_API_KEY'] = 'your_tiingo_api_key_here'
```

## 🚀 Usage

### 📓 Using the Jupyter Notebooks

**Note**: A GPU runtime is required for training deep learning models. For example, a T4 GPU in Google Colab works perfectly.

The experiments and examples in this project are provided as Jupyter notebooks under the `notebooks` folder.

To run an experiment:

1. Open a notebook from the `notebooks` folder.
2. Modify the **Constants** section at the top of the notebook if needed:

```python
# Dataset settings
TICKER = '^GSPC'  # S&P 500 index
STOCK = 'S&P 500'

# Date range and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d'  # daily frequency

# Train, validation, test split
TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'
```

3. Run the notebook to train and evaluate models or explore your own datasets.

### 🖥️ Using the Source Code

If you want to integrate xLSTM-TS or other models into your own project:

1.	Explore the `src/ml` folder, which contains all model and workflow implementations, including training, evaluation, and preprocessing scripts.
2.	The `src/gui` folder contains an app under development, not yet complete.

## 📊 Dataset

The datasets used in this study include daily and hourly data for two indices: the S&P 500 and the Brazilian ETF EWZ. These were sourced from the Yahoo Finance and Tiingo APIs, with data spanning the following periods:

- **EWZ Daily**: 14/07/2000 - 29/12/2023
- **S&P 500 Daily**: 03/01/2000 - 29/12/2023
- **EWZ Hourly**: 13/07/2020 - 11/07/2024
- **S&P 500 Hourly**: 13/07/2020 - 11/07/2024

The data fields include Date, High, Low, Close, Adjusted Close, and Volume, with the **Close** price serving as the target variable for trend prediction.

## 🔧 Preprocessing

### 🌀 Noise Reduction

For forecasting experiments, use **causal wavelet denoising** so each denoised value is computed only from data available at that timestamp:

```python
from ml.data.preprocessing import wavelet_denoising

df["Close_denoised"] = wavelet_denoising(df["Close"].values)
df["Noise"] = df["Close"] - df["Close_denoised"]
```

For offline visualization only, the old full-series smoother is available as `offline_wavelet_denoising()`. Do not use it for forecasting features, targets, validation metrics, or test metrics.

### 📐 Data Splitting

Each dataset was divided into training, validation, and test sets to enable robust model evaluation:

- **Daily Data**: Training (86%), Validation (7%), Test (7%)
- **Hourly Data**: Training (75%), Validation (12.5%), Test (12.5%)

For xLSTM-TS, split sequences chronologically before fitting the scaler:

```python
from ml.models.xlstm_ts.preprocessing import (
    create_feature_target_sequences,
    normalise_feature_target_split_data_xlstm,
    split_train_val_test_xlstm,
)

X, y, dates = create_feature_target_sequences(
    df["Close"].values.reshape(-1, 1),
    df["Close"].values.reshape(-1, 1),
    df.index,
)
train_X, train_y, train_dates, val_X, val_y, val_dates, test_X, test_y, test_dates = split_train_val_test_xlstm(
    X,
    y,
    dates,
    TRAIN_END_DATE,
    VAL_END_DATE,
)
train_X, train_y, val_X, val_y, test_X, test_y, feature_scaler, target_scaler = normalise_feature_target_split_data_xlstm(
    train_X,
    train_y,
    val_X,
    val_y,
    test_X,
    test_y,
)
```

For causal-denoised xLSTM experiments, use `Close_denoised` as the feature sequence and raw `Close` as the target sequence. For Darts univariate models, the notebook trains on the causal-denoised series but evaluates predictions against the raw close series; this should be reported as causal-denoised input with raw-price evaluation, not as a denoised target benchmark.

`normalise_data_xlstm()` is retained for small/manual workflows, but it must only be fitted on training data.

## 🧠 Models

The focus of this repository is the xLSTM-TS model, an adaptation of the **Extended Long Short-Term Memory (xLSTM)** architecture, as proposed by Beck et al. (2024), specifically designed for time series applications. To benchmark its performance, we also implemented several well-regarded deep learning models for time series forecasting using the Darts library:

1. **xLSTM-TS (Extended LSTM for Time Series)** - Our proposed model that adapts the xLSTM architecture for time series forecasting.
2. **Temporal Convolutional Network (TCN)** - Uses causal convolutions for capturing temporal dependencies.
3. **Neural Basis Expansion Analysis for Time Series Forecasting (N-BEATS)** - Employs residual connections to capture both short- and long-term trends in time series data.
4. **Temporal Fusion Transformer (TFT)** - Integrates attention mechanisms for interpretability in multi-horizon forecasting.
5. **Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)** - A hierarchical approach optimised for long-horizon forecasts.
6. **Time-series Dense Encoder (TiDE)** - Combines dense encoders with MLPs for efficient predictive modelling.

## 📈 Results

### 📊 Performance Metrics

The models were evaluated using several metrics, including Accuracy, F1 Score, MAE, RMSE, RMSSE, and MASE. The previously reported denoised stock-direction metrics are deprecated because they used offline denoising and should not be cited as live forecasting results. Corrected result tables should distinguish:

- raw input with raw target;
- causal-denoised input with raw-price evaluation;
- offline-denoised analysis, explicitly labelled as non-causal;
- naive baselines such as majority class and previous-return direction.

### Corrected daily xLSTM-TS rerun

The leakage-safe daily S&P 500 xLSTM-TS rerun is stored in [`data/results/xlstm_daily_leakage_safe_results.csv`](data/results/xlstm_daily_leakage_safe_results.csv). It was run on Google Colab T4 from branch `fix/leakage-safe-denoising` with:

```bash
python -u scripts/run_xlstm_daily_only.py --epochs 20 --patience 5 --batch-size 32 --output-dir /content
```

| Pipeline | Test Accuracy | F1 Score | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Current raw xLSTM-TS | 49.20% | 51.40% | 51.65 | 64.39 |
| Current causal-denoised-feature xLSTM-TS | 47.87% | 49.74% | 74.63 | 90.62 |

### 🗝️ Key Findings

- **Wavelet Denoising**: Full-series denoising is useful for offline signal analysis but invalid for live forecasting. Forecasting experiments must use causal denoising or raw data.
- **Model Performance**: xLSTM-TS remains the main architectural contribution. In the corrected daily xLSTM-TS rerun, causal denoising did not improve directional accuracy over the raw pipeline.
- **Timeframe Sensitivity**: Predictions of daily trends generally achieved higher accuracy than hourly trends, likely due to the higher volatility in shorter time frames.

## 🤝 Contributions

- **Gonzalo López Gil** - School of Computing, Dublin City University (Dublin, Ireland)  
  Email: gonzalo.lopezgil2@mail.dcu.ie
- **Paul Duhamel-Sebline** - School of Computing, Dublin City University (Dublin, Ireland)  
  Email: paul.duhamelsebline2@mail.dcu.ie
- **Andrew McCarren** - Insight Centre for Data Analytics, Dublin City University (Dublin, Ireland)  
  Email: andrew.mccarren@dcu.ie

## 📚 References

For citing this work, please use the following references:

### 📝 Our Paper

```bibtex
@misc{gil2024evaluationdeeplearningmodels,
      title={An Evaluation of Deep Learning Models for Stock Market Trend Prediction}, 
      author={Gonzalo Lopez Gil and Paul Duhamel-Sebline and Andrew McCarren},
      year={2024},
      eprint={2408.12408},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.12408}, 
}
```

### 📑 Original xLSTM Paper

```bibtex
@misc{beck2024xlstmextendedlongshortterm,
      title={xLSTM: Extended Long Short-Term Memory}, 
      author={Maximilian Beck and Korbinian Pöppel and Markus Spanring and Andreas Auer and Oleksandra Prudnikova and Michael Kopp and Günter Klambauer and Johannes Brandstetter and Sepp Hochreiter},
      year={2024},
      eprint={2405.04517},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.04517}, 
}
```
