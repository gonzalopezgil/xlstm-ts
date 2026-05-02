# src/ml/models/shared/visualisation.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def _is_darts_timeseries(value):
    try:
        from darts import TimeSeries
    except ModuleNotFoundError:
        return False

    return isinstance(value, TimeSeries)

def visualise(actual, prediction, stock, model_name, data_type, show_complete=True, dates=None):
    if _is_darts_timeseries(actual):
        dates = actual.time_index
        actual = actual.values().flatten()
    if _is_darts_timeseries(prediction):
        prediction = prediction.values().flatten()

    title = f"{model_name} Predictions (Trained with "
    if data_type == "Original":
        title += "Original Data)"
    elif data_type in {"Denoised", "Causal Denoised", "Causal Denoised Features"}:
        title += "Causal Denoised Data)"
    else:
        title = f"{model_name} Predictions"

    if not show_complete:
        sample = 30
        dates = dates[:sample]
        actual = actual[:sample]
        prediction = prediction[:sample]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Original Data', color='black', linestyle='-', linewidth=2)
    plt.plot(dates, prediction, label='1-day Forecast', color='grey', linestyle='--', linewidth=2)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=17)
    plt.xlabel('Date', fontsize=17, weight='bold')
    plt.ylabel(f'{stock} Close Price', fontsize=17, weight='bold')
    plt.title(title, fontsize=15)

    plt.grid(False)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes, cbar=False)

    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.55, f'\n({cm_norm[i, j]:.2%})',
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='black',
                     fontsize=9)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion matrix')
    plt.show()
