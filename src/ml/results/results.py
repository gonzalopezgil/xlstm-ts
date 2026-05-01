# src/ml/results/results.py

import pandas as pd
import matplotlib.pyplot as plt

DATA_TYPE_COLNAME = 'data_type'

def display_metrics(metrics_accumulator, metrics_accumulator_denoised):
    # Convert metrics dictionaries to DataFrames
    results_df = pd.DataFrame.from_dict(metrics_accumulator, orient="index")
    results_df_denoised = pd.DataFrame.from_dict(metrics_accumulator_denoised, orient="index")

    # Add a column to indicate the data type
    results_df[DATA_TYPE_COLNAME] = 0 # Original Data
    results_df_denoised[DATA_TYPE_COLNAME] = 1 # Causal denoised data

    # Combine the DataFrames
    combined_df = pd.concat([results_df, results_df_denoised])

    # Move DATA_TYPE_COLNAME to the first column
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index(DATA_TYPE_COLNAME)))
    combined_df = combined_df[cols]

    # Separate the data for plotting
    original_df = combined_df[combined_df[DATA_TYPE_COLNAME] == 0]
    denoised_df = combined_df[combined_df[DATA_TYPE_COLNAME] == 1]

    # Create bar charts for RMSSE and 'Direction Accuracy'
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    bar_width = 0.35
    index = range(len(original_df))

    metric_chosen = 'RMSSE'

    if metric_chosen in combined_df.columns:
        axes[0].bar(index, original_df[metric_chosen], bar_width, label='Original Data', color='blue')
        axes[0].bar([i + bar_width for i in index], denoised_df[metric_chosen], bar_width, label='Causal Denoised Data', color='orange')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel(f'{metric_chosen} value')
        axes[0].set_title(f'{metric_chosen} for Models')
        axes[0].set_xticks([i + bar_width / 2 for i in index])
        axes[0].set_xticklabels(original_df.index, rotation=45, ha="right")
        axes[0].legend()

    axes[1].bar(index, original_df['Test Accuracy'], bar_width, label='Original Data', color='blue')
    axes[1].bar([i + bar_width for i in index], denoised_df['Test Accuracy'], bar_width, label='Causal Denoised Data', color='orange')
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Direction Accuracy (%)')
    axes[1].set_title('Direction Accuracy for Models')
    axes[1].set_xticks([i + bar_width / 2 for i in index])
    axes[1].set_xticklabels(original_df.index, rotation=45, ha="right")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return combined_df

# Function to add percentage signs to appropriate columns
def _format_percentage(df):
    # Columns that should be formatted as percentages
    percentage_columns = ['Test Accuracy', 'Train Accuracy', 'Validation Accuracy', 'MAPE', 'Recall', 'Precision (Rise)', 'Precision (Fall)', 'F1 Score']
    for column in percentage_columns:
        df[column] = df[column].apply(lambda x: f"{x:.2f}%")
    return df

def show_results(results, data_type):
    if data_type == 'Original':
        data_type_code = 0
    elif data_type in ('Denoised', 'Causal Denoised'):
        data_type_code = 1

    final_results_sorted = results.sort_values(by=DATA_TYPE_COLNAME, ascending=False)
    
    data_results = final_results_sorted[final_results_sorted[DATA_TYPE_COLNAME] == data_type_code].drop(columns=[DATA_TYPE_COLNAME]).round(2)

    # Format the original and denoised data
    data = _format_percentage(data_results)

    return data

def save_results(results_original, results_denoised, dataset_name):
    # Save DataFrame as CSV
    csv_filename = f'xlstm_predictions_{dataset_name}.csv'
    results_original.to_csv(csv_filename, index=False)

    # Save DataFrame as CSV
    csv_filename = f'xlstm_predictions_{dataset_name}_denoised.csv'
    results_denoised.to_csv(csv_filename, index=False)
