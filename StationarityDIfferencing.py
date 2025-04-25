import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

# Create output directory
output_dir = 'differencing_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the non-stationary series based on your data
non_stationary_cases = [
    {
        'file': 'Data/Weekly_CushingOKWTI_CrudeOil_Spot.xls',
        'series_name': 'CushingOKWTI_CrudeOil_Spot',
        'period': '1Y_Stress_Testing',
        'start': '2022-01-01',
        'end': '2022-12-30',
        'original_adf_pvalue': 0.156
    }
]


def compute_log_returns(prices):
    """Compute log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def perform_test_statistics(series, name):
    """Perform ADF and Ljung-Box tests."""
    # ADF test
    adf_result = adfuller(series.dropna(), regression='c', autolag='AIC')

    # Ljung-Box test (up to lag 20)
    lb_test = acorr_ljungbox(series.dropna(), lags=20, return_df=True)

    return {
        'ADF_Statistic': adf_result[0],
        'ADF_pvalue': adf_result[1],
        'ADF_Lags_Used': adf_result[2],
        'ADF_Observations': adf_result[3],
        'ADF_Critical_1%': adf_result[4]['1%'],
        'ADF_Critical_5%': adf_result[4]['5%'],
        'ADF_Critical_10%': adf_result[4]['10%'],
        'LjungBox_Lag5_Stat': lb_test.loc[5, 'lb_stat'],
        'LjungBox_Lag5_pvalue': lb_test.loc[5, 'lb_pvalue'],
        'LjungBox_Lag10_Stat': lb_test.loc[10, 'lb_stat'],
        'LjungBox_Lag10_pvalue': lb_test.loc[10, 'lb_pvalue'],
        'LjungBox_Lag20_Stat': lb_test.loc[20, 'lb_stat'],
        'LjungBox_Lag20_pvalue': lb_test.loc[20, 'lb_pvalue']
    }


# Results storage
results_data = []

# Process each non-stationary series
for case in non_stationary_cases:
    print(f"\nProcessing: {case['series_name']} - {case['period']}")

    try:
        # Load data
        data = pd.read_excel(case['file'], sheet_name='Data 1', skiprows=2)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index)
        prices = data.iloc[:, 0]

        # Filter for the specific period
        start_date = pd.to_datetime(case['start'])
        end_date = pd.to_datetime(case['end'])
        period_prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]

        # Compute log returns (original series)
        log_returns = compute_log_returns(period_prices)

        # Perform first-order differencing on log returns
        differenced_returns = log_returns.diff().dropna()

        # Get test statistics for original series
        original_stats = perform_test_statistics(log_returns, "Original")

        # Get test statistics for differenced series
        diff_stats = perform_test_statistics(differenced_returns, "Differenced")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot original series
        axes[0, 0].plot(log_returns.index, log_returns.values)
        axes[0, 0].set_title(
            f'Original Log Returns - {case["series_name"]}\nADF p-value: {original_stats["ADF_pvalue"]:.4f}')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Returns')

        # Plot differenced series
        axes[0, 1].plot(differenced_returns.index, differenced_returns.values)
        axes[0, 1].set_title(f'First-Order Differenced Returns\nADF p-value: {diff_stats["ADF_pvalue"]:.4f}')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Differenced Returns')

        # ACF of original series
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(log_returns, ax=axes[1, 0], lags=30, alpha=0.05)
        axes[1, 0].set_title('ACF of Original Series')

        # ACF of differenced series
        plot_acf(differenced_returns, ax=axes[1, 1], lags=30, alpha=0.05)
        axes[1, 1].set_title('ACF of Differenced Series')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{case['series_name']}_{case['period']}_differencing_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Store results
        results_data.append({
            'Series': case['series_name'],
            'Period': case['period'],
            'Original_ADF_pvalue': original_stats['ADF_pvalue'],
            'Original_ADF_Statistic': original_stats['ADF_Statistic'],
            'Original_LB_Lag10_pvalue': original_stats['LjungBox_Lag10_pvalue'],
            'Differenced_ADF_pvalue': diff_stats['ADF_pvalue'],
            'Differenced_ADF_Statistic': diff_stats['ADF_Statistic'],
            'Differenced_LB_Lag10_pvalue': diff_stats['LjungBox_Lag10_pvalue'],
            'Is_Stationary_After_Diff': diff_stats['ADF_pvalue'] < 0.05,
            'Observations': len(differenced_returns)
        })

        # Create Ljung-Box comparison plot
        plt.figure(figsize=(12, 6))

        # Get Ljung-Box p-values for all lags
        lb_test_original = acorr_ljungbox(log_returns.dropna(), lags=20, return_df=True)
        lb_test_diff = acorr_ljungbox(differenced_returns.dropna(), lags=20, return_df=True)

        plt.plot(range(1, 21), lb_test_original['lb_pvalue'], 'o-', label='Original Series')
        plt.plot(range(1, 21), lb_test_diff['lb_pvalue'], 's-', label='Differenced Series')
        plt.axhline(y=0.05, color='r', linestyle='--', label='5% Significance Level')
        plt.title(f'Ljung-Box Test P-values - {case["series_name"]} ({case["period"]})')
        plt.xlabel('Lag')
        plt.ylabel('P-value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{output_dir}/{case['series_name']}_{case['period']}_ljungbox_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error processing {case['series_name']} - {case['period']}: {e}")

# Create summary DataFrame
results_df = pd.DataFrame(results_data)
results_df.to_excel(f'{output_dir}/differencing_test_statistics.xlsx', index=False)

# Create a summary visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ADF p-value comparison
x = range(len(results_df))
width = 0.35
ax1.bar([i - width / 2 for i in x], results_df['Original_ADF_pvalue'], width, label='Original')
ax1.bar([i + width / 2 for i in x], results_df['Differenced_ADF_pvalue'], width, label='Differenced')
ax1.axhline(y=0.05, color='r', linestyle='--', label='5% Significance')
ax1.set_ylabel('ADF p-value')
ax1.set_title('ADF Test P-values: Original vs Differenced Series')
ax1.set_xticks(x)
ax1.set_xticklabels([f"{row['Series']}\n{row['Period']}" for _, row in results_df.iterrows()], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Ljung-Box p-value comparison (Lag 10)
ax2.bar([i - width / 2 for i in x], results_df['Original_LB_Lag10_pvalue'], width, label='Original')
ax2.bar([i + width / 2 for i in x], results_df['Differenced_LB_Lag10_pvalue'], width, label='Differenced')
ax2.axhline(y=0.05, color='r', linestyle='--', label='5% Significance')
ax2.set_ylabel('Ljung-Box p-value (Lag 10)')
ax2.set_title('Ljung-Box Test P-values (Lag 10): Original vs Differenced Series')
ax2.set_xticks(x)
ax2.set_xticklabels([f"{row['Series']}\n{row['Period']}" for _, row in results_df.iterrows()], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/test_statistics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a visual summary table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
for _, row in results_df.iterrows():
    table_data.append([
        row['Series'],
        row['Period'],
        f"{row['Original_ADF_pvalue']:.4f}",
        f"{row['Differenced_ADF_pvalue']:.4f}",
        "Yes" if row['Is_Stationary_After_Diff'] else "No"
    ])

table = ax.table(cellText=table_data,
                 colLabels=['Series', 'Period', 'Original ADF p-value', 'Differenced ADF p-value',
                            'Stationary After Diff?'],
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

plt.title('First-Order Differencing Results Summary', fontsize=14, pad=20)
plt.savefig(f'{output_dir}/differencing_summary_table.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAnalysis complete. Results saved in '{output_dir}' directory.")
print("\nSummary of differencing results:")
print(results_df[['Series', 'Period', 'Original_ADF_pvalue', 'Differenced_ADF_pvalue', 'Is_Stationary_After_Diff']])