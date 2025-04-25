import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
from scipy import stats

# Create output directory
output_dir = 'residual_analysis_nonstationary'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the non-stationary series and periods from your ADF results
non_stationary_cases = [
    {
        'file': 'Data/Weekly_CushingOKWTI_CrudeOil_Spot.xls',
        'series_name': 'CushingOKWTI_CrudeOil_Spot',
        'period': '1Y_Stress_Testing',
        'start': '2022-01-01',
        'end': '2022-12-30',
        'adf_pvalue': 0.156
    }
]


def compute_log_returns(prices):
    """Compute log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def create_residual_diagnostics(series, series_name, period, adf_pvalue):
    """Create comprehensive residual diagnostic plots for non-stationary series."""
    fig = plt.figure(figsize=(20, 15))

    # 1. Time series plot of log returns
    ax1 = plt.subplot(4, 3, 1)
    plt.plot(series.index, series.values, linewidth=1)
    plt.title(f'Log Returns - {series_name}\n{period} (ADF p-value: {adf_pvalue:.3f})')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.grid(True, alpha=0.3)

    # 4. Distribution analysis
    ax4 = plt.subplot(4, 3, 4)
    sns.histplot(series, bins=30, kde=True, stat='density', alpha=0.7)
    x = np.linspace(series.min(), series.max(), 100)
    norm_density = stats.norm.pdf(x, series.mean(), series.std())
    plt.plot(x, norm_density, 'r--', linewidth=2, label='Normal Distribution')
    plt.title('Distribution of Returns')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()

    # 5. Q-Q Plot
    ax5 = plt.subplot(4, 3, 5)
    stats.probplot(series, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')

    # 7. Rolling mean and variance
    ax7 = plt.subplot(4, 3, 7)
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    plt.plot(series.index, series, alpha=0.3, label='Returns')
    plt.plot(rolling_mean.index, rolling_mean, 'r-', label='12-week Rolling Mean')
    plt.title('Rolling Mean Analysis')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Rolling standard deviation
    ax8 = plt.subplot(4, 3, 8)
    plt.plot(rolling_std.index, rolling_std, 'g-', label='12-week Rolling Std')
    plt.title('Rolling Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9. CUSUM test
    ax9 = plt.subplot(4, 3, 9)
    standardized_returns = (series - series.mean()) / series.std()
    cumsum = np.cumsum(standardized_returns)
    plt.plot(cumsum.index, cumsum.values)
    plt.title('CUSUM Test')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sum')
    plt.grid(True, alpha=0.3)

    # 10. First differences (to check if differencing helps)
    ax10 = plt.subplot(4, 3, 10)
    diff_series = series.diff().dropna()
    plt.plot(diff_series.index, diff_series.values)
    plt.title('First Differences of Log Returns')
    plt.xlabel('Date')
    plt.ylabel('Differenced Returns')
    plt.grid(True, alpha=0.3)

    # 12. Ljung-Box test results
    ax12 = plt.subplot(4, 3, 12)
    lb_results = acorr_ljungbox(series, lags=20, return_df=True)
    plt.plot(lb_results.index, lb_results['lb_pvalue'], 'o-')
    plt.axhline(y=0.05, color='r', linestyle='--', label='5% Significance')
    plt.title('Ljung-Box Test P-values')
    plt.xlabel('Lag')
    plt.ylabel('P-value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{output_dir}/{series_name}_{period}_residual_diagnostics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_differencing_analysis(series, series_name, period):
    """Create analysis plots for differencing transformations."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Original series
    axes[0, 0].plot(series.index, series.values)
    axes[0, 0].set_title(f'Original Log Returns - {series_name}')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Returns')

    # First difference
    diff1 = series.diff().dropna()
    axes[0, 1].plot(diff1.index, diff1.values)
    axes[0, 1].set_title('First Difference')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Differenced Returns')

    # Second difference
    diff2 = series.diff().diff().dropna()
    axes[1, 0].plot(diff2.index, diff2.values)
    axes[1, 0].set_title('Second Difference')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('2nd Differenced Returns')

    # Seasonal difference (if enough data)
    if len(series) > 52:
        seasonal_diff = series.diff(52).dropna()
        axes[1, 1].plot(seasonal_diff.index, seasonal_diff.values)
        axes[1, 1].set_title('Seasonal Difference (52 weeks)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data for seasonal differencing',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Seasonal Differenced Returns')

    # ACF of original
    plot_acf(series, ax=axes[2, 0], lags=30, alpha=0.05)
    axes[2, 0].set_title('ACF of Original Series')

    # ACF of first difference
    plot_acf(diff1, ax=axes[2, 1], lags=30, alpha=0.05)
    axes[2, 1].set_title('ACF of First Difference')

    plt.tight_layout()
    filename = f"{output_dir}/{series_name}_{period}_differencing_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Process each non-stationary case
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

        # Compute log returns
        log_returns = compute_log_returns(period_prices)

        # Create diagnostic plots
        create_residual_diagnostics(log_returns, case['series_name'], case['period'], case['adf_pvalue'])

        # Create differencing analysis
        create_differencing_analysis(log_returns, case['series_name'], case['period'])

        # Additional volatility analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Rolling volatility
        rolling_vol = log_returns.rolling(window=12).std() * np.sqrt(52)  # Annualized
        ax1.plot(rolling_vol.index, rolling_vol.values, label='12-week Rolling Volatility')
        ax1.plot(log_returns.index, np.abs(log_returns), alpha=0.3, label='Absolute Returns')
        ax1.set_title(f'{case["series_name"]} - {case["period"]}: Rolling Volatility Analysis')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volatility / Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative returns to check for trends
        cumulative_returns = (1 + log_returns).cumprod() - 1
        ax2.plot(cumulative_returns.index, cumulative_returns.values)
        ax2.set_title('Cumulative Returns (to check for trends)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Returns')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{case['series_name']}_{case['period']}_volatility_trends.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error processing {case['series_name']} - {case['period']}: {e}")

# Create a summary of transformations needed
summary_data = []
for case in non_stationary_cases:
    summary_data.append({
        'Series': case['series_name'],
        'Period': case['period'],
        'ADF p-value': case['adf_pvalue'],
        'Recommendation': 'Consider differencing or detrending'
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_excel(f'{output_dir}/non_stationary_series_recommendations.xlsx', index=False)

print(f"\nAnalysis complete. Residual diagnostic plots saved in '{output_dir}' directory.")