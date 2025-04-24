import pandas as pd
import numpy as np
import pymc as pm  # PyMC version 4+
import arviz as az
from arch import arch_model
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
from pathlib import Path
import glob

warnings.filterwarnings('ignore')

class SVGARCHBacktest:
    def __init__(self, data_files):
        self.data_files = data_files
        self.results = {}
        
        # Define the save directory with absolute path
        self.save_dir = os.path.abspath('sv_garch_backtest_results')
        print(f"Files will be saved to: {self.save_dir}")
        
        # Define the 2 testing windows
        self.testing_windows = {
            '1_Year_Stress_Testing': {
                'start': '2022-01-01',
                'end': '2022-12-30',
                'description': '1 Year Stress Testing Window'
            },
            '2_Year_Recent': {
                'start': '2023-04-14',
                'end': '2025-04-04',
                'description': '2 Year Recent Window'
            }
        }
    
    def fit_sv_model(self, returns, model_type='sv1'):
        """Fit stochastic volatility model using PyMC (version 4+)."""
        y = returns.values
        T = len(y)
        
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=10)
            phi = pm.Beta('phi', alpha=20, beta=1.5)  # Ensure stationarity
            sigma_h = pm.HalfNormal('sigma_h', sigma=1)
            
            # Initialize volatility process
            h_init = pm.Normal('h_init', mu=0, sigma=1)
            
            # Create AR(1) process for volatility
            h = pm.AR('h', rho=[phi], sigma=sigma_h, shape=T, 
                      init_dist=pm.Normal.dist(mu=0, sigma=1))
            
            if model_type == 'sv1':
                # Normal observation equation
                obs = pm.Normal('obs', mu=0, sigma=pm.math.exp(h/2), observed=y)
            else:  # sv_t
                # Student-t observation equation
                nu = pm.Gamma('nu', alpha=2, beta=0.1)
                obs = pm.StudentT('obs', nu=nu, mu=0, sigma=pm.math.exp(h/2), observed=y)
            
            # Fit model
            trace = pm.sample(1000, tune=500, cores=1, 
                             chains=2,
                             return_inferencedata=True,
                             progressbar=True)
            
        return trace
    
    def generate_mean_reversion_signals(self, prices, window=20, z_score_threshold=2.0):
        """Generate mean-reversion trading signals based on z-scores."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        z_scores = (prices - rolling_mean) / rolling_std
        
        signals = pd.Series(0, index=prices.index)
        signals[z_scores > z_score_threshold] = -1  # Sell signal
        signals[z_scores < -z_score_threshold] = 1  # Buy signal
        
        positions = signals.copy()
        for i in range(1, len(positions)):
            if positions[i] == 0:
                positions[i] = positions[i-1]
        
        return signals, positions
    
    def calculate_strategy_returns(self, prices, positions):
        """Calculate returns for the mean-reversion strategy."""
        price_returns = prices.pct_change()
        strategy_returns = positions.shift(1) * price_returns
        strategy_returns = strategy_returns.dropna()
        return strategy_returns
    
    def calculate_rolling_var(self, returns, window_size=52, confidence_level=0.99):
        """Calculate rolling 1-year (52 weeks) VaR using SV models."""
        var_series = {}
        model_params_series = {}
        
        # Define SV-based VaR models
        models = {
            'SV-Normal': lambda y: arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='normal'),
            'SV-t': lambda y: arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='t'),
            'SV-GJR': lambda y: arch_model(y, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='normal')
        }
        
        for model_name, model_func in models.items():
            rolling_var = []
            dates = []
            model_params_list = []
            
            # Calculate rolling VaR
            for i in range(window_size, len(returns)):
                window_returns = returns.iloc[i-window_size:i]
                
                try:
                    # Fit model
                    model = model_func(window_returns)
                    res = model.fit(disp='off', options={'maxiter': 100})
                    
                    # Store model parameters
                    params_dict = {}
                    for param_name in res.params.index:
                        params_dict[param_name] = res.params[param_name]
                    
                    # Calculate mean return for the window
                    mean_return = window_returns.mean()
                    
                    # Get one-step ahead volatility forecast
                    forecast = res.forecast(horizon=1)
                    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
                    
                    # Calculate VaR with mean component
                    if model_name == 'SV-t':
                        df = res.params['nu']
                        var_99 = -(mean_return + stats.t.ppf(1-confidence_level, df) * volatility_forecast[0])
                    else:
                        var_99 = -(mean_return + stats.norm.ppf(1-confidence_level) * volatility_forecast[0])
                    
                    rolling_var.append(var_99)
                    dates.append(returns.index[i])
                    model_params_list.append(params_dict)
                    
                except Exception as e:
                    rolling_var.append(np.nan)
                    dates.append(returns.index[i])
                    model_params_list.append({})
                
                # Print progress every 52 weeks
                if (i - window_size) % 52 == 0:
                    print(f"Progress {model_name}: {i - window_size}/{len(returns) - window_size}")
            
            var_series[model_name] = pd.Series(rolling_var, index=dates)
            model_params_series[model_name] = pd.DataFrame(model_params_list, index=dates)
        
        return var_series, model_params_series
    
    def plot_returns_vs_var_for_window(self, series_name, strategy_returns, var_series, 
                                     window_name, window_info, save_dir=None):
        """Create a plot of strategy returns vs rolling VaR for a specific window."""
        if save_dir is None:
            save_dir = self.save_dir
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Filter data for the testing window
        start_date = pd.to_datetime(window_info['start'])
        end_date = pd.to_datetime(window_info['end'])
        
        window_returns = strategy_returns[(strategy_returns.index >= start_date) & 
                                        (strategy_returns.index <= end_date)]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot strategy returns
        ax.plot(window_returns.index, window_returns, 'gray', alpha=0.6, label='Strategy Returns')
        
        # Plot rolling VaR for each model
        colors = {'SV-Normal': 'blue', 'SV-t': 'red', 'SV-GJR': 'green'}
        
        for model_name, var_vals in var_series.items():
            # Filter VaR for the testing window
            window_var = var_vals[(var_vals.index >= start_date) & (var_vals.index <= end_date)]
            ax.plot(window_var.index, -window_var, color=colors[model_name], 
                    linewidth=2, label=f'{model_name} VaR (99%)')
            
            # Highlight VaR breaches
            aligned_returns = window_returns.loc[window_var.index]
            breaches = aligned_returns < -window_var
            
            if any(breaches):
                breach_dates = breaches[breaches].index
                breach_values = aligned_returns[breaches]
                ax.scatter(breach_dates, breach_values, color=colors[model_name], 
                          marker='o', s=50, alpha=0.7, label=f'{model_name} Breaches')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_title(f'{series_name} - {window_info["description"]}\nStrategy Returns vs Rolling 1-Year SV-based VaR (99%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns / VaR')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(save_dir, f"{series_name}_{window_name}_returns_vs_rolling_var.png")
        print(f"Saving plot to: {filename}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Verify file was created
        if os.path.exists(filename):
            print(f"Successfully created: {filename}")
        else:
            print(f"Failed to create: {filename}")
        
        # Create a secondary plot showing cumulative breaches
        fig2, ax2 = plt.subplots(figsize=(15, 8))
        
        for model_name, var_vals in var_series.items():
            window_var = var_vals[(var_vals.index >= start_date) & (var_vals.index <= end_date)]
            aligned_returns = window_returns.loc[window_var.index]
            breaches = aligned_returns < -window_var
            cumulative_breaches = breaches.cumsum()
            
            ax2.plot(cumulative_breaches.index, cumulative_breaches, color=colors[model_name], 
                    linewidth=2, label=f'{model_name} Cumulative Breaches')
        
        ax2.set_title(f'{series_name} - {window_info["description"]}\nCumulative SV-based VaR Breaches Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Breaches')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the cumulative breaches plot
        filename = os.path.join(save_dir, f"{series_name}_{window_name}_cumulative_breaches.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    def generate_window_summary_statistics(self, strategy_returns, var_series, window_info):
        """Generate summary statistics for a specific testing window."""
        start_date = pd.to_datetime(window_info['start'])
        end_date = pd.to_datetime(window_info['end'])
        
        window_returns = strategy_returns[(strategy_returns.index >= start_date) & 
                                        (strategy_returns.index <= end_date)]
        
        summary_stats = []
        
        for model_name, var_vals in var_series.items():
            window_var = var_vals[(var_vals.index >= start_date) & (var_vals.index <= end_date)]
            aligned_returns = window_returns.loc[window_var.index]
            breaches = aligned_returns < -window_var
            
            stats_dict = {
                'Model': model_name,
                'Window': window_info['description'],
                'Total_Observations': len(aligned_returns),
                'Number_of_Breaches': breaches.sum(),
                'Breach_Rate': breaches.mean(),
                'Expected_Breach_Rate': 0.01,
                'Average_VaR': window_var.mean(),
                'Std_VaR': window_var.std(),
                'Min_VaR': window_var.min(),
                'Max_VaR': window_var.max(),
                'Average_Return': window_returns.mean(),
                'Std_Return': window_returns.std()
            }
            
            # Calculate average loss given breach
            if breaches.sum() > 0:
                losses_given_breach = aligned_returns[breaches]
                stats_dict['Average_Loss_Given_Breach'] = losses_given_breach.mean()
                stats_dict['Max_Loss_Given_Breach'] = losses_given_breach.min()
            else:
                stats_dict['Average_Loss_Given_Breach'] = np.nan
                stats_dict['Max_Loss_Given_Breach'] = np.nan
            
            summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def run_analysis(self):
        """Run the complete rolling VaR analysis and SV model fitting."""
        # Create directory at the start
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Created directory: {self.save_dir}")
        
        for file_path in self.data_files:
            series_name = Path(file_path).stem.replace('Weekly_', '')
            print(f"\n{'='*50}")
            print(f"Processing {series_name}")
            print(f"{'='*50}")
            
            try:
                # Load data
                data = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
                data = data.set_index(data.columns[0])
                data.index = pd.to_datetime(data.index)
                
                # Get prices and generate strategy for full period
                prices = data.iloc[:, 0]
                
                # Need to ensure we have enough history for 1-year rolling window
                all_returns_start = '2021-01-01'
                prices_with_history = prices[prices.index >= all_returns_start]
                
                if len(prices_with_history) < 52:
                    print(f"Not enough data for {series_name}")
                    continue
                
                # Generate signals and calculate strategy returns
                signals, positions = self.generate_mean_reversion_signals(prices_with_history)
                all_strategy_returns = self.calculate_strategy_returns(prices_with_history, positions)
                
                # Fit SV models on the entire history
                print("Fitting SV models...")
                sv_results = {}
                log_returns = np.log(prices_with_history / prices_with_history.shift(1)).dropna()
                
                # Fit SV1 model
                try:
                    print("Fitting SV1 model...")
                    sv1_trace = self.fit_sv_model(log_returns, model_type='sv1')
                    sv_results['sv1'] = sv1_trace
                    
                    # Save summary statistics
                    summary_df = pm.summary(sv1_trace)
                    summary_df.to_excel(os.path.join(self.save_dir, f"{series_name}_sv1_summary.xlsx"))
                    print("SV1 model fitted successfully")
                except Exception as e:
                    print(f"Error fitting SV1: {e}")
                
                # Fit SV-t model
                try:
                    print("Fitting SV-t model...")
                    svt_trace = self.fit_sv_model(log_returns, model_type='sv_t')
                    sv_results['sv_t'] = svt_trace
                    
                    # Save summary statistics
                    summary_df = pm.summary(svt_trace)
                    summary_df.to_excel(os.path.join(self.save_dir, f"{series_name}_sv_t_summary.xlsx"))
                    print("SV-t model fitted successfully")
                except Exception as e:
                    print(f"Error fitting SV-t: {e}")
                
                # Calculate rolling VaR for the entire period
                print("\nCalculating rolling 1-year SV-based VaR...")
                var_series, model_params_series = self.calculate_rolling_var(all_strategy_returns)
                
                # Process each testing window
                series_results = {}
                for window_name, window_info in self.testing_windows.items():
                    print(f"\nProcessing {window_name}: {window_info['description']}")
                    
                    # Extract data for this window
                    start_date = pd.to_datetime(window_info['start'])
                    end_date = pd.to_datetime(window_info['end'])
                    
                    # Filter strategy returns for the testing window
                    window_returns = all_strategy_returns[(all_strategy_returns.index >= start_date) & 
                                                        (all_strategy_returns.index <= end_date)]
                    
                    # Plot results for this window
                    self.plot_returns_vs_var_for_window(series_name, all_strategy_returns, var_series, 
                                                      window_name, window_info)
                    
                    # Generate summary statistics for this window
                    window_stats = self.generate_window_summary_statistics(all_strategy_returns, 
                                                                         var_series, window_info)
                    
                    # Save window-specific statistics
                    window_stats.to_excel(os.path.join(self.save_dir, f"{series_name}_{window_name}_stats.xlsx"), index=False)
                    
                    series_results[window_name] = {
                        'summary_stats': window_stats,
                        'sv_results': sv_results
                    }
                
                # Store results
                self.results[series_name] = series_results
                
                print(f"Completed analysis for {series_name}")
                
            except Exception as e:
                print(f"Failed to process {series_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # List directory contents at the end
        files_created = os.listdir(self.save_dir)
        print(f"\nFiles in {self.save_dir}:")
        for file in files_created:
            print(f"  - {file}")
    
    def create_comparative_plots(self):
        """Create comparative plots across all series for the testing windows."""
        if not self.results:
            print("No results available for comparative plots")
            return
        
        for window_name in self.testing_windows.keys():
            # Breach rate comparison across models and series for this window
            fig, ax = plt.subplots(figsize=(15, 10))
            
            all_breach_rates = []
            series_names = []
            model_names = []
            
            for series_name, results in self.results.items():
                if window_name in results:
                    summary_stats = results[window_name]['summary_stats']
                    for _, row in summary_stats.iterrows():
                        all_breach_rates.append(row['Breach_Rate'])
                        series_names.append(series_name)
                        model_names.append(row['Model'])
            
            if all_breach_rates:
                # Create DataFrame for plotting
                breach_df = pd.DataFrame({
                    'Series': series_names,
                    'Model': model_names,
                    'Breach_Rate': all_breach_rates
                })
                
                # Pivot for better visualization
                pivot_df = breach_df.pivot(index='Series', columns='Model', values='Breach_Rate')
                
                pivot_df.plot(kind='bar', width=0.8, ax=ax)
                ax.axhline(y=0.01, color='red', linestyle='--', label='Expected 1%')
                ax.set_title(f'SV-based VaR Breach Rates - {self.testing_windows[window_name]["description"]}')
                ax.set_ylabel('Breach Rate')
                ax.set_xlabel('Series')
                ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, f"{window_name}_breach_rate_comparison.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Create overall summary across both windows
        overall_summary = []
        for series_name, results in self.results.items():
            for window_name, window_data in results.items():
                for _, row in window_data['summary_stats'].iterrows():
                    summary_dict = row.to_dict()
                    summary_dict['Series'] = series_name
                    overall_summary.append(summary_dict)
        
        if overall_summary:
            overall_df = pd.DataFrame(overall_summary)
            overall_df.to_excel(os.path.join(self.save_dir, "overall_summary.xlsx"), index=False)

def list_created_files():
    """List all files created in the results directory."""
    save_dir = 'sv_garch_backtest_results'
    
    if os.path.exists(save_dir):
        print(f"\n{'='*50}")
        print(f"Files in {save_dir}:")
        print(f"{'='*50}")
        
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{file} - {file_size/1024:.2f} KB")
    else:
        print(f"Directory {save_dir} does not exist!")
    
    # Also check current working directory
    print(f"\nCurrent working directory: {os.getcwd()}")

# Example usage
if __name__ == "__main__":
    data_files = [
        'Weekly_CushingOKWTI_CrudeOil_Spot.xls',
        'Weekly_HenryHub_NaturalGas_Spot.xls',
        'Weekly_NYHarborConventional_Gasoline_Petroleum_Spot.xls'
    ]
    
    # Create analysis object
    analysis = SVGARCHBacktest(data_files)
    
    # Run the analysis
    analysis.run_analysis()
    
    # Create comparative plots
    analysis.create_comparative_plots()
    
    print("\nAnalysis Complete")
    print("Check the 'sv_garch_backtest_results' directory for all results.")
    
    # List created files
    list_created_files()
