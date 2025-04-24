import pandas as pd
import numpy as np
from pathlib import Path
from cmdstanpy import CmdStanModel

# your list of Excel files
file_paths = [
    'Weekly_CushingOKWTI_CrudeOil_Spot.xls',
    'Weekly_EuropeBrent_CrudeOil_Spot.xls',
    'Weekly_GulfCoast_KeroseneJetFuel_Petroleum_Spot.xls',
    'Weekly_HenryHub_NaturalGas_Spot.xls',
    'Weekly_LA_UltraLowSulfurDiesel_Petroleum_Spot.xls',
    'Weekly_MontBelvieu_TXPropane_Petroleum_Spot.xls',
    'Weekly_NYHarbor2_HeatingOil_Petroleum_Spot.xls',
    'Weekly_NYHarborConventional_Gasoline_Petroleum_Spot.xls',
    'Weekly_USGulfCoastConventional_Gasoline_Petroleum_Spot.xls'
]

sv1_code = """
data {
  int<lower=1> T;
  vector[T] y;
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_h;
  vector[T] h_std;
}
transformed parameters {
  vector[T] h;
  h[1] = mu + h_std[1]*sigma_h/sqrt(1-phi*phi);
  for (t in 2:T)
    h[t] = mu + phi*(h[t-1]-mu) + sigma_h*h_std[t];
}
model {
  // priors
  mu        ~ normal(0, 10);
  phi       ~ normal(0, 0.5);
  sigma_h   ~ normal(0, 2);
  h_std     ~ normal(0, 1);
  
  // observation eq.
  for (t in 1:T)
    y[t] ~ normal(0, exp(0.5*h[t]));
}
"""

# 1) compile all of your .stan files once
models = {
    'sv1': CmdStanModel(stan_file='sv1.stan'),
    'sv_t': CmdStanModel(stan_file='sv_t.stan'),
    'sv_ma': CmdStanModel(stan_file='sv_ma.stan'),
    'sv_l': CmdStanModel(stan_file='sv_l.stan'),
}

# 2) loop through files, fit each model, collect summaries
results = {}

for fp in file_paths:
    name = Path(fp).stem
    df = pd.read_excel(fp, index_col=0, sheet_name=0)
    df.index = pd.to_datetime(df.index)
    
    # assume your Excel has a single price column; 
    # if there are multiple, pick the first:
    price = df.iloc[:, 0]
    
    # log-returns
    y = np.log(price / price.shift(1)).dropna().values
    T = len(y)
    
    print(f'\n=== Fitting all SV models for {name} (T={T}) ===')
    results[name] = {}
    
    for model_key, model in models.items():
        fit = model.sample(
            data={'T': T, 'y': y},
            chains=4, 
            parallel_chains=4
        )
        
        # save the fit object
        results[name][model_key] = fit
        
        # print a tiny summary (you can customize which params you show)
        print(f'\n-- {model_key} --')
        print(fit.summary().loc[['mu','phi','sigma_h']])
