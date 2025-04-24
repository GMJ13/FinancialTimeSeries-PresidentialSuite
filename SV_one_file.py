import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel

# 1) LOAD YOUR DATA
# – adjust filename, sheet and column name as needed
df = pd.read_excel('Weekly_CushingOKWTI_CrudeOil_Spot.xls', sheet_name='Data 1', skiprows=2)
df = df.set_index(df.columns[0])
df.index = pd.to_datetime(df.index)
df.columns = ['WTI']
rets = np.log(df / df.shift(1)).dropna()
y = rets['WTI'].values  # or whatever your column is called
T = len(y)

# helper to write & compile
def make_model(name, code):
    with open(f'{name}.stan', 'w') as f:
        f.write(code)
    return CmdStanModel(stan_file=f'{name}.stan')

# 2) STANDARD SV (AR(1) log‐volatility)
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

sv1 = make_model('sv1', sv1_code)
fit1 = sv1.sample(data={'T': T, 'y': y}, chains=4)
print(fit1.summary())

# 3) SV‐t (t‐distributed measurement noise)
sv_t_code = """
data {
  int<lower=1> T;
  vector[T] y;
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_h;
  real<lower=2> nu;
  vector[T] h_std;
}
transformed parameters {
  vector[T] h;
  h[1] = mu + h_std[1]*sigma_h/sqrt(1-phi*phi);
  for (t in 2:T)
    h[t] = mu + phi*(h[t-1]-mu) + sigma_h*h_std[t];
}
model {
  mu        ~ normal(0, 10);
  phi       ~ normal(0, 0.5);
  sigma_h   ~ normal(0, 2);
  nu        ~ gamma(2, 0.1);
  h_std     ~ normal(0, 1);
  
  for (t in 1:T)
    y[t] ~ student_t(nu, 0, exp(0.5*h[t]));
}
"""

sv_t = make_model('sv_t', sv_t_code)
fit_t = sv_t.sample(data={'T': T, 'y': y}, chains=4)
print(fit_t.summary())

# 4) SV‐MA (MA(1) in the observation noise)
sv_ma_code = """
data {
  int<lower=1> T;
  vector[T] y;
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_h;
  real psi;
  vector[T] h_std;
  vector[T] u;          // innovation series
}
transformed parameters {
  vector[T] h;
  h[1] = mu + h_std[1]*sigma_h/sqrt(1-phi*phi);
  for (t in 2:T)
    h[t] = mu + phi*(h[t-1]-mu) + sigma_h*h_std[t];
}
model {
  mu        ~ normal(0, 10);
  phi       ~ normal(0, 0.5);
  sigma_h   ~ normal(0, 2);
  psi       ~ normal(0, 1);
  h_std     ~ normal(0, 1);
  u         ~ normal(0, 1);

  // MA(1) observation:
  y[1]      ~ normal(mu + u[1], exp(0.5*h[1]));
  for (t in 2:T)
    y[t] ~ normal(mu + u[t] + psi * u[t-1], exp(0.5*h[t]));
}
"""

sv_ma = make_model('sv_ma', sv_ma_code)
fit_ma = sv_ma.sample(data={'T': T, 'y': y}, chains=4)
print(fit_ma.summary())

# 5) SV with leverage (allow corr between ε^y and ε^h)
sv_l_code = """
data {
  int<lower=1> T;
  vector[T] y;
}
parameters {
  real mu;
  real<lower=-1,upper=1> phi;
  real<lower=0> sigma_h;
  real<lower=-1,upper=1> rho;
  vector[T] h_std;
  vector[T] eps_y_std;
}
transformed parameters {
  vector[T] h;
  vector[T] eps_h;
  h[1] = mu + h_std[1] * sigma_h / sqrt(1-phi*phi);
  eps_h[1] = h_std[1] * sigma_h;
  for (t in 2:T) {
    h[t]     = mu + phi*(h[t-1]-mu) + h_std[t]*sigma_h;
    eps_h[t] = h_std[t]*sigma_h;
  }
}
model {
  mu          ~ normal(0, 10);
  phi         ~ normal(0, 0.5);
  sigma_h     ~ normal(0, 2);
  rho         ~ uniform(-1, 1);
  h_std       ~ normal(0, 1);
  eps_y_std   ~ normal(0, 1);
  
  for (t in 1:T) {
    // build bivariate residuals:
    vector[2] tmp;
    tmp[1] = eps_y_std[t];
    tmp[2] = h_std[t];
    tmp ~ multi_normal_cholesky(
      [0,0]', 
      cholesky_decompose([[1, rho],[rho,1]])
    );
    // link to y and h innovations:
    y[t]     ~ normal(mu + eps_y_std[t]*exp(0.5*h[t]), 0);
  }
}
"""

sv_l = make_model('sv_l', sv_l_code)
fit_l = sv_l.sample(data={'T': T, 'y': y}, chains=4)
print(fit_l.summary())
