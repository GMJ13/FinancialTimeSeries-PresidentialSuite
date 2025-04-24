import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

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

for file_path in file_paths:
    print(f"\nProcessing file: {file_path}")

    # Load data
    try:
        data = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        continue

    # Compute log returns
    returns = np.log(data / data.shift(1)).dropna()
    series_name = file_path.split('.')[0]
    returns.columns = [series_name]
    y = returns[series_name].values

    # GARCH(1,1)
    try:
        garch11 = arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
        res_garch11 = garch11.fit(disp='off')
        print(f"\nGARCH(1,1) Results for {series_name}")
        print(res_garch11.summary())
    except Exception as e:
        print(f"GARCH(1,1) failed for {series_name}: {e}")

    # GARCH-t
    try:
        garch_t = arch_model(y, mean='Constant', vol='Garch', p=1, q=1, dist='t')
        res_garch_t = garch_t.fit(disp='off')
        print(f"\nGARCH-t Results for {series_name}")
        print(res_garch_t.summary())
    except Exception as e:
        print(f"GARCH-t failed for {series_name}: {e}")

    # GARCH-MA
    try:
        ma1 = ARIMA(y, order=(0, 0, 1)).fit()
        u = ma1.resid
        garch_ma = arch_model(u, mean='Zero', vol='Garch', p=1, q=1, dist='normal')
        res_garch_ma = garch_ma.fit(disp='off')
        print(f"\nGARCH-MA Results for {series_name}")
        print(res_garch_ma.summary())
    except Exception as e:
        print(f"GARCH-MA failed for {series_name}: {e}")

    # GJR-GARCH
    try:
        gjr = arch_model(y, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='normal')
        res_gjr = gjr.fit(disp='off')
        print(f"\nGJR-GARCH Results for {series_name}")
        print(res_gjr.summary())
    except Exception as e:
        print(f"GJR-GARCH failed for {series_name}: {e}")
