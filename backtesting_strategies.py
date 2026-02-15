# BACKTEST STRATEGIE MARKOWITZ PURE (SANS CONTRAINTE DE POIDS MAXIMAL)

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Données 
tickers = ['ADYEN.AS', 'VALE', 'SHOP', 'ICLN', 'JPM']
benchmark = 'SPY'
all_tickers = tickers + [benchmark]

start_date = '2018-01-01'
end_date = '2023-12-31'
risk_free_rate = 0.02 / 252  # Taux journalier

prices = yf.download(all_tickers, start=start_date, end=end_date)['Close'].dropna()
returns = prices.pct_change().dropna()

# Optimisation de portefeuille 
def optimize_portfolio(mu, cov, risk_free_rate):
    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return - (port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1.0)] * len(mu)  
    init_guess = np.ones(len(mu)) / len(mu)

    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Tracé de la frontière efficiente + point de Sharpe max 
def plot_efficient_frontier(mu, cov, risk_free_rate):
    target_returns = np.linspace(min(mu), max(mu), 50)
    vols, rets = [], []
    sharpe_max = -np.inf
    optimal_w, optimal_vol, optimal_ret = None, None, None

    for r in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - r},
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1.0)] * len(mu)  # Suppression de la contrainte de max 50%
        init_guess = np.ones(len(mu)) / len(mu)

        result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
                          init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            vol = np.sqrt(np.dot(result.x.T, np.dot(cov, result.x)))
            vols.append(vol)
            rets.append(r)
            sharpe = (r - risk_free_rate) / vol
            if sharpe > sharpe_max:
                sharpe_max = sharpe
                optimal_w = result.x
                optimal_vol = vol
                optimal_ret = r

    plt.figure(figsize=(10,6))
    plt.plot(vols, rets, label="Frontière efficiente")
    plt.scatter(optimal_vol, optimal_ret, c='red', marker='*', s=200, label='Portefeuille optimal')
    plt.xlabel("Volatilité")
    plt.ylabel("Rendement attendu")
    plt.title("Frontière efficiente (dernière fenêtre)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Backtest 
window_size = 63  # 3 mois
step_size = 63

dates = []
port_values = []
spy_values = []
weights_history = []
capital, spy_capital = 1.0, 1.0

for i in range(0, len(returns) - 2 * window_size, step_size):
    train = returns[tickers].iloc[i:i + window_size]
    test = returns[tickers + [benchmark]].iloc[i + window_size:i + 2 * window_size]

    mu = train.mean()
    cov = train.cov()
    weights = optimize_portfolio(mu, cov, risk_free_rate)
    weights_history.append((test.index[0], weights))

    for date, row in test.iterrows():
        capital *= (1 + np.dot(weights, row[tickers]))
        spy_capital *= (1 + row[benchmark])
        port_values.append(capital)
        spy_values.append(spy_capital)
        dates.append(date)

# Courbes de performance 
df_results = pd.DataFrame({
    'Date': dates,
    'Portefeuille_Markowitz': port_values,
    'SPY': spy_values
}).set_index('Date')

plt.figure(figsize=(12, 6))
df_results.plot(title="Stratégie Markowitz PURE vs S&P 500", grid=True)
plt.ylabel("Valeur du portefeuille (capital cumulé)")
plt.xlabel("Date")
plt.show()

#  Frontière efficiente finale
mu_final = returns[tickers].iloc[-window_size:].mean()
cov_final = returns[tickers].iloc[-window_size:].cov()
plot_efficient_frontier(mu_final, cov_final, risk_free_rate)

# Interprétation simple
def print_performance(df):
    perf = df.pct_change().dropna()
    ann_return = (df.iloc[-1] / df.iloc[0]) ** (252 / len(df)) - 1
    ann_vol = perf.std() * np.sqrt(252)
    sharpe = (ann_return - 0.02) / ann_vol
    max_dd = (df / df.cummax() - 1).min()

    print("\n--- Performances comparées ---")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Rendement annuel moyen: {ann_return[col]:.2%}")
        print(f"  Volatilité annuelle: {ann_vol[col]:.2%}")
        print(f"  Ratio de Sharpe: {sharpe[col]:.2f}")
        print(f"  Max Drawdown: {max_dd[col]:.2%}")

print_performance(df_results)

# Visualisation de l'évolution des poids du portefeuille
weights_df = pd.DataFrame({date: w for date, w in weights_history}, index=tickers).T
weights_df.plot(marker='o', figsize=(12,6), title="Évolution des pondérations du portefeuille")
plt.ylabel("Poids")
plt.xlabel("Date")
plt.grid(True)
plt.legend(title="Actifs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
