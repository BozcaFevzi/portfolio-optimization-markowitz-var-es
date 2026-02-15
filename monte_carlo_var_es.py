import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Téléchargement des données
tickers = [
    'AAPL',        # Apple - USA, Technologie
    'TM',          # Toyota - Japon, Automobile
    'NVS',         # Novartis - Suisse, Pharmaceutique
    'ITUB',        # Itaú Unibanco - Brésil, Banque
]

data = yf.download(tickers, start='2022-01-01', end='2024-12-31')['Close']
returns = data.pct_change().dropna()


# Paramètres
n_assets = len(tickers)
n_simulations = 100000
confidence_level = 0.95
mean_returns = returns.mean()
cov_matrix = returns.cov()


# Fonction rendement et volatilité
def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility


# Simulation Monte Carlo de rendements journaliers
simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_simulations)

# Expected Shortfall
def expected_shortfall(weights):
    portfolio_returns = simulated_returns @ weights
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    es = portfolio_returns[portfolio_returns <= var].mean()
    return -es

# Value at Risk
def value_at_risk(weights):
    portfolio_returns = simulated_returns @ weights
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    return -var

# Variance
def portfolio_variance(weights):
    return np.dot(weights.T, np.dot(cov_matrix, weights))


# Contraintes
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))
initial_guess = np.array([1 / n_assets] * n_assets)

# Optimisation
opt_markowitz = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_es = minimize(expected_shortfall, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_var = minimize(value_at_risk, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

def show_weights(weights, title):
    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

show_weights(opt_markowitz.x, "Répartition Markowitz")
show_weights(opt_es.x, "Répartition Expexted ShortFall")
show_weights(opt_var.x, "Répartition Value at Risk")


def sharpe_ratio(returns, rf_rate):
    excess_returns = returns.mean() - rf_rate
    return (excess_returns * 252) / (returns.std() * np.sqrt(252))

def sortino_ratio(returns, rf_rate):
    downside_returns = returns[returns < rf_rate]
    downside_std = downside_returns.std()
    excess_returns = returns.mean() - rf_rate
    return (excess_returns * 252) / (downside_std * np.sqrt(252))

# Taux sans risque annuel (2%), converti en journalier
rf_rate = 0.02 / 252

# Résultats
def summarize(weights, name):
    port_returns = returns @ weights
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    port_var = value_at_risk(weights)
    port_es = expected_shortfall(weights)
    sharpe = sharpe_ratio(port_returns, rf_rate)
    sortino = sortino_ratio(port_returns, rf_rate)
    return {
        'Portefeuille': name,
        'Rendement espéré (%)': port_return * 252 * 100,
        'Volatilité (%)': port_vol * np.sqrt(252) * 100,
        'VaR 95% (%)': port_var * 100,
        'ES 95% (%)': port_es * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino
    }

summary = pd.DataFrame([
    summarize(opt_markowitz.x, 'Markowitz'),
    summarize(opt_var.x, 'Min VaR'),
    summarize(opt_es.x, 'Min ES')
])

summary.set_index('Portefeuille', inplace=True)
print(summary.round(2))


def plot_density(weights, label, color):
    port_returns = simulated_returns @ weights
    # Densité
    density = gaussian_kde(port_returns)
    xs = np.linspace(np.min(port_returns), np.max(port_returns), 200)
    plt.plot(xs, density(xs), label=f"Densité {label}", color=color, linewidth=2)
    # Histogramme en fond
    plt.hist(port_returns, bins=50, alpha=0.2, density=True, color=color, edgecolor='black')

plt.figure(figsize=(14, 7))

plot_density(opt_markowitz.x, 'Markowitz', 'blue')
plot_density(opt_var.x, 'Min VaR', 'orange')
plot_density(opt_es.x, 'Min ES', 'green')

# Ajout des lignes VaR et ES
for w, col, name in [(opt_markowitz.x, 'blue', 'Markowitz'),
                     (opt_var.x, 'orange', 'Min VaR'),
                     (opt_es.x, 'green', 'Min ES')]:
    plt.axvline(-value_at_risk(w), color=col, linestyle='--', linewidth=2, label=f'VaR {name}')
    plt.axvline(-expected_shortfall(w), color=col, linestyle=':', linewidth=2, label=f'ES {name}')

plt.title('Densités estimées des rendements simulés des portefeuilles optimisés', fontsize=16)
plt.xlabel('Rendement simulé', fontsize=12)
plt.ylabel('Densité', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

## Pour l'etf
etf = yf.download("URTH", start=returns.index[0], end=returns.index[-1])['Close']
etf = etf.reindex(returns.index).ffill()
returns_etf = etf.pct_change().dropna()
cumulative_etf = (1 + returns_etf).cumprod() * 100


weights_mkv = opt_markowitz.x
weights_es = opt_es.x
weights_var = opt_var.x

# Rendements journaliers des portefeuilles
returns_mkv = returns @ weights_mkv
returns_es = returns @ weights_es
returns_var = returns @ weights_var

# Valeur cumulée (base 100)
cumulative_mkv = (1 + returns_mkv).cumprod() * 100
cumulative_es = (1 + returns_es).cumprod() * 100
cumulative_var = (1 + returns_var).cumprod() * 100

# Tracer les performances cumulées
plt.figure(figsize=(12, 6))
plt.plot(cumulative_mkv, label='Markowitz')
plt.plot(cumulative_es, label='Min ES')
plt.plot(cumulative_var, label='Min VaR')
plt.plot(cumulative_etf, label='MSCI World ETF (URTH)')
plt.title("Évolution de la valeur cumulée des portefeuilles optimisés")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée (base 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()