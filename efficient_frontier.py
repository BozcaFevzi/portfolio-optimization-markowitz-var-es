import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Récupération des données de marché
tickers = ["AAPL", "NESN.SW", "TCS.NS", "BHP.AX"]
data = yf.download(
    tickers,
    start="2018-01-01",
    end="2023-01-01",
    auto_adjust=True
)

# 2. Calcul des rendements journaliers
close_prices = data["Close"]
returns = close_prices.pct_change().dropna()

# 3. Statistiques annuelles
mean_returns = returns.mean() * 252      # rendement annualisé
cov_matrix = returns.cov() * 252         # matrice de covariance annualisée

n_assets = len(tickers)
n_portfolios = 10_000

# 4. Simulation Monte Carlo
results = np.zeros((3, n_portfolios))
weights_history = []
best_portfolio = {
    "weights": None,
    "sharpe": -np.inf,
    "ret": None,
    "vol": None
}

for i in range(n_portfolios):
    w = np.random.random(n_assets)
    w /= w.sum()
    weights_history.append(w)

    # rendement et volatilité du portefeuille
    port_return = np.dot(w, mean_returns)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = port_return / port_vol  # sans taux sans risque

    if sharpe > best_portfolio["sharpe"]:
        best_portfolio["weights"] = w
        best_portfolio["sharpe"] = sharpe
        best_portfolio["ret"] = port_return
        best_portfolio["vol"] = port_vol

    results[0, i] = port_return
    results[1, i] = port_vol
    results[2, i] = sharpe

# 5. Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(
    results[1, :],
    results[0, :],
    c=results[2, :],
    cmap="viridis",
    marker="o",
    s=10
)

plt.scatter(
    best_portfolio["vol"],
    best_portfolio["ret"],
    color="red",
    marker="*",
    s=200,
    label="Sharpe max"
)

plt.xlabel("Volatilité (risque)")
plt.ylabel("Rendement espéré")
plt.title("Portefeuilles simulés (Monte Carlo)")
plt.colorbar(label="Ratio de Sharpe")
plt.legend()
plt.grid(True)
plt.show()
