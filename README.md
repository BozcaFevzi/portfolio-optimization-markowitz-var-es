# Quantitative Portfolio Optimization: Markowitz, VaR, and Expected Shortfall

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Finance](https://img.shields.io/badge/Finance-Quantitative-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

This project explores and implements Modern Portfolio Theory (MPT) and advanced risk management techniques to construct optimal investment portfolios.

We conduct a comparative analysis between the classic **Mean-Variance (Markowitz)** approach and tail-risk minimization strategies, specifically **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**. The project includes rigorous backtesting on historical market data (2018–2023) to evaluate the robustness of these strategies under realistic constraints.

## 🚀 Methodology

The study is divided into theoretical derivation and empirical application, covering the following key areas:

### 1. Mathematical Optimization
- **Markowitz Framework:** Derivation and resolution of the $\phi$, $\mu$, and $\sigma$ allocation problems using Lagrangian multipliers.
- **Efficient Frontier:** Numerical construction of the risk-return trade-off curve.
- **Risk Measures:** Implementation of VaR (95%) and Expected Shortfall (CVaR 95%) minimization algorithms.

### 2. Monte Carlo Simulations
- Generation of 100,000+ scenarios based on multivariate normal distributions to model asset behavior.
- Empirical estimation of tail risks (VaR/ES) to overcome the limitations of historical variance.

### 3. Backtesting & Strategy Analysis
- **Data:** Historical adjusted close prices from Yahoo Finance (assets: AAPL, TM, NVS, ITUB, etc. and benchmarks: SPY, GLD).
- **Reallocation Strategies:** Comparison of Daily vs. 3-Week rebalancing frequencies.
- **Constraints:** Implementation of "Pure" (unconstrained) vs. "Ameliorated" (capped weights 5%-40%) strategies to limit concentration risk.
- **Performance Metrics:** Analysis via Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.

## 👥 Contributors

This project was a collaborative effort. Feel free to reach out to the authors:

* **Fevzi BOZCA** - [Email](mailto:bozca.fevzi@hotmail.com)
* **Maxime KALARDASHTI** - [Email](mailto:kalardashti.maxime@gmail.com)
* **Amin ICHOU** - [Email](mailto:aminichou@hotmail.fr)

## 📂 Repository Structure

```text
.
├── report/                        # Final academic report
│   └── TER_portfolio_optimization.pdf
├── src/                           # Core Python scripts
│   ├── monte_carlo_var_es.py      # Monte Carlo simulations for VaR and ES estimation
│   ├── efficient_frontier.py      # Markowitz optimization and efficient frontier construction
│   ├── weights_constrained_backtest.py  # Portfolio optimization with weight constraints + backtesting
│   └── backtesting_strategies.py  # Strategy performance evaluation and comparison
├── LICENCE
└── README.md
