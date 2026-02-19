import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def _port_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray):
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)
    return ret, vol


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
    ret, vol = _port_stats(weights, mean_returns, cov_matrix)
    return -(ret - risk_free) / vol


def _portfolio_vol(weights, mean_returns, cov_matrix):
    return _port_stats(weights, mean_returns, cov_matrix)[1]


def optimize_max_sharpe(
    returns: pd.DataFrame, risk_free: float = 0.045
) -> dict:
    """Optimoi suurimman Sharpe-luvun salkku."""
    mean_ret = returns.mean().values
    cov = returns.cov().values
    n = len(mean_ret)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        _neg_sharpe, x0=np.ones(n) / n,
        args=(mean_ret, cov, risk_free),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    weights = dict(zip(returns.columns, result.x))
    ret, vol = _port_stats(result.x, mean_ret, cov)
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": (ret - risk_free) / vol}


def optimize_min_volatility(returns: pd.DataFrame) -> dict:
    """Optimoi pienimmän volatiliteetin salkku."""
    mean_ret = returns.mean().values
    cov = returns.cov().values
    n = len(mean_ret)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        _portfolio_vol, x0=np.ones(n) / n,
        args=(mean_ret, cov),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    weights = dict(zip(returns.columns, result.x))
    ret, vol = _port_stats(result.x, mean_ret, cov)
    return {"weights": weights, "return": ret, "volatility": vol}


def efficient_frontier(
    returns: pd.DataFrame, n_portfolios: int = 5000
) -> pd.DataFrame:
    """Monte Carlo -simulaatio tehokkaalle rintamalle."""
    mean_ret = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    n = len(mean_ret)
    results = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret = float(w @ mean_ret)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - 0.045) / vol
        results.append([ret, vol, sharpe] + list(w))
    cols = ["return", "volatility", "sharpe"] + list(returns.columns)
    return pd.DataFrame(results, columns=cols)


def equal_weight(tickers: list[str]) -> dict:
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}
