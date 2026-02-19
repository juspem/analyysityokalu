import numpy as np
import pandas as pd


TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    r = returns[tickers]
    return r.dot(w)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def annualized_return(returns: pd.Series) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return float(total ** (1 / n_years) - 1) if n_years > 0 else 0.0


def annualized_volatility(returns: pd.Series) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.045) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    return (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.045) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    ann_ret = annualized_return(returns)
    downside_returns = returns[returns < 0]
    downside = downside_returns.std() * np.sqrt(TRADING_DAYS) if len(downside_returns) > 0 else 0.0
    return (ann_ret - risk_free) / downside if downside > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    ann_ret = annualized_return(returns)
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else 0.0


def beta_alpha(port_returns: pd.Series, bench_returns: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if len(aligned) == 0:
        return 0.0, 0.0
    aligned.columns = ["portfolio", "benchmark"]
    cov = np.cov(aligned["portfolio"], aligned["benchmark"])
    if cov[1, 1] == 0:
        return 0.0, 0.0
    beta = cov[0, 1] / cov[1, 1]
    alpha = annualized_return(aligned["portfolio"]) - beta * annualized_return(aligned["benchmark"])
    return float(beta), float(alpha)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty or len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def compute_all_metrics(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    benchmark: str = "SPY",
    risk_free: float = 0.045,
) -> dict:
    beta, alpha = beta_alpha(port_returns, bench_returns)
    return {
        "Vuosituotto": f"{annualized_return(port_returns):.2%}",
        "Volatiliteetti": f"{annualized_volatility(port_returns):.2%}",
        "Sharpe": f"{sharpe_ratio(port_returns, risk_free):.2f}",
        "Sortino": f"{sortino_ratio(port_returns, risk_free):.2f}",
        "Max Drawdown": f"{max_drawdown(port_returns):.2%}",
        "Calmar": f"{calmar_ratio(port_returns):.2f}",
        f"Beta (vs {benchmark})": f"{beta:.2f}",
        f"Alpha (vs {benchmark})": f"{alpha:.2%}",
        "VaR 95%": f"{value_at_risk(port_returns):.2%}",
        "CVaR 95%": f"{conditional_var(port_returns):.2%}",
    }
