"""
ETF- ja rahastoanalyysi:
  - Omistusten päällekkäisyys ETFien välillä
  - Maajakauma
  - Sektorijakauma
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ── Apufunktiot ────────────────────────────────────────────────────────────────

def _get_fund_data(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


# ── Omistukset ─────────────────────────────────────────────────────────────────

def fetch_top_holdings(ticker: str, max_holdings: int = 25) -> pd.DataFrame:
    """
    Hae ETF:n/rahaston top-omistukset yfinancesta.
    Palauttaa DataFramen: symbol, name, weight (0–1).
    """
    try:
        t = _get_fund_data(ticker)
        # Uudempi yfinance API
        try:
            holdings = t.funds_data.top_holdings
            if holdings is not None and not holdings.empty:
                df = holdings.copy()
                # Normalisoi sarakkeiden nimet
                df.columns = [c.lower().strip() for c in df.columns]
                rename = {}
                for c in df.columns:
                    if "symbol" in c or "ticker" in c:
                        rename[c] = "symbol"
                    elif "name" in c or "holding" in c:
                        rename[c] = "name"
                    elif "weight" in c or "percent" in c:
                        rename[c] = "weight"
                df = df.rename(columns=rename)
                # Varmista weight on desimaalina
                if "weight" in df.columns:
                    if df["weight"].max() > 1.5:
                        df["weight"] = df["weight"] / 100
                return df[["symbol", "name", "weight"]].head(max_holdings)
        except Exception:
            pass

        # Vanhempi API fallback
        info = t.info
        holdings_list = []
        for key in ["holdings", "fundInceptionDate"]:
            if key == "holdings" and key in info:
                for h in info[key][:max_holdings]:
                    holdings_list.append({
                        "symbol": h.get("symbol", "N/A"),
                        "name": h.get("holdingName", h.get("symbol", "N/A")),
                        "weight": h.get("holdingPercent", 0),
                    })
        if holdings_list:
            return pd.DataFrame(holdings_list)

        return pd.DataFrame(columns=["symbol", "name", "weight"])

    except Exception as e:
        return pd.DataFrame(columns=["symbol", "name", "weight"])


def fetch_sector_weights(ticker: str) -> pd.Series:
    """Hae sektorijakauma ETF:lle tai osakkeelle."""
    try:
        t = _get_fund_data(ticker)
        # ETF-sektoridata
        try:
            sector_df = t.funds_data.sector_weightings
            if sector_df is not None and not sector_df.empty:
                s = sector_df.squeeze()
                if s.max() > 1.5:
                    s = s / 100
                return s.rename(ticker)
        except Exception:
            pass

        # Yksittäinen osake — yksi sektori 100%
        info = t.info
        sector = info.get("sector")
        if sector:
            return pd.Series({sector: 1.0}, name=ticker)

        return pd.Series(dtype=float, name=ticker)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def fetch_country_weights(ticker: str) -> pd.Series:
    """Hae maajakauma ETF:lle tai osakkeelle."""
    try:
        t = _get_fund_data(ticker)
        try:
            eq = t.funds_data.equity_holdings
            if eq is not None and not eq.empty:
                # funds_data.equity_holdings ei sisällä maajakaumaa suoraan —
                # kokeillaan asset_classes tai muita kenttiä
                pass
        except Exception:
            pass

        # Kokeile fund_profile
        try:
            profile = t.funds_data.fund_overview
            if profile is not None:
                pass
        except Exception:
            pass

        # Fallback: yksittäiselle osakkeelle haetaan kotimaa info-kentästä
        info = t.info
        country = info.get("country", info.get("countryCode"))
        if country:
            # Normalisoi lyhenteet → täydet nimet
            country = _normalize_country(country)
            return pd.Series({country: 1.0}, name=ticker)

        return pd.Series(dtype=float, name=ticker)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


# ── Manuaaliset maajakaumat tunnetuille ETF:ille ──────────────────────────────
# yfinance ei aina palauta maajakaumaa ETF:ille, joten tässä on yleisimpien
# ETFien arviot lähdeaineiston perusteella (päivitetty 2024/2025).
# Käyttäjä voi ylikirjoittaa nämä manuaalisesti UI:ssa.

KNOWN_ETF_COUNTRIES: dict[str, dict[str, float]] = {
    "SPY":  {"United States": 1.0},
    "VOO":  {"United States": 1.0},
    "IVV":  {"United States": 1.0},
    "QQQ":  {"United States": 1.0},
    "VTI":  {"United States": 1.0},
    "VXUS": {"United States": 0.02, "Japan": 0.14, "United Kingdom": 0.09,
             "China": 0.08, "France": 0.07, "Canada": 0.07, "Switzerland": 0.06,
             "Germany": 0.06, "India": 0.05, "Australia": 0.04, "Other": 0.32},
    "VEA":  {"Japan": 0.23, "United Kingdom": 0.15, "France": 0.11,
             "Switzerland": 0.10, "Germany": 0.09, "Australia": 0.08,
             "Netherlands": 0.05, "Sweden": 0.04, "Hong Kong": 0.04, "Other": 0.11},
    "VWO":  {"China": 0.30, "India": 0.19, "Taiwan": 0.17, "Brazil": 0.06,
             "Saudi Arabia": 0.05, "South Africa": 0.04, "Mexico": 0.03, "Other": 0.16},
    "EFA":  {"Japan": 0.24, "United Kingdom": 0.14, "France": 0.11,
             "Switzerland": 0.10, "Germany": 0.09, "Australia": 0.07, "Other": 0.25},
    "IWDA": {"United States": 0.70, "Japan": 0.06, "United Kingdom": 0.04,
             "France": 0.03, "Canada": 0.03, "Switzerland": 0.03, "Other": 0.11},
    "EUNL": {"United States": 0.70, "Japan": 0.06, "United Kingdom": 0.04,
             "France": 0.03, "Canada": 0.03, "Switzerland": 0.03, "Other": 0.11},
    "CSPX": {"United States": 1.0},
    "VWCE": {"United States": 0.63, "Japan": 0.06, "United Kingdom": 0.04,
             "China": 0.03, "France": 0.03, "Canada": 0.03, "Other": 0.18},
    "EIMI": {"China": 0.28, "India": 0.18, "Taiwan": 0.17, "Brazil": 0.06,
             "Saudi Arabia": 0.05, "South Africa": 0.04, "Other": 0.22},
    "FLX5.DE":  {"United States": 1.0},
    "FLXI.DE":  {"India": 1.0}
}

KNOWN_ETF_SECTORS: dict[str, dict[str, float]] = {
    "SPY":  {"Technology": 0.31, "Healthcare": 0.13, "Financial Services": 0.13,
             "Consumer Discretionary": 0.11, "Industrials": 0.09,
             "Communication Services": 0.09, "Consumer Staples": 0.06,
             "Energy": 0.04, "Utilities": 0.02, "Materials": 0.02},
    "VOO":  {"Technology": 0.31, "Healthcare": 0.13, "Financial Services": 0.13,
             "Consumer Discretionary": 0.11, "Industrials": 0.09,
             "Communication Services": 0.09, "Consumer Staples": 0.06,
             "Energy": 0.04, "Utilities": 0.02, "Materials": 0.02},
    "QQQ":  {"Technology": 0.51, "Communication Services": 0.16,
             "Consumer Discretionary": 0.14, "Healthcare": 0.06,
             "Industrials": 0.05, "Financial Services": 0.04, "Other": 0.04},
    "VTI":  {"Technology": 0.29, "Healthcare": 0.13, "Financial Services": 0.13,
             "Consumer Discretionary": 0.10, "Industrials": 0.10,
             "Communication Services": 0.09, "Other": 0.16},
    "FLX5.DE":  {"Technology": 0.39, "Financial Services": 0.14, 
                 "Communication Services": 0.13, "Healthcare": 0.12, 
                 "Consumer Discretionary": 0.11, "Industrials": 0.04,
                 "Consumer Staples": 0.05, "Materials": 0.02},
    "FLXI.DE":  {"Financial Services": 0.28, "Consumer Discretionary": 0.12,
                 "Technology": 0.11, "Industrials": 0.09, "Materials": 0.09,
                 "Energy": 0.09, "Healthcare": 0.06, "Consumer Staples": 0.06,
                 "Communication Services": 0.05, "Other": 0.05}
}


def get_country_weights(ticker: str) -> dict[str, float]:
    """Hae maajakauma — ensin tunnetut ETFit, sitten yfinance."""
    up = ticker.upper()
    if up in KNOWN_ETF_COUNTRIES:
        return KNOWN_ETF_COUNTRIES[up]
    series = fetch_country_weights(ticker)
    if not series.empty:
        return series.to_dict()
    return {}


def get_sector_weights(ticker: str) -> dict[str, float]:
    """Hae sektorijakauma — ensin tunnetut ETFit, sitten yfinance."""
    up = ticker.upper()
    if up in KNOWN_ETF_SECTORS:
        return KNOWN_ETF_SECTORS[up]
    series = fetch_sector_weights(ticker)
    if not series.empty:
        return series.to_dict()
    return {}


# ── Portfoliotason aggregaatit ─────────────────────────────────────────────────

def portfolio_country_exposure(
    tickers: list[str], weights: dict[str, float]
) -> pd.Series:
    """Laske portfolion painotettu maajakauma."""
    combined: dict[str, float] = {}
    for ticker in tickers:
        w = weights.get(ticker, 0)
        countries = get_country_weights(ticker)
        for country, share in countries.items():
            combined[country] = combined.get(country, 0) + w * share
    if not combined:
        return pd.Series(dtype=float)
    s = pd.Series(combined).sort_values(ascending=False)
    return s / s.sum()  # normalisoi 100%:iin


def portfolio_sector_exposure(
    tickers: list[str], weights: dict[str, float]
) -> pd.Series:
    """Laske portfolion painotettu sektorijakauma."""
    combined: dict[str, float] = {}
    for ticker in tickers:
        w = weights.get(ticker, 0)
        sectors = get_sector_weights(ticker)
        for sector, share in sectors.items():
            combined[sector] = combined.get(sector, 0) + w * share
    if not combined:
        return pd.Series(dtype=float)
    s = pd.Series(combined).sort_values(ascending=False)
    return s / s.sum()


# ── Päällekkäisyysanalyysi ─────────────────────────────────────────────────────

def compute_overlap(
    holdings_a: pd.DataFrame, holdings_b: pd.DataFrame
) -> dict:
    """
    Laske kahden ETF:n omistusten päällekkäisyys (yritykset/osakkeet).
    Palauttaa: overlap_weight, shared_symbols, details DataFrame.
    """
    if holdings_a.empty or holdings_b.empty:
        return {"overlap_weight": 0.0, "shared_symbols": [], "details": pd.DataFrame()}

    a = holdings_a.set_index("symbol")["weight"].to_dict()
    b = holdings_b.set_index("symbol")["weight"].to_dict()

    shared = set(a.keys()) & set(b.keys())
    overlap_weight = sum(min(a[s], b[s]) for s in shared)

    details = []
    for sym in sorted(shared, key=lambda s: -(a[s] + b[s]) / 2):
        details.append({
            "Osake": sym,
            "Paino A": a[sym],
            "Paino B": b[sym],
            "Päällekkäisyys": min(a[sym], b[sym]),
        })

    return {
        "overlap_weight": overlap_weight,
        "shared_symbols": list(shared),
        "details": pd.DataFrame(details),
    }


def compute_sector_overlap(
    sectors_a: dict[str, float], sectors_b: dict[str, float]
) -> dict:
    """
    Laske kahden ETF:n sektoripäällekkäisyys.
    Palauttaa: overlap_weight, shared_sectors, details DataFrame.
    """
    if not sectors_a or not sectors_b:
        return {"overlap_weight": 0.0, "shared_sectors": [], "details": pd.DataFrame()}

    a = sectors_a.copy()
    b = sectors_b.copy()

    shared = set(a.keys()) & set(b.keys())
    overlap_weight = sum(min(a[s], b[s]) for s in shared)

    details = []
    for sector in sorted(shared, key=lambda s: -(a[s] + b[s]) / 2):
        details.append({
            "Sektori": sector,
            "Paino A": a[sector],
            "Paino B": b[sector],
            "Päällekkäisyys": min(a[sector], b[sector]),
        })

    return {
        "overlap_weight": overlap_weight,
        "shared_sectors": list(shared),
        "details": pd.DataFrame(details),
    }


def compute_country_overlap(
    countries_a: dict[str, float], countries_b: dict[str, float]
) -> dict:
    """
    Laske kahden ETF:n maapäällekkäisyys.
    Palauttaa: overlap_weight, shared_countries, details DataFrame.
    """
    if not countries_a or not countries_b:
        return {"overlap_weight": 0.0, "shared_countries": [], "details": pd.DataFrame()}

    a = countries_a.copy()
    b = countries_b.copy()

    shared = set(a.keys()) & set(b.keys())
    overlap_weight = sum(min(a[s], b[s]) for s in shared)

    details = []
    for country in sorted(shared, key=lambda s: -(a[s] + b[s]) / 2):
        details.append({
            "Maa": country,
            "Paino A": a[country],
            "Paino B": b[country],
            "Päällekkäisyys": min(a[country], b[country]),
        })

    return {
        "overlap_weight": overlap_weight,
        "shared_countries": list(shared),
        "details": pd.DataFrame(details),
    }


def overlap_matrix(
    tickers: list[str], all_holdings: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Laske kaikkien ETFi-parien päällekkäisyysmatriisi (yritykset/osakkeet).
    Palauttaa NxN DataFrame prosentteina.
    """
    n = len(tickers)
    matrix = np.zeros((n, n))
    for i, a in enumerate(tickers):
        matrix[i, i] = 1.0  # itse itsensä kanssa 100%
        for j, b in enumerate(tickers):
            if i != j:
                result = compute_overlap(
                    all_holdings.get(a, pd.DataFrame()),
                    all_holdings.get(b, pd.DataFrame()),
                )
                matrix[i, j] = result["overlap_weight"]
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def sector_overlap_matrix(
    tickers: list[str]
) -> pd.DataFrame:
    """
    Laske kaikkien ETFi-parien sektoripäällekkäisyysmatriisi.
    Palauttaa NxN DataFrame prosentteina.
    """
    n = len(tickers)
    matrix = np.zeros((n, n))
    for i, a in enumerate(tickers):
        matrix[i, i] = 1.0
        sectors_a = get_sector_weights(a)
        for j, b in enumerate(tickers):
            if i != j:
                sectors_b = get_sector_weights(b)
                result = compute_sector_overlap(sectors_a, sectors_b)
                matrix[i, j] = result["overlap_weight"]
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def country_overlap_matrix(
    tickers: list[str]
) -> pd.DataFrame:
    """
    Laske kaikkien ETFi-parien maapäällekkäisyysmatriisi.
    Palauttaa NxN DataFrame prosentteina.
    """
    n = len(tickers)
    matrix = np.zeros((n, n))
    for i, a in enumerate(tickers):
        matrix[i, i] = 1.0
        countries_a = get_country_weights(a)
        for j, b in enumerate(tickers):
            if i != j:
                countries_b = get_country_weights(b)
                result = compute_country_overlap(countries_a, countries_b)
                matrix[i, j] = result["overlap_weight"]
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


# ── Apufunktiot ────────────────────────────────────────────────────────────────

COUNTRY_NAME_MAP = {
    "US": "United States", "USA": "United States",
    "GB": "United Kingdom", "UK": "United Kingdom",
    "DE": "Germany", "FR": "France", "JP": "Japan",
    "CN": "China", "CA": "Canada", "CH": "Switzerland",
    "AU": "Australia", "IN": "India", "BR": "Brazil",
    "KR": "South Korea", "TW": "Taiwan", "SE": "Sweden",
    "NL": "Netherlands", "ES": "Spain", "IT": "Italy",
    "HK": "Hong Kong", "SG": "Singapore", "DK": "Denmark",
    "NO": "Norway", "FI": "Finland", "IE": "Ireland",
    "ZA": "South Africa", "MX": "Mexico", "SA": "Saudi Arabia",
}

def _normalize_country(code_or_name: str) -> str:
    return COUNTRY_NAME_MAP.get(code_or_name.upper(), code_or_name)
