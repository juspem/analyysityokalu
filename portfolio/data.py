import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_prices(tickers: list[str], period_years: int = 3) -> pd.DataFrame:
    """Hae historiallinen hintataulukko useille tikkeille."""
    end = datetime.today()
    start = end - timedelta(days=period_years * 365)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # Handle MultiIndex columns (newer yfinance versions)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        prices = close
    else:
        if len(tickers) == 1:
            if "Close" in raw.columns:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            else:
                prices = raw.copy()
                prices.columns = [tickers[0]]
        else:
            prices = raw
    return prices.dropna(how="all")


def fetch_info(ticker: str) -> dict:
    """Hae perustiedot yksittäiselle tikkerille."""
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "Nimi": info.get("longName", ticker),
        "Sektori": info.get("sector", "N/A"),
        "Maa": info.get("country", "N/A"),
        "Valuutta": info.get("currency", "USD"),
        "Markkina-arvo": info.get("marketCap"),
        "P/E": info.get("trailingPE"),
        "Osinko": info.get("dividendYield"),
        "Beta": info.get("beta"),
        "52 viikon ylin hinta": info.get("fiftyTwoWeekHigh"),
        "52 viikon alin hinta": info.get("fiftyTwoWeekLow"),
    }


def get_benchmark(period_years: int = 3, ticker: str = "SPY") -> pd.Series:
    """Hae vertailuindeksi."""
    df = fetch_prices([ticker], period_years)
    return df[ticker]
