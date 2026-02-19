import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Apufunktiot ────────────────────────────────────────────────────────────────

def _make_future_dates(last_date, days_ahead: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date, periods=days_ahead + 1, freq="B")[1:]


def _prepare_prices(prices: pd.Series) -> pd.Series:
    """Varmista tz-naive indeksi ja tasainen business-day frekvenssi statsmodelsille."""
    s = prices.copy()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex(s.index)
    return s.asfreq("B").ffill()


# ── ARIMA ──────────────────────────────────────────────────────────────────────

def arima_forecast(
    prices: pd.Series,
    days_ahead: int = 60,
    p: int = 5,
    d: int = 1,
    q: int = 0,
    auto: bool = False,
) -> dict:
    """
    ARIMA(p,d,q) ennuste.

    Parametrit:
        p  – autoregressiivinen järjestys: kuinka monen aiemman päivän kurssia käytetään
             (tyypillisesti 1–10 osakkeille; suurempi = muistaa pidemmän historian)
        d  – differentiointiaste: 1 poistaa trendin (lähes aina 1 osakkeille),
             2 poistaa trendin kiihtyvyyden (harvoin tarpeen)
        q  – liukuvan keskiarvon virhejärjestys: korjaa ennustevirhettä aiemmista ajoista
             (0–3 sopii useimmiten)
        auto – jos True ja pmdarima on asennettu, etsii parhaat p/q automaattisesti AIC:n mukaan
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA

        s = _prepare_prices(prices)

        if auto:
            try:
                from pmdarima import auto_arima
                m_auto = auto_arima(s, d=d, seasonal=False,
                                    information_criterion="aic",
                                    stepwise=True, suppress_warnings=True)
                p, d, q = m_auto.order
            except ImportError:
                pass  # fallback käsin syötettyihin arvoihin

        model = _ARIMA(s, order=(p, d, q))
        fit = model.fit()
        fc = fit.get_forecast(steps=days_ahead)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        future = _make_future_dates(s.index[-1], days_ahead)

        return {
            "model": f"ARIMA({p},{d},{q})",
            "order": (p, d, q),
            "yhat": pd.Series(mean.values, index=future),
            "yhat_lower": pd.Series(ci.iloc[:, 0].values, index=future),
            "yhat_upper": pd.Series(ci.iloc[:, 1].values, index=future),
            "aic": round(fit.aic, 2),
            "bic": round(fit.bic, 2),
            "error": None,
        }
    except Exception as e:
        return {"model": "ARIMA", "yhat": None, "error": str(e)}


# ── SARIMA ─────────────────────────────────────────────────────────────────────

def sarima_forecast(
    prices: pd.Series,
    days_ahead: int = 60,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    P: int = 1,
    D: int = 0,
    Q: int = 1,
    m: int = 5,
    auto: bool = False,
) -> dict:
    """
    SARIMA(p,d,q)(P,D,Q,m) ennuste.

    Ei-kausiparametrit (p,d,q): kuten ARIMA yllä.

    Kausiparametrit:
        P  – kausiluonteinen AR-järjestys (yleensä 0–2)
        D  – kausiluonteinen differenssijärjestys (yleensä 0 tai 1)
        Q  – kausiluonteinen MA-järjestys (yleensä 0–2)
        m  – kausijakson pituus kaupankäyntipäivissä:
               5  = viikko  (suositeltava oletusarvo osakkeille)
              21  = kuukausi
              63  = kvartaali
             252  = vuosi (raskas, pidä P/Q/D pienenä)
        auto – pmdarima auto_arima kausiparametreille
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

        s = _prepare_prices(prices)

        if auto:
            try:
                from pmdarima import auto_arima
                m_auto = auto_arima(
                    s, d=d, D=D, seasonal=True, m=m,
                    information_criterion="aic",
                    stepwise=True, suppress_warnings=True,
                )
                p, d, q = m_auto.order
                P, D, Q, m = m_auto.seasonal_order
            except ImportError:
                pass

        model = _SARIMAX(s, order=(p, d, q), seasonal_order=(P, D, Q, m),
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fc = fit.get_forecast(steps=days_ahead)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        future = _make_future_dates(s.index[-1], days_ahead)

        return {
            "model": f"SARIMA({p},{d},{q})({P},{D},{Q},{m})",
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, m),
            "yhat": pd.Series(mean.values, index=future),
            "yhat_lower": pd.Series(ci.iloc[:, 0].values, index=future),
            "yhat_upper": pd.Series(ci.iloc[:, 1].values, index=future),
            "aic": round(fit.aic, 2),
            "bic": round(fit.bic, 2),
            "error": None,
        }
    except Exception as e:
        return {"model": "SARIMA", "yhat": None, "error": str(e)}


# ── SARIMAX ────────────────────────────────────────────────────────────────────

def sarimax_forecast(
    prices: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
    days_ahead: int = 60,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    P: int = 1,
    D: int = 0,
    Q: int = 1,
    m: int = 5,
) -> dict:
    """
    SARIMAX — SARIMA + ulkoiset (eXogenous) selittävät muuttujat.

    Ulkoiset muuttujat (exog_train / exog_future):
        DataFrame jossa sarakkeet voivat olla esim:
          - VIX         (markkinan implisiittinen volatiliteetti)
          - DXY         (dollari-indeksi)
          - US10Y       (10v korko)
          - SPY_return  (markkinatuotto)
          - OIL         (öljyn hinta, jos energia-osake)

        TÄRKEÄÄ: exog_future on arvauksesi/ennusteesi näistä muuttujista
        ennustejaksolla. Jos et tiedä tulevia arvoja, käytä SARIMA:a.
        Voit käyttää viimeistä tunnettua arvoa (naive) tai lineaarista trendiä.

    Parametrit p,d,q,P,D,Q,m: kuten SARIMA yllä.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

        s = _prepare_prices(prices)
        exog_aligned = exog_train.reindex(s.index).ffill().bfill()

        model = _SARIMAX(
            s, exog=exog_aligned,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)

        exog_fc = exog_future.values[:days_ahead]
        fc = fit.get_forecast(steps=days_ahead, exog=exog_fc)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        future = _make_future_dates(s.index[-1], days_ahead)

        return {
            "model": f"SARIMAX({p},{d},{q})({P},{D},{Q},{m}) + {list(exog_train.columns)}",
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, m),
            "exog_variables": list(exog_train.columns),
            "yhat": pd.Series(mean.values, index=future),
            "yhat_lower": pd.Series(ci.iloc[:, 0].values, index=future),
            "yhat_upper": pd.Series(ci.iloc[:, 1].values, index=future),
            "aic": round(fit.aic, 2),
            "bic": round(fit.bic, 2),
            "error": None,
        }
    except Exception as e:
        return {"model": "SARIMAX", "yhat": None, "error": str(e)}


# ── Automaattinen mallien vertailu ─────────────────────────────────────────────

def compare_models(
    prices: pd.Series,
    days_ahead: int = 60,
    arima_params: dict = None,
    sarima_params: dict = None,
) -> list:
    """Aja ARIMA ja SARIMA rinnakkain vertailua varten."""
    a_p = arima_params or {"p": 5, "d": 1, "q": 0}
    s_p = sarima_params or {"p": 1, "d": 1, "q": 1, "P": 1, "D": 0, "Q": 1, "m": 5}
    return [
        arima_forecast(prices, days_ahead=days_ahead, **a_p),
        sarima_forecast(prices, days_ahead=days_ahead, **s_p),
    ]


# ── Prophet ────────────────────────────────────────────────────────────────────

def prophet_forecast(prices: pd.Series, days_ahead: int = 90):
    try:
        from prophet import Prophet
        df = pd.DataFrame({"ds": prices.index, "y": prices.values})
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.1)
        m.fit(df)
        future = m.make_future_dataframe(periods=days_ahead, freq="B")
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
    except ImportError:
        return None


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def monte_carlo_simulation(
    prices: pd.Series, days_ahead: int = 252, n_simulations: int = 500
) -> pd.DataFrame:
    # Varmista että prices on yksiulotteinen Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.squeeze()

    returns = prices.pct_change().dropna()
    mu      = float(returns.mean())
    sigma   = float(returns.std())
    last_price = float(prices.iloc[-1])  # eksplisiittinen float — ei Series

    sims = np.zeros((days_ahead, n_simulations))
    for i in range(n_simulations):
        r = np.random.normal(mu, sigma, days_ahead)
        sims[:, i] = last_price * np.cumprod(1 + r)  # numpy cumprod — ei pandas

    dates = _make_future_dates(prices.index[-1], days_ahead)
    return pd.DataFrame(sims, index=dates)


# ── Liukuva keskiarvo ──────────────────────────────────────────────────────────

def moving_average_forecast(prices: pd.Series, days_ahead: int = 30) -> pd.Series:
    ma20 = prices.rolling(20).mean().iloc[-1]
    ma50 = prices.rolling(50).mean().iloc[-1]
    trend = (ma20 - ma50) / ma50 / 50
    last_price = prices.iloc[-1]
    future_dates = _make_future_dates(prices.index[-1], days_ahead)
    forecast = [last_price * (1 + trend * i) for i in range(1, days_ahead + 1)]
    return pd.Series(forecast, index=future_dates, name="MA Ennuste")
