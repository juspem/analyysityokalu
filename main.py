"""
Portfolioanalyysityökalu — käynnistys:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from portfolio import data, analysis, optimization, forecasting, visualization, etf_analysis

# ══════════════════════════════════════════════════════════════════════════════
# SIVUN ASETUKSET
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolioanalyysi",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIVUPALKKI — omistukset & asetukset
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Salkku")
    st.caption("Syötä tikkeri ja arvo. Muutokset päivittyvät automaattisesti.")

    if "holdings" not in st.session_state:
        st.session_state.holdings = [
            {"ticker": "FLX5.DE",  "value": 7764},
            {"ticker": "FLXI.DE",  "value": 1847},
            {"ticker": "MANTA.HE",   "value": 1362},
        ]

    # Callback joka tallentaa muutokset HETI kun käyttäjä muokkaa taulukkoa
    def _sync_holdings():
        editor_data = st.session_state.get("holdings_editor", {})
        
        # Handle edited rows
        edited_rows = editor_data.get("edited_rows", {})
        if edited_rows:
            for idx, row in edited_rows.items():
                idx = int(idx)
                if idx < len(st.session_state.holdings):
                    for key, value in row.items():
                        st.session_state.holdings[idx][key] = value
        
        # Handle added rows
        added_rows = editor_data.get("added_rows", [])
        if added_rows:
            st.session_state.holdings.extend(added_rows)
        
        # Handle deleted rows
        deleted_rows = editor_data.get("deleted_rows", [])
        if deleted_rows:
            # Sort in reverse to delete from end first
            delete_indices = sorted([int(idx) for idx in deleted_rows], reverse=True)
            for idx in delete_indices:
                if 0 <= idx < len(st.session_state.holdings):
                    st.session_state.holdings.pop(idx)

    holdings_df = pd.DataFrame(st.session_state.holdings)
    if holdings_df.empty:
        holdings_df = pd.DataFrame([{"ticker": "", "value": 0}])

    edited = st.data_editor(
        holdings_df,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Tikkeri", width="small"),
            "value":  st.column_config.NumberColumn("Arvo (€)", min_value=0, step=100),
        },
        use_container_width=True,
        key="holdings_editor",
        on_change=_sync_holdings,
    )

    st.divider()
    st.markdown("## Asetukset")
    period    = st.selectbox("Tarkastelujakso", [1, 2, 3, 5], index=2,
                              format_func=lambda x: f"{x} vuotta")
    benchmark = st.selectbox("Vertailuindeksi", ["Ei vertailua", "IWDA", "QQQ", "SPY", "VEA", "VOO", "VWO", "VTI"],
                            index=0, format_func=lambda x: x)
    use_benchmark = benchmark != "Ei vertailua"
    risk_free = st.slider("Riskitön korko (%)", 0.0, 8.0, 4.5, 0.1) / 100

    st.divider()
    total_val = sum(h.get("value", 0) for h in st.session_state.holdings
                    if h.get("ticker") and h.get("value", 0) > 0)
    st.metric("Salkun kokonaisarvo", f"{total_val:,.0f} €")

# ══════════════════════════════════════════════════════════════════════════════
# DATAN LATAUS (yhteinen kaikille välilehdille)
# ══════════════════════════════════════════════════════════════════════════════

holdings = [h for h in st.session_state.holdings if h.get("ticker") and h.get("value", 0) > 0]

if not holdings:
    st.title("Portfolioanalyysityökalu")
    st.info("Lisää omistuksia vasemmalle sivupalkkiin aloittaaksesi.")
    st.stop()

tickers      = [h["ticker"].strip().upper() for h in holdings]
values       = [h["value"] for h in holdings]
total_value  = sum(values)
current_weights = {t: v / total_value for t, v in zip(tickers, values)}

with st.spinner("Ladataan markkinadataa..."):
    try:
        prices     = data.fetch_prices(tickers, period_years=period)
        if use_benchmark:
            spy_prices = data.get_benchmark(period_years=period, ticker=benchmark)
    except Exception as e:
        st.error(f"Datan lataus epäonnistui: {e}")
        st.stop()

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"Tikkereitä ei löydy: {', '.join(missing)} — jätetään pois analyysistä.")
    tickers = [t for t in tickers if t in prices.columns]
    current_weights = {t: current_weights[t] for t in tickers}
    total_w = sum(current_weights.values())
    current_weights = {t: w / total_w for t, w in current_weights.items()}

if not tickers:
    st.error("Yhtään tikkeriä ei löydy — tarkista syötteet.")
    st.stop()

ret_df            = analysis.daily_returns(prices[tickers])
port_ret          = analysis.portfolio_returns(ret_df, current_weights)

if use_benchmark:
    spy_ret           = spy_prices.pct_change().dropna()
    spy_ret_aligned   = spy_ret.reindex(port_ret.index).dropna()
    port_ret_aligned  = port_ret.reindex(spy_ret_aligned.index).dropna()
    spy_ret_aligned   = spy_ret_aligned.loc[port_ret_aligned.index]

    if port_ret_aligned.empty or len(port_ret_aligned) == 0:
        st.error(f"Ei löytynyt yhteistä hintadataa salkun ja {benchmark}:n välillä. Tarkista, että tickereillä on dataa valitulla ajanjaksolla.")
        st.stop()

    if spy_ret_aligned.empty or len(spy_ret_aligned) == 0:
        st.error(f"{benchmark}-vertailudata on tyhjä. Tarkista, että {benchmark}:lla on dataa valitulla ajanjaksolla.")
        st.stop()
else:
    port_ret_aligned  = port_ret.dropna()
    spy_ret_aligned   = pd.Series(dtype=float)

cum_port          = analysis.cumulative_returns(port_ret_aligned).dropna()

# Handle timezone for plotting
try:
    if hasattr(cum_port.index, 'tz') and cum_port.index.tz is not None:
        cum_port.index = cum_port.index.tz_convert(None)
except Exception:
    pass

if use_benchmark:
    cum_spy           = analysis.cumulative_returns(spy_ret_aligned).dropna()
    try:
        if hasattr(cum_spy.index, 'tz') and cum_spy.index.tz is not None:
            cum_spy.index = cum_spy.index.tz_convert(None)
    except Exception:
        pass
    metrics           = analysis.compute_all_metrics(port_ret_aligned, spy_ret_aligned, benchmark, risk_free)
    spy_metrics = {
        "Vuosituotto":  f"{analysis.annualized_return(spy_ret_aligned):.2%}",
        "Volatiliteetti": f"{analysis.annualized_volatility(spy_ret_aligned):.2%}",
        "Sharpe":       f"{analysis.sharpe_ratio(spy_ret_aligned, risk_free):.2f}",
        "Max Drawdown": f"{analysis.max_drawdown(spy_ret_aligned):.2%}",
    }
else:
    cum_spy = pd.Series(dtype=float)
    # Calculate metrics without benchmark
    metrics = {
        "Vuosituotto": f"{analysis.annualized_return(port_ret_aligned):.2%}",
        "Volatiliteetti": f"{analysis.annualized_volatility(port_ret_aligned):.2%}",
        "Sharpe": f"{analysis.sharpe_ratio(port_ret_aligned, risk_free):.2f}",
        "Sortino": f"{analysis.sortino_ratio(port_ret_aligned, risk_free):.2f}",
        "Max Drawdown": f"{analysis.max_drawdown(port_ret_aligned):.2%}",
    }
    spy_metrics = {}

# ══════════════════════════════════════════════════════════════════════════════
# PÄÄNAVIGAATIO — viisi välilehteä
# ══════════════════════════════════════════════════════════════════════════════

TAB_NAMES = ["Yhteenveto", "Tuotto & Riski", "Hajautus", "Optimointi", "Ennusteet"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = TAB_NAMES[0]

# Style radio buttons to look like Streamlit tabs
st.markdown("""
<style>
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    gap: 0px;
    background: transparent;
    border-bottom: 1px solid rgba(250,250,250,0.2);
    padding-bottom: 0;
    margin-bottom: 1rem;
}
div[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center;
    justify-content: center;
    padding: 0.5rem 1rem;
    cursor: pointer;
    border-radius: 4px 4px 0 0;
    border: 1px solid transparent;
    border-bottom: none;
    color: rgba(250,250,250,0.6);
    font-size: 0.9rem;
    white-space: nowrap;
    background: transparent;
    margin-bottom: -1px;
    gap: 0;
}
div[data-testid="stRadio"] > div > label:hover {
    color: rgba(250,250,250,0.9);
    background: rgba(250,250,250,0.05);
}
div[data-testid="stRadio"] > div > label[aria-checked="true"] {
    color: #ffffff;
    background: rgba(250,250,250,0.1);
    border-color: rgba(250,250,250,0.2);
    border-bottom-color: transparent;
}
div[data-testid="stRadio"] > div > label > div:first-child,
div[data-testid="stRadio"] > div > label > div[data-testid="stMarkdownContainer"] ~ div {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stRadio"] > div > label > div[data-testid="stMarkdownContainer"] {
    display: flex;
    align-items: center;
    justify-content: center;

}
</style>
""", unsafe_allow_html=True)

selected_tab = st.radio(
    "Navigaatio",
    TAB_NAMES,
    index=TAB_NAMES.index(st.session_state.active_tab),
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)
st.session_state.active_tab = selected_tab

# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 1 — YHTEENVETO
# ══════════════════════════════════════════════════════════════════════════════

if selected_tab == TAB_NAMES[0]:
    st.markdown("### Avainmetriikat")

    metric_cols = list(metrics.items())
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(metric_cols):
            k, v = metric_cols[i]
            if use_benchmark:
                spy_v = spy_metrics.get(k)
                col.metric(k, v, f"{benchmark}: {spy_v}" if spy_v else None)
            else:
                col.metric(k, v)

    cols2 = st.columns(5)
    for i, col in enumerate(cols2):
        idx = i + 5
        if idx < len(metric_cols):
            col.metric(metric_cols[idx][0], metric_cols[idx][1])

    st.divider()

    # Tuottokaavio + hajautuspiirakka
    c_left, c_right = st.columns([3, 2])
    with c_left:
        if use_benchmark:
            st.plotly_chart(
                visualization.cumulative_returns_chart(cum_port, cum_spy, bench_name=benchmark),
                use_container_width=True,
                key="pc_1",
            )
        else:
            st.plotly_chart(
                visualization.cumulative_returns_chart(cum_port, pd.Series(dtype=float)),
                use_container_width=True,
                key="pc_1",
            )
    with c_right:
        st.plotly_chart(
            visualization.allocation_pie(current_weights, "Nykyinen hajautus"),
            use_container_width=True,
            key="pc_2",
        )

    # Osakekohtainen info
    st.markdown("### Osakekohtainen tiedot")
    with st.spinner("Haetaan perustietoja..."):
        info_data = []
        for t in tickers:
            try:
                info = data.fetch_info(t)
                info["ticker"] = t
                info_data.append(info)
            except Exception:
                pass
    
    if info_data:
        info_df = pd.DataFrame(info_data)
        
        # Lisätään Yahoo Finance -sarake klikattavalla linkillä
        if "ticker" in info_df.columns:
            # Luodaan URL-sarake
            info_df["Yahoo Finance"] = info_df["ticker"].apply(
                lambda x: f"https://finance.yahoo.com/quote/{x}"
            )
            info_df = info_df.set_index("ticker")
            
            # Käytetään LinkColumn tai fallback -ratkaisua
            try:
                st.dataframe(
                    info_df,
                    use_container_width=True,
                    column_config={
                        "Yahoo Finance": st.column_config.LinkColumn(
                            "Yahoo Finance",
                            display_text="Avaa tiedot"
                        )
                    }
                )
            except AttributeError:
                # Fallback: näytetään linkki erikseen ennen taulukkoa
                st.markdown("**Yahoo Finance -linkit:**")
                for t in info_df.index:
                    url = f"https://finance.yahoo.com/quote/{t}"
                    st.markdown(f"- [{t}]({url})")
                st.dataframe(info_df.drop(columns=["Yahoo Finance"]), use_container_width=True)
        else:
            st.dataframe(info_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 2 — TUOTTO & RISKI
# ══════════════════════════════════════════════════════════════════════════════

if selected_tab == TAB_NAMES[1]:
    st.markdown("### Tuotto & Riskianalyysi")

    risk_subtabs = st.tabs(["Tuotto", "Drawdown", "Rullaavat mittarit", "Korrelaatiot"])

    with risk_subtabs[0]:
        if use_benchmark:
            st.plotly_chart(
                visualization.cumulative_returns_chart(cum_port, cum_spy, bench_name=benchmark),
                use_container_width=True,
                key="pc_3",
            )
        else:
            st.plotly_chart(
                visualization.cumulative_returns_chart(cum_port, pd.Series(dtype=float)),
                use_container_width=True,
                key="pc_3",
            )
        # Vuosittainen tuotto per osake
        st.markdown("#### Yksittäisten arvopapereiden kumulatiivinen tuotto")
        fig_ind = __import__("plotly.graph_objects", fromlist=["Figure"]).Figure()
        import plotly.graph_objects as go
        for t in tickers:
            ind_ret = prices[t].pct_change().dropna()
            ind_cum = analysis.cumulative_returns(ind_ret) * 100
            fig_ind.add_trace(go.Scatter(
                x=ind_cum.index, y=ind_cum,
                name=t, mode="lines", line=dict(width=2),
            ))
        fig_ind.update_layout(
            title="Yksittäiset arvopaperit vs aika",
            yaxis_title="Tuotto %", template="plotly_dark", hovermode="x unified",
        )
        st.plotly_chart(fig_ind, use_container_width=True, key="pc_4")

    with risk_subtabs[1]:
        st.plotly_chart(visualization.drawdown_chart(port_ret_aligned), use_container_width=True, key="pc_5")

        # Drawdown-taulukko
        dd_cum  = (1 + port_ret_aligned).cumprod()
        dd_roll = dd_cum.cummax()
        drawdown_series = (dd_cum - dd_roll) / dd_roll * 100
        worst_5 = drawdown_series.nsmallest(5)
        st.markdown("#### Pahimmat drawdown-päivät")
        st.dataframe(
            worst_5.rename("Drawdown %").reset_index().rename(columns={"index": "Päivä"}),
            use_container_width=True, hide_index=True,
        )

    with risk_subtabs[2]:
        window = st.slider("Rullaava ikkuna (päiviä)", 20, 120, 60)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                visualization.rolling_sharpe_chart(port_ret_aligned, window),
                use_container_width=True,
                key="pc_6",
            )
        with c2:
            # Rullaava volatiliteetti
            roll_vol = port_ret_aligned.rolling(window).std() * np.sqrt(252) * 100
            fig_vol = go.Figure(go.Scatter(
                x=roll_vol.index, y=roll_vol,
                line=dict(color="#FFA15A", width=2), name="Rullaava volatiliteetti",
            ))
            fig_vol.add_hline(
                y=roll_vol.mean(), line_dash="dash",
                line_color="gray", annotation_text="Keskiarvo",
            )
            fig_vol.update_layout(
                title=f"Rullaava volatiliteetti ({window} pv)",
                yaxis_title="Vuosittainen vol %", template="plotly_dark",
            )
            st.plotly_chart(fig_vol, use_container_width=True, key="pc_7")

        # Rullaava Beta — only show when benchmark is selected
        if use_benchmark:
            _p = port_ret_aligned.copy()
            _s = spy_ret_aligned.copy()
            if hasattr(_p.index, "tz") and _p.index.tz is not None:
                _p.index = _p.index.tz_localize(None)
            if hasattr(_s.index, "tz") and _s.index.tz is not None:
                _s.index = _s.index.tz_localize(None)
            common_idx = _p.index.intersection(_s.index)
            _p = _p.loc[common_idx]
            _s = _s.loc[common_idx]
            roll_cov  = _p.rolling(window).cov(_s)
            roll_vspy = _s.rolling(window).var()
            roll_beta = (roll_cov / roll_vspy).dropna()
            if roll_beta.empty:
                st.warning("Betaa ei voitu laskea — liian lyhyt yhteinen aikaväli.")
            else:
                # Drop NaN values before calculating stats
                roll_beta_valid = roll_beta.dropna()
                if roll_beta_valid.empty:
                    st.warning("Beta-arvot sisältävät vain virheellisiä arvoja.")
                else:
                    fig_beta = go.Figure(go.Scatter(
                        x=roll_beta.index, y=roll_beta,
                        line=dict(color="#19D3F3", width=2), name="Rullaava Beta",
                    ))
                    fig_beta.add_hline(y=1, line_dash="dot", line_color="white", annotation_text="β=1")
                    fig_beta.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="β=0")
                    # Calculate y-axis range with bounded minimum and consistent padding
                    # Use .agg() for single-pass computation with explicit min/max extraction
                    beta_stats = roll_beta_valid.agg(['min', 'max'])
                    
                    # Safely get min/max values with fallback
                    beta_min = float(beta_stats.loc['min']) if 'min' in beta_stats.index else 0.0
                    beta_max = float(beta_stats.loc['max']) if 'max' in beta_stats.index else 1.0

                    # Validate computed statistics to handle edge cases (e.g., all NaN after dropna)
                    if np.isnan(beta_min) or np.isnan(beta_max):
                        st.warning("Beta-arvot sisältävät virheellisiä arvoja.")
                        beta_min, beta_max = -1.0, 1.0  # Fallback to default range
                    
                    # Define padding as a named constant for better maintainability
                    BETA_PADDING_RATIO = 0.2  # 20% padding applied to data range
                    
                    # Calculate dynamic padding based on actual data range for better visualization
                    beta_range = beta_max - beta_min
                    beta_padding = beta_range * BETA_PADDING_RATIO if beta_range > 0 else 0.1
                    
                    # Ensure y-axis always spans a reasonable range for beta values
                    # Beta typically ranges from -∞ to +∞, but we bound it sensibly
                    y_axis_min = min(beta_min - beta_padding, -1)  # Allow view below zero
                    y_axis_max = beta_max + beta_padding

                    fig_beta.update_layout(
                        title=f"Rullaava Beta vs {benchmark} ({window} pv)",
                        yaxis_title="Beta",
                        template="plotly_dark",
                        yaxis=dict(range=[y_axis_min, y_axis_max]),
                    )
                    st.plotly_chart(fig_beta, use_container_width=True, key="pc_8")
        else:
            st.info("Valitse vertailuindeksi nähdäksesi Beta-analyysin.")

    with risk_subtabs[3]:
        if len(tickers) > 1:
            corr = analysis.correlation_matrix(ret_df)
            st.plotly_chart(visualization.correlation_heatmap(corr), use_container_width=True, key="pc_9")

            # Scatter-matriisi
            st.markdown("#### Tuottojen hajontakaaviot")
            import plotly.express as px
            fig_scatter = px.scatter_matrix(
                ret_df * 100,
                dimensions=tickers,
                title="Päivätuottojen scatter-matriisi",
                template="plotly_dark",
                opacity=0.4,
            )
            fig_scatter.update_traces(marker=dict(size=2))
            st.plotly_chart(fig_scatter, use_container_width=True, key="pc_10")
        else:
            st.info("Korrelaatioanalyysi vaatii vähintään 2 arvopaperia.")

# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 3 — HAJAUTUS
# ══════════════════════════════════════════════════════════════════════════════

if selected_tab == TAB_NAMES[2]:
    geo_subtabs = st.tabs(["Maajakauma", "Sektorijakauma", "ETF-päällekkäisyydet"])

    # ── Maajakauma ─────────────────────────────────────────────────────────────
    with geo_subtabs[0]:
        st.caption("Portfolion maantieteellinen paino laskettuna kaikkien omistustesi perusteella.")

        with st.spinner("Lasketaan maajakaumaa..."):
            country_exp = etf_analysis.portfolio_country_exposure(tickers, current_weights)

        if country_exp.empty:
            st.warning("Maajakaumaa ei saatu automaattisesti. Tarkista tikkerit.")
        else:
            st.plotly_chart(
                visualization.country_choropleth(country_exp),
                use_container_width=True,
                key="pc_11",
            )
            c_bar, c_tbl = st.columns([2, 1])
            with c_bar:
                st.plotly_chart(visualization.country_bar_chart(country_exp), use_container_width=True, key="pc_12")
            with c_tbl:
                st.markdown("**Maapaino-taulukko**")
                df_c = country_exp.reset_index()
                df_c.columns = ["Maa", "Paino"]
                df_c["Paino"] = df_c["Paino"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_c, use_container_width=True, hide_index=True)

        with st.expander("Puuttuuko ETF:si listalta?"):
            st.markdown(
                "Maajakauma perustuu tunnettuihin ETF-jakaumiin (SPY, VOO, VWCE, IWDA jne.) "
                "sekä yksittäisten osakkeiden kotimaahan. Lisää oma ETF muokkaamalla "
                "`KNOWN_ETF_COUNTRIES`-sanakirjaa tiedostossa `portfolio/etf_analysis.py`."
            )
            t_sel = st.selectbox("Tarkastele tikkerin nykyistä jakaumaa", tickers, key="cntry_sel")
            cw = etf_analysis.get_country_weights(t_sel)
            if cw:
                for c, w in cw.items():
                    st.write(f"- {c}: {w*100:.1f}%")
            else:
                st.write("Ei automaattisia tietoja.")

    # ── Sektorijakauma ─────────────────────────────────────────────────────────
    with geo_subtabs[1]:
        st.caption("Portfolion sektoripaino laskettuna kaikkien omistustesi perusteella.")

        with st.spinner("Lasketaan sektorijakaumaa..."):
            sector_exp = etf_analysis.portfolio_sector_exposure(tickers, current_weights)
            ticker_sectors_weighted = {
                t: {s: v * current_weights.get(t, 0)
                    for s, v in etf_analysis.get_sector_weights(t).items()}
                for t in tickers
            }
            if use_benchmark:
                spy_sectors = pd.Series(etf_analysis.get_sector_weights(benchmark))
            else:
                spy_sectors = pd.Series(dtype=float)

        if sector_exp.empty:
            st.warning("Sektorijakaumaa ei saatu automaattisesti. Tarkista tikkerit.")
        else:
            c_s1, c_s2 = st.columns([3, 2])
            with c_s1:
                if use_benchmark:
                    st.plotly_chart(
                        visualization.sector_bar_chart(sector_exp, spy_sectors),
                        use_container_width=True,
                        key="pc_13",
                    )
                else:
                    st.plotly_chart(
                        visualization.sector_bar_chart(sector_exp),
                        use_container_width=True,
                        key="pc_13",
                    )
            with c_s2:
                st.plotly_chart(
                    visualization.sector_sunburst(sector_exp, ticker_sectors_weighted),
                    use_container_width=True,
                    key="pc_14",
                )

            if use_benchmark:
                st.markdown(f"**Sektoripaino-taulukko (portfoliosi vs {benchmark})**")
                sector_table = pd.DataFrame({
                    "Sektori":      sector_exp.index,
                    "Portfoliosi":  [f"{v*100:.1f}%" for v in sector_exp.values],
                    benchmark:      [f"{spy_sectors.get(s, 0)*100:.1f}%" for s in sector_exp.index],
                    "Ero":          [f"{(sector_exp[s] - spy_sectors.get(s, 0))*100:+.1f}%"
                                     for s in sector_exp.index],
                })
                st.dataframe(sector_table, use_container_width=True, hide_index=True)

    # ── ETF-päällekkäisyydet ───────────────────────────────────────────────────
    with geo_subtabs[2]:
        st.markdown(
            "Vertaa ETFiesi sisäisiä omistuksia toisiinsa. "
            "**Korkea päällekkäisyys** tarkoittaa että maksat kahdesta rahastosta "
            "jotka omistavat samoja osakkeita — hajautus ei lisäänny."
        )
        st.caption("Omistusdata haetaan yfinancesta — toimii parhaiten tunnetuille US-ETF:ille.")

        # Päällekkäisyyden tyyppi - radio buttons
        overlap_type = st.radio(
            "Valitse päällekkäisyyden tyyppi:",
            ["Yritykset (osakkeet)", "Sektorit", "Maat"],
            horizontal=True,
            key="overlap_type"
        )

        if overlap_type == "Yritykset (osakkeet)":
            overlap_key = "holdings_data"
            overlap_title = "ETF-omistusten päällekkäisyysmatriisi"
        elif overlap_type == "Sektorit":
            overlap_key = "sector_data"
            overlap_title = "ETF-sektorien päällekkäisyysmatriisi"
        else:
            overlap_key = "country_data"
            overlap_title = "ETF-maiden päällekkäisyysmatriisi"

        if len(tickers) < 2:
            st.info("Lisää vähintään 2 omistusta päällekkäisyysanalyysia varten.")
        else:
            # Yritykset-osake: tarvitsee erillisen datan haun
            if overlap_type == "Yritykset (osakkeet)":
                if st.button("Hae ETF-omistusdata", key="fetch_holdings_btn"):
                    st.session_state["holdings_data"] = {}
                    prog = st.progress(0, text="Haetaan omistuksia...")
                    for i, t in enumerate(tickers):
                        st.session_state["holdings_data"][t] = etf_analysis.fetch_top_holdings(t)
                        prog.progress((i + 1) / len(tickers), text=f"Haettu: {t}")
                    prog.empty()
                    st.success("Valmis!")

                all_holdings = st.session_state.get("holdings_data", {})

                if not all_holdings:
                    st.info("Paina 'Hae ETF-omistusdata' -nappia aloittaaksesi analyysin.")
                else:
                    # Status-taulukko
                    fetch_status = [
                        {"Tikkeri": t,
                         "Omistuksia": len(all_holdings.get(t, pd.DataFrame())),
                         "Status": "OK" if t in all_holdings and not all_holdings[t].empty
                                   else "Ei dataa"}
                        for t in tickers
                    ]
                    st.dataframe(pd.DataFrame(fetch_status), use_container_width=True, hide_index=True)

                    # Yksittäiset omistukset expanderissa
                    with st.expander("Näytä ETF-omistukset tikkerieittäin"):
                        for t in tickers:
                            h = all_holdings.get(t, pd.DataFrame())
                            if not h.empty:
                                st.markdown(f"**{t}** — {len(h)} omistusta")
                                dh = h.copy()
                                dh["weight"] = dh["weight"].map(lambda x: f"{x*100:.2f}%")
                                c_h1, c_h2 = st.columns([1, 2])
                                with c_h1:
                                    st.dataframe(dh, use_container_width=True, hide_index=True)
                                with c_h2:
                                    st.plotly_chart(
                                        visualization.etf_holdings_chart(h, t),
                                        use_container_width=True,
                                        key="pc_15",
                                    )

                    etfs_with_data = [t for t in tickers
                                      if t in all_holdings and not all_holdings[t].empty]

                    if len(etfs_with_data) >= 2:
                        st.markdown("### Päällekkäisyysmatriisi")
                        matrix = etf_analysis.overlap_matrix(etfs_with_data, all_holdings)
                        st.plotly_chart(
                            visualization.overlap_matrix_chart(matrix),
                            use_container_width=True,
                            key="pc_16",
                        )

                        # Automaattiset varoitukset
                        st.markdown("### Huomiot")
                        warned = False
                        for i, a in enumerate(etfs_with_data):
                            for j, b in enumerate(etfs_with_data):
                                if i < j:
                                    ov = matrix.loc[a, b]
                                    if ov > 0.60:
                                        st.error(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                                 f"harkitse toisen korvaamista.")
                                        warned = True
                                    elif ov > 0.40:
                                        st.warning(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                                   f"melko korkea, tarkista onko molemmat tarpeen.")
                                        warned = True
                        if not warned:
                            st.success("Ei merkittäviä päällekkäisyyksiä (alle 40%).")

                        # Parivertailu
                        st.markdown("### Yksityiskohtainen parivertailu")
                        ov_c1, ov_c2 = st.columns(2)
                        etf_a = ov_c1.selectbox("ETF A", etfs_with_data, key="ov_a")
                        etf_b = ov_c2.selectbox("ETF B", etfs_with_data,
                                                index=min(1, len(etfs_with_data) - 1), key="ov_b")
                        if etf_a != etf_b:
                            ov_result = etf_analysis.compute_overlap(
                                all_holdings[etf_a], all_holdings[etf_b]
                            )
                            st.metric(
                                f"Päällekkäisyys: {etf_a} ∩ {etf_b}",
                                f"{ov_result['overlap_weight']*100:.1f}%",
                            )
                            st.plotly_chart(
                                visualization.overlap_detail_chart(
                                    ov_result["details"], etf_a, etf_b
                                ),
                                use_container_width=True,
                                key="pc_17",
                            )

            elif overlap_type == "Sektorit":
                # Sektoripäällekkäisyys - ei tarvitse erillistä dataa
                st.markdown("### Sektoripäällekkäisyys")
                st.caption("Vertailu perustuu ETF:ien sektorijakaumiin.")

                # Näytä sektorijakaumat ensin
                with st.expander("Näytä ETF-sektorijakaumat"):
                    for t in tickers:
                        sectors = etf_analysis.get_sector_weights(t)
                        if sectors:
                            st.markdown(f"**{t}**")
                            sector_df = pd.DataFrame(list(sectors.items()), columns=["Sektori", "Paino"])
                            sector_df["Paino"] = sector_df["Paino"].map(lambda x: f"{x*100:.1f}%")
                            st.dataframe(sector_df, use_container_width=True, hide_index=True)
                        else:
                            st.markdown(f"**{t}**: Ei sektoritietoja saatavilla")

                if len(tickers) >= 2:
                    matrix = etf_analysis.sector_overlap_matrix(tickers)
                    st.plotly_chart(
                        visualization.overlap_matrix_chart(matrix),
                        use_container_width=True,
                        key="pc_16_sector",
                    )

                    # Automaattiset varoitukset
                    st.markdown("### Huomiot")
                    warned = False
                    for i, a in enumerate(tickers):
                        for j, b in enumerate(tickers):
                            if i < j:
                                ov = matrix.loc[a, b]
                                if ov > 0.70:
                                    st.error(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                             f"hyvin samankaltainen sektorijakauma, hajautus vähäistä.")
                                    warned = True
                                elif ov > 0.50:
                                    st.warning(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                               f"melko samankaltainen sektorijakauma.")
                                    warned = True
                    if not warned:
                        st.success("Ei merkittäviä sektoripäällekkäisyyksiä (alle 50%).")

                    # Parivertailu
                    st.markdown("### Yksityiskohtainen parivertailu")
                    ov_c1, ov_c2 = st.columns(2)
                    etf_a = ov_c1.selectbox("ETF A", tickers, key="ov_a_sector")
                    etf_b = ov_c2.selectbox("ETF B", tickers,
                                            index=min(1, len(tickers) - 1), key="ov_b_sector")
                    if etf_a != etf_b:
                        sectors_a = etf_analysis.get_sector_weights(etf_a)
                        sectors_b = etf_analysis.get_sector_weights(etf_b)
                        ov_result = etf_analysis.compute_sector_overlap(sectors_a, sectors_b)
                        st.metric(
                            f"Sektoripäällekkäisyys: {etf_a} ∩ {etf_b}",
                            f"{ov_result['overlap_weight']*100:.1f}%",
                        )
                        st.plotly_chart(
                            visualization.sector_overlap_detail_chart(
                                ov_result["details"], etf_a, etf_b
                            ),
                            use_container_width=True,
                            key="pc_17_sector",
                        )

            else:  # Maat
                # Määrittely: Finland vs USA
                st.markdown("### Maapäällekkäisyys")
                st.caption("Vertailu perustuu ETF:ien maajakaumiin.")

                # Näytä maajakaumat ensin
                with st.expander("Näytä ETF-maajakaumat"):
                    for t in tickers:
                        countries = etf_analysis.get_country_weights(t)
                        if countries:
                            st.markdown(f"**{t}**")
                            country_df = pd.DataFrame(list(countries.items()), columns=["Maa", "Paino"])
                            country_df["Paino"] = country_df["Paino"].map(lambda x: f"{x*100:.1f}%")
                            st.dataframe(country_df, use_container_width=True, hide_index=True)
                        else:
                            st.markdown(f"**{t}**: Ei maatietoja saatavilla")

                if len(tickers) >= 2:
                    matrix = etf_analysis.country_overlap_matrix(tickers)
                    st.plotly_chart(
                        visualization.overlap_matrix_chart(matrix),
                        use_container_width=True,
                        key="pc_16_country",
                    )

                    # Automaattiset varoitukset
                    st.markdown("### Huomiot")
                    warned = False
                    for i, a in enumerate(tickers):
                        for j, b in enumerate(tickers):
                            if i < j:
                                ov = matrix.loc[a, b]
                                if ov > 0.70:
                                    st.error(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                             f"hyvin samankaltainen maajakauma, hajautus vähäistä.")
                                    warned = True
                                elif ov > 0.50:
                                    st.warning(f"**{a} ∩ {b}**: {ov*100:.1f}% — "
                                               f"melko samankaltainen maajakauma.")
                                    warned = True
                    if not warned:
                        st.success("Ei merkittäviä maapäällekkäisyyksiä (alle 50%).")

                    # Parivertailu
                    st.markdown("### Yksityiskohtainen parivertailu")
                    ov_c1, ov_c2 = st.columns(2)
                    etf_a = ov_c1.selectbox("ETF A", tickers, key="ov_a_country")
                    etf_b = ov_c2.selectbox("ETF B", tickers,
                                            index=min(1, len(tickers) - 1), key="ov_b_country")
                    if etf_a != etf_b:
                        countries_a = etf_analysis.get_country_weights(etf_a)
                        countries_b = etf_analysis.get_country_weights(etf_b)
                        ov_result = etf_analysis.compute_country_overlap(countries_a, countries_b)
                        st.metric(
                            f"Maapäällekkäisyys: {etf_a} ∩ {etf_b}",
                            f"{ov_result['overlap_weight']*100:.1f}%",
                        )
                        st.plotly_chart(
                            visualization.country_overlap_detail_chart(
                                ov_result["details"], etf_a, etf_b
                            ),
                            use_container_width=True,
                            key="pc_17_country",
                        )

# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 4 — OPTIMOINTI
# ══════════════════════════════════════════════════════════════════════════════

if selected_tab == TAB_NAMES[3]:
    st.markdown("### Portfolion optimointi")
    st.caption(
        "Modern Portfolio Theory (MPT) etsii painojakaumaa joka maksimoi tuoton "
        "suhteessa riskiin tai minimoi volatiliteetin."
    )

    with st.spinner("Lasketaan tehokasta rintamaa (4 000 simulaatiota)..."):
        frontier     = optimization.efficient_frontier(ret_df, n_portfolios=4000)
        opt_sharpe   = optimization.optimize_max_sharpe(ret_df, risk_free)
        opt_minvol   = optimization.optimize_min_volatility(ret_df)

    opt_subtabs = st.tabs([
        "Max Sharpe",
        "Min Volatiliteetti",
        "Tehokas rintama 2D",
        "Tehokas rintama 3D",
    ])

    with opt_subtabs[0]:
        st.markdown("**Paras Sharpe-luku** — maksimoi tuotto/riski-suhdetta.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Odotettu vuosituotto", f"{opt_sharpe['return']:.2%}")
        c2.metric("Volatiliteetti",       f"{opt_sharpe['volatility']:.2%}")
        c3.metric("Sharpe-luku",          f"{opt_sharpe['sharpe']:.2f}")

        w_df = pd.DataFrame([
            {
                "Tikkeri":           t,
                "Nykyinen paino":    f"{current_weights.get(t, 0):.1%}",
                "Optimaalinen paino":f"{w:.1%}",
                "Muutos":            f"{w - current_weights.get(t, 0):+.1%}",
            }
            for t, w in opt_sharpe["weights"].items()
        ])
        st.dataframe(w_df, use_container_width=True, hide_index=True)

        c_pie1, c_pie2 = st.columns(2)
        with c_pie1:
            st.plotly_chart(
                visualization.allocation_pie(current_weights, "Nykyinen"),
                use_container_width=True,
                key="pc_18",
            )
        with c_pie2:
            st.plotly_chart(
                visualization.allocation_pie(opt_sharpe["weights"], "Max Sharpe -optimoitu"),
                use_container_width=True,
                key="pc_19",
            )

    with opt_subtabs[1]:
        st.markdown("**Pienin mahdollinen riski** — sopii konservatiiviselle sijoittajalle.")
        c1, c2 = st.columns(2)
        c1.metric("Odotettu vuosituotto", f"{opt_minvol['return']:.2%}")
        c2.metric("Volatiliteetti",       f"{opt_minvol['volatility']:.2%}")

        w_df2 = pd.DataFrame([
            {
                "Tikkeri":            t,
                "Nykyinen paino":     f"{current_weights.get(t, 0):.1%}",
                "Optimaalinen paino": f"{w:.1%}",
                "Muutos":             f"{w - current_weights.get(t, 0):+.1%}",
            }
            for t, w in opt_minvol["weights"].items()
        ])
        st.dataframe(w_df2, use_container_width=True, hide_index=True)

        c_pie3, c_pie4 = st.columns(2)
        with c_pie3:
            st.plotly_chart(
                visualization.allocation_pie(current_weights, "Nykyinen"),
                use_container_width=True,
                key="pc_20",
            )
        with c_pie4:
            st.plotly_chart(
                visualization.allocation_pie(opt_minvol["weights"], "Min Vol -optimoitu"),
                use_container_width=True,
                key="pc_21",
            )

    with opt_subtabs[2]:
        current_stats = {
            "return":     analysis.annualized_return(port_ret_aligned),
            "volatility": analysis.annualized_volatility(port_ret_aligned),
        }
        st.plotly_chart(
            visualization.efficient_frontier_2d(frontier, current_stats),
            use_container_width=True,
            key="pc_22",
        )
        st.caption(
            "Jokainen piste = satunnaisesti generoitu salkku. Väri = Sharpe-luku. "
            "Punainen tähti = nykyinen salkkusi."
        )

    with opt_subtabs[3]:
        st.plotly_chart(
            visualization.efficient_frontier_3d(frontier),
            use_container_width=True,
            key="pc_23",
        )
        st.caption("3D-akseli: X = volatiliteetti, Y = tuotto, Z = Sharpe-luku.")

# ══════════════════════════════════════════════════════════════════════════════
# VÄLILEHTI 5 — ENNUSTEET
# ══════════════════════════════════════════════════════════════════════════════

if selected_tab == TAB_NAMES[4]:
    st.markdown("### Hintaennusteet & simulaatiot")

    # Yhteiset valinnat ennuste-välilehdelle
    fc_col1, fc_col2, fc_col3 = st.columns([2, 1, 1])
    
    # Vaihtoehto: yksittäinen arvopaperi tai koko salkku
    fc_target_type = fc_col1.radio(
        "Ennusteen kohde",
        ["Yksittäinen arvopaperi", "Koko salkku"],
        key="fc_target_type",
        horizontal=True,
    )
    
    if fc_target_type == "Yksittäinen arvopaperi":
        fc_ticker = fc_col2.selectbox("Valitse arvopaperi", tickers, key="fc_ticker_main")
        fc_days   = fc_col3.number_input("Ennustejakso (kaupankäyntipäiviä)", 20, 365, 60,
                                         key="fc_days_main")
    else:
        # Koko salkku - käytetään portfolion painotettua hintasarjaa
        st.caption(f"Koko salkku: {len(tickers)} arvopaperia, painotettu keskiarvo")
        fc_ticker = None  # Merkitaan että käytetään koko salkkua
        fc_days = fc_col3.number_input("Ennustejakso (kaupankäyntipäiviä)", 20, 365, 60,
                                         key="fc_days_main_port")

    fc_subtabs = st.tabs([
        "Monte Carlo",
        "ARIMA",
        "SARIMA",
        "SARIMAX",
        "Prophet",
        "Mallien vertailu",
    ])

    # ── Monte Carlo ─────────────────────────────────────────────────────────────
    with fc_subtabs[0]:
        st.markdown(
            "Simuloi tuhansilla satunnaisilla skenaarioilla mihin hinta voi päätyä. "
            "Perustuu historialliseen tuottoon ja volatiliteettiin."
        )
        n_sims = st.slider("Simulaatioiden määrä", 100, 2000, 500, 100)

        if st.button("Aja Monte Carlo", key="btn_mc"):
            with st.spinner("Simuloidaan..."):
                # Käytä joko yksittäistä arvopaperia tai koko salkkua
                if fc_target_type == "Yksittäinen arvopaperi":
                    sims = forecasting.monte_carlo_simulation(
                        prices[fc_ticker], days_ahead=fc_days, n_simulations=n_sims
                    )
                    ticker_label = fc_ticker
                    last_price = prices[fc_ticker].iloc[-1]
                else:
                    # Laske portfolion painotettu hinta
                    port_prices = sum(prices[t] * current_weights.get(t, 0) for t in tickers if t in prices.columns)
                    sims = forecasting.monte_carlo_simulation(
                        port_prices, days_ahead=fc_days, n_simulations=n_sims
                    )
                    ticker_label = "Salkku"
                    last_price = port_prices.iloc[-1]
            
            st.plotly_chart(
                visualization.monte_carlo_chart(sims, ticker_label),
                use_container_width=True,
                key="pc_24",
            )
            p10   = sims.iloc[-1].quantile(0.10)
            p50   = sims.iloc[-1].quantile(0.50)
            p90   = sims.iloc[-1].quantile(0.90)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Pessimistinen (10%)",  f"${p10:.2f}", f"{(p10/last_price-1)*100:+.1f}%")
            mc2.metric("Mediaani (50%)",       f"${p50:.2f}", f"{(p50/last_price-1)*100:+.1f}%")
            mc3.metric("Optimistinen (90%)",   f"${p90:.2f}", f"{(p90/last_price-1)*100:+.1f}%")

    # ── ARIMA ───────────────────────────────────────────────────────────────────
    with fc_subtabs[1]:
        st.markdown("""
        **ARIMA(p, d, q)** — klassinen aikasarjamalli.
        - **p** — AR-viive: kuinka monen päivän historia vaikuttaa ennusteeseen
        - **d** — differenssi: `1` poistaa trendin (suositeltava osakkeille)
        - **q** — MA-viive: korjaa aiempien ennustevirheiden vaikutusta
        """)
        ac1, ac2, ac3 = st.columns(3)
        a_p = ac1.number_input("p", 0, 20, 5, key="a_p",
                               help="Autoregressiivinen järjestys. Kokeile 1–10.")
        a_d = ac2.number_input("d", 0, 2,  1, key="a_d",
                               help="1 = poistaa trendin. Lähes aina 1 osakkeille.")
        a_q = ac3.number_input("q", 0, 10, 0, key="a_q",
                               help="MA-virheviiveet. Kokeile 0–3.")
        auto_a = st.checkbox("Auto-haku parhaimmille arvoille (vaatii pmdarima)", key="auto_a")

        if st.button("Laske ARIMA", key="btn_arima"):
            with st.spinner("Lasketaan..."):
                # Käytä joko yksittäistä arvopaperia tai koko salkkua
                if fc_target_type == "Yksittäinen arvopaperi":
                    price_series = prices[fc_ticker]
                    ticker_label = fc_ticker
                else:
                    price_series = sum(prices[t] * current_weights.get(t, 0) for t in tickers if t in prices.columns)
                    ticker_label = "Salkku"
                
                res = forecasting.arima_forecast(
                    price_series, days_ahead=fc_days,
                    p=a_p, d=a_d, q=a_q, auto=auto_a,
                )
            if res["error"]:
                st.error(f"Virhe: {res['error']}")
            else:
                rc1, rc2 = st.columns(2)
                rc1.metric("AIC", res["aic"])
                rc2.metric("BIC", res["bic"])
                st.plotly_chart(
                    visualization.forecast_chart_timeseries(price_series, res, ticker_label),
                    use_container_width=True,
                    key="pc_25",
                )

    # ── SARIMA ──────────────────────────────────────────────────────────────────
    with fc_subtabs[2]:
        st.markdown("""
        **SARIMA(p,d,q)(P,D,Q,m)** — ARIMA + kausikomponentti.
        Ei-kausiparametrit kuten ARIMA. Kausiparametrit:
        - **P** — kausiluonteinen AR-järjestys
        - **D** — kausiluonteinen differenssi (yleensä 0)
        - **Q** — kausiluonteinen MA-järjestys
        - **m** — kausijakson pituus: `5`=viikko · `21`=kk · `63`=kvartaali · `252`=vuosi
        """)
        sc1, sc2, sc3 = st.columns(3)
        s_p = sc1.number_input("p", 0, 10, 1, key="s_p")
        s_d = sc2.number_input("d", 0, 2,  1, key="s_d")
        s_q = sc3.number_input("q", 0, 10, 1, key="s_q")
        ss1, ss2, ss3, ss4 = st.columns(4)
        s_P = ss1.number_input("P", 0, 3,   1, key="s_P")
        s_D = ss2.number_input("D", 0, 1,   0, key="s_D")
        s_Q = ss3.number_input("Q", 0, 3,   1, key="s_Q")
        s_m = ss4.number_input("m", 2, 252, 5, key="s_m")
        auto_s = st.checkbox("Auto-haku (pmdarima)", key="auto_s")

        if st.button("Laske SARIMA", key="btn_sarima"):
            with st.spinner("Lasketaan... (voi kestää hetken)"):
                # Käytä joko yksittäistä arvopaperia tai koko salkkua
                if fc_target_type == "Yksittäinen arvopaperi":
                    price_series = prices[fc_ticker]
                    ticker_label = fc_ticker
                else:
                    price_series = sum(prices[t] * current_weights.get(t, 0) for t in tickers if t in prices.columns)
                    ticker_label = "Salkku"
                
                res = forecasting.sarima_forecast(
                    price_series, days_ahead=fc_days,
                    p=s_p, d=s_d, q=s_q,
                    P=s_P, D=s_D, Q=s_Q, m=s_m, auto=auto_s,
                )
            if res["error"]:
                st.error(f"Virhe: {res['error']}")
            else:
                rc1, rc2 = st.columns(2)
                rc1.metric("AIC", res["aic"])
                rc2.metric("BIC", res["bic"])
                st.plotly_chart(
                    visualization.forecast_chart_timeseries(price_series, res, ticker_label),
                    use_container_width=True,
                    key="pc_26",
                )

    # ── SARIMAX ─────────────────────────────────────────────────────────────────
    with fc_subtabs[3]:
        # SARIMAX ei ole käytettävissä koko salkulle, koska se vaatii ulkoisia muuttujia
        if fc_target_type == "Koko salkku":
            st.warning("SARIMAX-ennusteet eivät ole käytettävissä koko salkulle. Valitse 'Yksittäinen arvopaperi' yllä.")
        else:
            st.markdown("""
            **SARIMAX** — SARIMA + ulkoiset selittävät muuttujat.
            Valitse alla mitä ulkoisia muuttujia käytät ja miten ennustat niiden tulevat arvot.
            """)
            xc1, xc2, xc3 = st.columns(3)
            sx_p = xc1.number_input("p", 0, 10, 1, key="sx_p")
            sx_d = xc2.number_input("d", 0, 2,  1, key="sx_d")
            sx_q = xc3.number_input("q", 0, 10, 1, key="sx_q")
            xs1, xs2, xs3, xs4 = st.columns(4)
            sx_P = xs1.number_input("P", 0, 3,   1, key="sx_P")
            sx_D = xs2.number_input("D", 0, 1,   0, key="sx_D")
            sx_Q = xs3.number_input("Q", 0, 3,   1, key="sx_Q")
            sx_m = xs4.number_input("m", 2, 252, 5, key="sx_m")

            st.markdown("**Ulkoiset muuttujat:**")
            use_spy   = st.checkbox("VOO (markkinatuotto)",     value=True,  key="sx_spy")
            use_vix   = st.checkbox("^VIX (volatiliteetti)",    value=False, key="sx_vix")
            use_dxy   = st.checkbox("DX-Y.NYB (dollari-indeksi)", value=False, key="sx_dxy")
            use_rates = st.checkbox("^TNX (10v US-korko)",      value=False, key="sx_rates")

            naive_mode = st.radio(
                "Tulevat arvot ennustejaksolle:",
                ["Käytä viimeistä tunnettua arvoa (naive)", "Syötä käsin"],
                key="sx_mode",
            )
            manual_vals = {}
            if naive_mode == "Syötä käsin":
                if use_spy:   manual_vals["VOO"]   = st.number_input("VOO arvo",   value=580.0, key="sx_spy_v")
                if use_vix:   manual_vals["VIX"]   = st.number_input("VIX arvo",   value=18.0,  key="sx_vix_v")
                if use_dxy:   manual_vals["DXY"]   = st.number_input("DXY arvo",   value=104.0, key="sx_dxy_v")
                if use_rates: manual_vals["US10Y"] = st.number_input("US10Y (%)",  value=4.3,   key="sx_rt_v")

            if st.button("Laske SARIMAX", key="btn_sarimax"):
                exog_map = {}
                if use_spy:   exog_map["VOO"]   = "VOO"
                if use_vix:   exog_map["^VIX"]  = "VIX"
                if use_dxy:   exog_map["DX-Y.NYB"] = "DXY"
                if use_rates: exog_map["^TNX"]  = "US10Y"

                if not exog_map:
                    st.warning("Valitse vähintään yksi ulkoinen muuttuja.")
                else:
                    with st.spinner("Ladataan ulkoinen data ja lasketaan..."):
                        try:
                            exog_tickers = list(exog_map.keys())
                            exog_prices  = data.fetch_prices(exog_tickers, period_years=period)
                            exog_prices  = exog_prices.rename(columns=exog_map)

                            future_dates = pd.date_range(
                                prices[fc_ticker].index[-1], periods=fc_days + 1, freq="B"
                            )[1:]

                            if naive_mode == "Käytä viimeistä tunnettua arvoa (naive)":
                                exog_future = pd.DataFrame(
                                    {c: [exog_prices[c].iloc[-1]] * fc_days
                                     for c in exog_prices.columns},
                                    index=future_dates,
                                )
                            else:
                                exog_future = pd.DataFrame(
                                    {c: [manual_vals.get(c, exog_prices[c].iloc[-1])] * fc_days
                                     for c in exog_prices.columns},
                                    index=future_dates,
                                )

                            res = forecasting.sarimax_forecast(
                                prices[fc_ticker],
                                exog_train=exog_prices, exog_future=exog_future,
                                days_ahead=fc_days,
                                p=sx_p, d=sx_d, q=sx_q,
                                P=sx_P, D=sx_D, Q=sx_Q, m=sx_m,
                            )
                            if res["error"]:
                                st.error(f"Virhe: {res['error']}")
                            else:
                                rc1, rc2, rc3 = st.columns(3)
                                rc1.metric("AIC", res["aic"])
                                rc2.metric("BIC", res["bic"])
                                rc3.metric("Selittäjät", ", ".join(res["exog_variables"]))
                                st.plotly_chart(
                                    visualization.forecast_chart_timeseries(
                                        prices[fc_ticker], res, fc_ticker
                                    ),
                                    use_container_width=True,
                                    key="pc_27",
                                )
                        except Exception as e:
                            st.error(f"Virhe: {e}")

    # ── Prophet ─────────────────────────────────────────────────────────────────
    with fc_subtabs[4]:
        st.markdown(
            "**Prophet** (Meta) — tunnistaa automaattisesti trendit, "
            "viikonpäivävaihtelun ja vuosikausivaihtelun."
        )
        if st.button("Laske Prophet-ennuste", key="btn_prophet"):
            with st.spinner("Lasketaan..."):
                # Käytä joko yksittäistä arvopaperia tai koko salkkua
                if fc_target_type == "Yksittäinen arvopaperi":
                    price_series = prices[fc_ticker]
                    ticker_label = fc_ticker
                else:
                    price_series = sum(prices[t] * current_weights.get(t, 0) for t in tickers if t in prices.columns)
                    ticker_label = "Salkku"
                
                fc_prophet = forecasting.prophet_forecast(price_series, days_ahead=fc_days)
            if fc_prophet is None:
                st.warning("Prophet ei ole asennettu: `pip install prophet`")
            else:
                st.plotly_chart(
                    visualization.forecast_chart(price_series, fc_prophet, ticker_label),
                    use_container_width=True,
                    key="pc_28",
                )

    # ── Mallien vertailu ─────────────────────────────────────────────────────────
    with fc_subtabs[5]:
        st.markdown("""
        Aja ARIMA ja SARIMA rinnakkain ja vertaa AIC/BIC-arvoja.
        **Pienempi AIC/BIC = parempi sopivuus dataan.**
        """)
        vc1, vc2 = st.columns(2)
        cmp_ap = vc1.number_input("ARIMA p", 0, 10, 5, key="cmp_ap")
        cmp_aq = vc1.number_input("ARIMA q", 0, 10, 0, key="cmp_aq")
        cmp_sp = vc2.number_input("SARIMA p", 0, 10, 1, key="cmp_sp")
        cmp_sm = vc2.number_input("SARIMA m", 2, 252, 5, key="cmp_sm")

        if st.button("Vertaile malleja", key="btn_compare"):
            with st.spinner("Lasketaan molemmat mallit..."):
                # Käytä joko yksittäistä arvopaperia tai koko salkkua
                if fc_target_type == "Yksittäinen arvopaperi":
                    price_series = prices[fc_ticker]
                    ticker_label = fc_ticker
                else:
                    price_series = sum(prices[t] * current_weights.get(t, 0) for t in tickers if t in prices.columns)
                    ticker_label = "Salkku"
                
                results = forecasting.compare_models(
                    price_series, days_ahead=fc_days,
                    arima_params={"p": cmp_ap, "d": 1, "q": cmp_aq},
                    sarima_params={"p": cmp_sp, "d": 1, "q": 1,
                                   "P": 1, "D": 0, "Q": 1, "m": cmp_sm},
                )
            ok = [r for r in results if not r.get("error")]
            if ok:
                summary = [{"Malli": r["model"], "AIC": r["aic"], "BIC": r["bic"]} for r in ok]
                best    = min(summary, key=lambda x: x["AIC"])["Malli"]
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
                st.success(f"Paras AIC: **{best}**")
                st.plotly_chart(
                    visualization.forecast_comparison_chart(price_series, ok, ticker_label),
                    use_container_width=True,
                    key="pc_29",
                )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Data: Yahoo Finance · Optimointi: MPT (scipy) · "
    "Ennusteet: ARIMA / SARIMA / SARIMAX (statsmodels) · Prophet (Meta) · Monte Carlo"
)
