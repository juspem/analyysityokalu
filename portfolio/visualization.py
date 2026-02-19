import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


COLORS = px.colors.qualitative.Plotly


def cumulative_returns_chart(
    port_cum: pd.Series, bench_cum: pd.Series = None, bench_name: str = "Vertailuindeksi"
) -> go.Figure:
    fig = go.Figure()
    if bench_cum is not None and len(bench_cum) > 0:
        fig.add_trace(go.Scatter(
            x=bench_cum.index, y=bench_cum * 100,
            name=bench_name, line=dict(color="#F5A623", width=2)
        ))
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum * 100,
        name="Salkku", line=dict(color="#4C9BE8", width=2.5)
    ))
    fig.update_layout(
        title="Kumulatiivinen tuotto (%)",
        yaxis_title="Tuotto %", xaxis_title="",
        template="plotly_dark", hovermode="x unified"
    )
    return fig


def allocation_pie(weights: dict, title: str = "Hajautus") -> go.Figure:
    labels = list(weights.keys())
    values = [v * 100 for v in weights.values()]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4, textinfo="label+percent",
        marker=dict(colors=COLORS)
    ))
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def efficient_frontier_2d(frontier_df: pd.DataFrame, current: dict = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe"),
            size=4, opacity=0.6
        ),
        name="Simuloidut salkut",
        hovertemplate="Vol: %{x:.1f}%<br>Tuotto: %{y:.1f}%<extra></extra>"
    ))
    if current:
        fig.add_trace(go.Scatter(
            x=[current["volatility"] * 100], y=[current["return"] * 100],
            mode="markers", marker=dict(color="red", size=14, symbol="star"),
            name="Nykyinen salkku"
        ))
    fig.update_layout(
        title="Tehokas rintama (Monte Carlo)",
        xaxis_title="Volatiliteetti %", yaxis_title="Odotettu vuosituotto %",
        template="plotly_dark"
    )
    return fig


def efficient_frontier_3d(frontier_df: pd.DataFrame) -> go.Figure:
    """3D-visualisointi: tuotto, volatiliteetti ja Sharpe."""
    fig = go.Figure(go.Scatter3d(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["return"] * 100,
        z=frontier_df["sharpe"],
        mode="markers",
        marker=dict(
            size=2.5,
            color=frontier_df["sharpe"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Sharpe"),
            opacity=0.7
        ),
        hovertemplate="Vol: %{x:.1f}%<br>Tuotto: %{y:.1f}%<br>Sharpe: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title="3D Tehokas Rintama",
        scene=dict(
            xaxis_title="Volatiliteetti %",
            yaxis_title="Tuotto %",
            zaxis_title="Sharpe-luku",
            bgcolor="#1e1e2e"
        ),
        template="plotly_dark",
        height=650
    )
    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(title="Korrelaatiomatriisi", template="plotly_dark")
    return fig


def monte_carlo_chart(simulations: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    sample = simulations.iloc[:, :100]
    for col in sample.columns:
        fig.add_trace(go.Scatter(
            x=sample.index, y=sample[col],
            line=dict(width=0.5, color="rgba(100, 180, 255, 0.15)"),
            showlegend=False, hoverinfo="skip"
        ))
    p10 = simulations.quantile(0.10, axis=1)
    p50 = simulations.quantile(0.50, axis=1)
    p90 = simulations.quantile(0.90, axis=1)
    fig.add_trace(go.Scatter(x=p90.index, y=p90, name="90. persentiili",
                              line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=p50.index, y=p50, name="Mediaani",
                              line=dict(color="white", width=2.5)))
    fig.add_trace(go.Scatter(x=p10.index, y=p10, name="10. persentiili",
                              line=dict(color="red", width=2)))
    fig.update_layout(
        title=f"Monte Carlo -simulaatio: {ticker} (500 skenaariota)",
        yaxis_title="Hinta ($)", template="plotly_dark"
    )
    return fig


def drawdown_chart(returns: pd.Series) -> go.Figure:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max * 100
    fig = go.Figure(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy", line=dict(color="#EF553B"),
        name="Drawdown"
    ))
    fig.update_layout(
        title="Historiallinen Drawdown (%)",
        yaxis_title="Drawdown %", template="plotly_dark"
    )
    return fig


def rolling_sharpe_chart(returns: pd.Series, window: int = 60) -> go.Figure:
    rolling = returns.rolling(window)
    sharpe = (rolling.mean() * 252 - 0.045) / (rolling.std() * np.sqrt(252))
    fig = go.Figure(go.Scatter(
        x=sharpe.index, y=sharpe,
        line=dict(color="#AB63FA", width=2), name=f"Sharpe ({window}pv rullaava)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Sharpe = 1")
    fig.update_layout(title=f"Rullaava Sharpe-luku ({window} pv)",
                      template="plotly_dark")
    return fig


def forecast_chart(
    historical: pd.Series, forecast: pd.DataFrame, ticker: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical,
        name="Historia", line=dict(color="#00CC96")
    ))
    if forecast is not None:
        future = forecast[forecast.index > historical.index[-1]]
        fig.add_trace(go.Scatter(
            x=future.index, y=future["yhat"],
            name="Ennuste", line=dict(color="orange", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]).index,
            y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Luottamusväli"
        ))
    fig.update_layout(title=f"{ticker} – Prophet-ennuste", template="plotly_dark")
    return fig


def forecast_chart_timeseries(
    historical: pd.Series, result: dict, ticker: str
) -> go.Figure:
    """Ennustekaavio ARIMA/SARIMA/SARIMAX-tuloksille (dict-muoto)."""
    fig = go.Figure()
    # Näytä vain viimeiset 180 pv historiaa
    hist = historical.iloc[-180:]
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist,
        name="Historia", line=dict(color="#00CC96", width=2)
    ))
    yhat = result["yhat"]
    yhat_lower = result.get("yhat_lower")
    yhat_upper = result.get("yhat_upper")

    fig.add_trace(go.Scatter(
        x=yhat.index, y=yhat,
        name="Ennuste", line=dict(color="orange", width=2.5, dash="dash")
    ))
    if yhat_lower is not None and yhat_upper is not None:
        fig.add_trace(go.Scatter(
            x=list(yhat_upper.index) + list(yhat_lower.index[::-1]),
            y=list(yhat_upper) + list(yhat_lower[::-1]),
            fill="toself", fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="95% luottamusväli"
        ))
    fig.update_layout(
        title=f"{ticker} — {result['model']}",
        yaxis_title="Hinta ($)", template="plotly_dark", hovermode="x unified"
    )
    return fig


def forecast_comparison_chart(
    historical: pd.Series, results: list, ticker: str
) -> go.Figure:
    """Vertaa useita ennustemalleja samassa kaaviossa."""
    fig = go.Figure()
    hist = historical.iloc[-180:]
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist,
        name="Historia", line=dict(color="#00CC96", width=2.5)
    ))
    palette = ["#FFA500", "#AB63FA", "#EF553B", "#19D3F3"]
    for i, res in enumerate(results):
        if res.get("yhat") is not None:
            fig.add_trace(go.Scatter(
                x=res["yhat"].index, y=res["yhat"],
                name=res["model"],
                line=dict(color=palette[i % len(palette)], width=2, dash="dash")
            ))
    fig.update_layout(
        title=f"{ticker} — mallien vertailu",
        yaxis_title="Hinta ($)", template="plotly_dark", hovermode="x unified"
    )
    return fig


def sector_treemap(holdings: list[dict]) -> go.Figure:
    """Treemap sektorijakauma arvon mukaan."""
    df = pd.DataFrame(holdings)
    if "sector" not in df.columns or "value" not in df.columns:
        return go.Figure()
    fig = px.treemap(df, path=["sector", "ticker"], values="value",
                     color="value", color_continuous_scale="Blues")
    fig.update_layout(title="Sektorijakauma", template="plotly_dark")
    return fig


# ── ETF-päällekkäisyys ─────────────────────────────────────────────────────────

def overlap_matrix_chart(matrix_df: pd.DataFrame) -> go.Figure:
    """Heatmap ETFien välisistä omistuspäällekkäisyyksistä."""
    labels = list(matrix_df.columns)
    values = matrix_df.values * 100

    fig = go.Figure(go.Heatmap(
        z=values,
        x=labels, y=labels,
        colorscale="RdYlGn_r",
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in values],
        texttemplate="%{text}",
        colorbar=dict(title="Päällekkäisyys %"),
        hovertemplate="%{y} ∩ %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="ETF-omistusten päällekkäisyysmatriisi",
        template="plotly_dark",
        height=max(350, len(labels) * 80),
        xaxis=dict(side="bottom"),
    )
    return fig


def overlap_detail_chart(details_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    """Pylväskaavio kahden ETF:n yhteisistä omistuksista (yritykset/osakkeet)."""
    if details_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ei yhteisiä omistuksia", template="plotly_dark")
        return fig

    top = details_df.head(20).sort_values("Päällekkäisyys")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["Osake"], x=top["Paino A"] * 100,
        name=ticker_a, orientation="h",
        marker_color="#00CC96", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=top["Osake"], x=top["Paino B"] * 100,
        name=ticker_b, orientation="h",
        marker_color="#AB63FA", opacity=0.85,
    ))
    fig.update_layout(
        title=f"Yhteiset omistukset: {ticker_a} vs {ticker_b} (top 20)",
        xaxis_title="Paino %",
        barmode="group",
        template="plotly_dark",
        height=max(400, len(top) * 28),
    )
    return fig


def sector_overlap_detail_chart(details_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    """Pylväskaavio kahden ETF:n yhteisistä sektoreista."""
    if details_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ei yhteisiä sektoreita", template="plotly_dark")
        return fig

    top = details_df.sort_values("Päällekkäisyys")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["Sektori"], x=top["Paino A"] * 100,
        name=ticker_a, orientation="h",
        marker_color="#00CC96", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=top["Sektori"], x=top["Paino B"] * 100,
        name=ticker_b, orientation="h",
        marker_color="#AB63FA", opacity=0.85,
    ))
    fig.update_layout(
        title=f"Yhteiset sektorit: {ticker_a} vs {ticker_b}",
        xaxis_title="Paino %",
        barmode="group",
        template="plotly_dark",
        height=max(350, len(top) * 40),
    )
    return fig


def country_overlap_detail_chart(details_df: pd.DataFrame, ticker_a: str, ticker_b: str) -> go.Figure:
    """Pylväskaavio kahden ETF:n yhteisistä maista."""
    if details_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ei yhteisiä maita", template="plotly_dark")
        return fig

    top = details_df.sort_values("Päällekkäisyys")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["Maa"], x=top["Paino A"] * 100,
        name=ticker_a, orientation="h",
        marker_color="#00CC96", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=top["Maa"], x=top["Paino B"] * 100,
        name=ticker_b, orientation="h",
        marker_color="#AB63FA", opacity=0.85,
    ))
    fig.update_layout(
        title=f"Yhteiset maat: {ticker_a} vs {ticker_b}",
        xaxis_title="Paino %",
        barmode="group",
        template="plotly_dark",
        height=max(350, len(top) * 40),
    )
    return fig


def etf_holdings_chart(holdings_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Donitsikaavio ETF:n top-omistuksista."""
    if holdings_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: ei omistustietoja saatavilla", template="plotly_dark")
        return fig

    top = holdings_df.head(15).copy()
    other_w = max(0, 1.0 - top["weight"].sum())
    if other_w > 0.001:
        top = pd.concat([top, pd.DataFrame([{"symbol": "Muut", "name": "Muut", "weight": other_w}])],
                        ignore_index=True)
    fig = go.Figure(go.Pie(
        labels=top["symbol"],
        values=(top["weight"] * 100).round(2),
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>",
        marker=dict(colors=px.colors.qualitative.Plotly + px.colors.qualitative.Dark24),
    ))
    fig.update_layout(
        title=f"{ticker} — top {len(holdings_df.head(15))} omistusta",
        template="plotly_dark",
    )
    return fig


# ── Maajakauma ─────────────────────────────────────────────────────────────────

def country_choropleth(country_weights: pd.Series, title: str = "Maantieteellinen hajautus") -> go.Figure:
    """Maailmankartta portfolion maajakaumasta."""
    if country_weights.empty:
        fig = go.Figure()
        fig.update_layout(title="Ei maajakaumatietoja", template="plotly_dark")
        return fig

    df = country_weights.reset_index()
    df.columns = ["country", "weight"]
    df["weight_pct"] = (df["weight"] * 100).round(2)
    df["text"] = df.apply(lambda r: f"{r['country']}: {r['weight_pct']:.1f}%", axis=1)

    # "Other"-kategoria ei näy kartalla, poistetaan
    df = df[df["country"] != "Other"]

    fig = go.Figure(go.Choropleth(
        locations=df["country"],
        locationmode="country names",
        z=df["weight_pct"],
        text=df["text"],
        hovertemplate="%{text}<extra></extra>",
        colorscale="Blues",
        colorbar=dict(title="Paino %"),
        marker_line_color="rgba(255,255,255,0.3)",
        marker_line_width=0.5,
    ))
    fig.update_layout(
        title=title,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
            landcolor="#2a2a3e",
            oceancolor="#1a1a2e",
            showocean=True,
            coastlinecolor="rgba(255,255,255,0.2)",
            showlakes=False,
        ),
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def country_bar_chart(country_weights: pd.Series) -> go.Figure:
    """Vaakapylväskaavio maajakaumasta."""
    if country_weights.empty:
        return go.Figure()

    df = country_weights.head(15).sort_values()
    colors = [f"rgba(0, {int(150 + 105 * v / df.max())}, {int(200 * v / df.max())}, 0.85)"
              for v in df.values]

    fig = go.Figure(go.Bar(
        x=df.values * 100,
        y=df.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df.values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Maajakauma (top 15)",
        xaxis_title="Paino %",
        template="plotly_dark",
        height=max(350, len(df) * 32),
        margin=dict(l=120),
    )
    return fig


# ── Sektorijakauma ─────────────────────────────────────────────────────────────

SECTOR_COLORS = {
    "Technology": "#636EFA",
    "Healthcare": "#EF553B",
    "Financial Services": "#00CC96",
    "Consumer Discretionary": "#AB63FA",
    "Industrials": "#FFA15A",
    "Communication Services": "#19D3F3",
    "Consumer Staples": "#FF6692",
    "Energy": "#B6E880",
    "Utilities": "#FF97FF",
    "Materials": "#FECB52",
    "Real Estate": "#72B7B2",
    "Basic Materials": "#FECB52",
}


def sector_sunburst(sector_weights: pd.Series, ticker_sectors: dict[str, dict]) -> go.Figure:
    """
    Aurinkokaavio (sunburst): ulkokehä = sektorit, sisäkehä = ETF/osake.
    ticker_sectors = {ticker: {sector: weight}}
    """
    ids, labels, parents, values, colors = [], [], [], [], []

    # Juuritaso
    for sector, total_w in sector_weights.items():
        ids.append(sector)
        labels.append(sector)
        parents.append("")
        values.append(round(total_w * 100, 2))
        colors.append(SECTOR_COLORS.get(sector, "#AAAAAA"))

    # Lapsisolmut per tikkeri
    for ticker, sectors in ticker_sectors.items():
        for sector, w in sectors.items():
            if w > 0:
                node_id = f"{sector}__{ticker}"
                ids.append(node_id)
                labels.append(ticker)
                parents.append(sector)
                values.append(round(w * 100, 2))
                colors.append(SECTOR_COLORS.get(sector, "#AAAAAA"))

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=1)),
        branchvalues="total",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        insidetextorientation="radial",
    ))
    fig.update_layout(
        title="Sektorijakauma — portfoliotaso",
        template="plotly_dark",
        height=550,
        margin=dict(t=60, l=0, r=0, b=0),
    )
    return fig


def sector_bar_chart(sector_weights: pd.Series, compare: pd.Series = None) -> go.Figure:
    """Pylväskaavio sektorijakaumasta, valinnainen SPY-vertailu."""
    fig = go.Figure()

    sorted_sectors = sector_weights.sort_values(ascending=False)

    fig.add_trace(go.Bar(
        x=sorted_sectors.index,
        y=sorted_sectors.values * 100,
        name="Portfoliosi",
        marker_color=[SECTOR_COLORS.get(s, "#888") for s in sorted_sectors.index],
        text=[f"{v*100:.1f}%" for v in sorted_sectors.values],
        textposition="outside",
    ))

    if compare is not None and not compare.empty:
        compare_aligned = compare.reindex(sorted_sectors.index, fill_value=0)
        fig.add_trace(go.Bar(
            x=compare_aligned.index,
            y=compare_aligned.values * 100,
            name="SPY (vertailu)",
            marker_color="rgba(255,255,255,0.25)",
            text=[f"{v*100:.1f}%" for v in compare_aligned.values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Sektorijakauma",
        yaxis_title="Paino %",
        barmode="group",
        template="plotly_dark",
        height=450,
        xaxis_tickangle=-30,
    )
    return fig
