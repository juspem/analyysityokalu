**Portfolioanalyysityökalu**

Streamlit-sovellus sijoitussalkun analysointiin. Tuotto, riski, hajautus, optimointi ja hintaennusteet.

**Välilehdet**

Yhteenveto: avainmetriikat, kumulatiivinen tuotto vertailuindeksiä vastaan, hajautuskaavio, osakekohtaiset tiedot.
Tuotto ja riski: kumulatiivinen tuotto per arvopaperi, drawdown-käyrä ja pahimmat päivät, rullaava Sharpe ja volatiliteetti, korrelaatiomatriisi.
Hajautus: maa- ja sektorijakauma karttana ja taulukkona. ETF:t puretaan todellisiin sijoituskohteisiin, joten indeksirahastoista koostuva salkku näyttää oikean jakauman. Mukana ETF-päällekkäisyysanalyysi.
Optimointi: MPT-optimointi maksimi-Sharpelle ja minimivolatiliteetille, vertailu nykyisiin painoihin, tehokas rintama 2D ja 3D 4 000 simuloidusta salkusta.
Ennusteet: ARIMA, SARIMA, SARIMAX, Prophet ja Monte Carlo. SARIMAX tukee ulkoisia selittäjiä (markkinatuotto, VIX, dollari-indeksi, US 10v korko). Mallien vertailu AIC- ja BIC-arvoilla.

**Metriikat**

Vuosituotto, volatiliteetti, Sharpe, Sortino, max drawdown ja vertailuindeksin vastaavat luvut.

**Rakenne**

Logiikka on portfolio-paketissa: data, analysis, optimization, forecasting, visualization ja etf_analysis. main.py sisältää vain käyttöliittymän.

**Käynnistys**

pip install -r requirements.txt

streamlit run main.py

**Teknologiat**

Python, Streamlit, pandas, NumPy, scipy, statsmodels, Plotly, yfinance.
