"""Interactive Streamlit Portfolio Risk Engine.

Run from this directory with: streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk_engine import (
    ASSET_CLASSES,
    DEFAULT_TICKERS,
    HISTORICAL_EVENTS,
    MARKET_BETAS,
    asset_metrics,
    compute_risk_table,
    correlation_stress,
    download_yahoo_prices,
    historical_event_returns,
    load_prices,
    portfolio_metrics,
    portfolio_returns,
    reverse_stress,
    risk_contributions,
    rolling_historical_backtest,
    simple_returns,
    volatility_stress,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "portfolio_prices.csv"
META_PATH = BASE_DIR / "data_source.json"

st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Portfolio Risk Engine")
st.caption("Returns · Volatilité · Sharpe · VaR · Expected Shortfall · Monte Carlo · Backtesting · Stress tests")


def read_metadata() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "kind": "unknown",
        "source": "Source non documentée",
        "note": "Vérifiez la provenance du fichier CSV.",
    }


def save_metadata(payload: dict) -> None:
    META_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def euro(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.0f} €".replace(",", " ")


def pct(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}%}"


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and abs(denominator) > 1e-10 else float("nan")


def formatted_risk_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for col in [c for c in result.columns if "(€)" in c]:
        result[col] = result[col].map(euro)
    return result


with st.sidebar:
    st.header("1. Données de marché")
    st.write("L’univers comprend des actions/ETF actions, des obligations, des matières premières et de l’immobilier coté.")
    start_date = st.date_input(
        "Début du téléchargement Yahoo Finance",
        value=date(2006, 1, 1),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        help="2006 permet aussi de couvrir l'épisode de crise de 2008 pour les actifs disposant de cet historique.",
    )
    if st.button("Télécharger les cours ajustés Yahoo Finance", type="primary", use_container_width=True):
        try:
            with st.spinner("Téléchargement des cours ajustés depuis Yahoo Finance…"):
                fresh = download_yahoo_prices(list(ASSET_CLASSES.keys()), start=start_date, end=date.today())
                fresh.to_csv(CSV_PATH, index_label="Date", date_format="%Y-%m-%d")
                save_metadata({
                    "kind": "yahoo_finance",
                    "source": "Yahoo Finance via yfinance",
                    "price_field": "Adjusted Close (Close avec auto_adjust=True)",
                    "start_date": str(fresh.index.min().date()),
                    "end_date": str(fresh.index.max().date()),
                    "tickers": list(fresh.columns),
                    "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "note": "Calendrier aligné ; valeurs manquantes intermédiaires propagées vers l'avant, sans remplissage avant la première cotation.",
                })
            st.success(f"{len(fresh):,} lignes téléchargées.".replace(",", " "))
            st.rerun()
        except Exception as exc:  # show source failure to the user, keep local CSV available
            st.error(f"Téléchargement impossible : {exc}")

    st.divider()
    st.header("2. Portefeuille")
    try:
        all_prices = load_prices(CSV_PATH)
    except Exception as exc:
        st.error(f"Impossible de charger {CSV_PATH.name} : {exc}")
        st.stop()

    available_tickers = [ticker for ticker in ASSET_CLASSES if ticker in all_prices.columns]
    default_available = [ticker for ticker in DEFAULT_TICKERS if ticker in available_tickers]
    if len(default_available) < min(6, len(available_tickers)):
        default_available = available_tickers[: min(6, len(available_tickers))]

    selected_tickers = st.multiselect(
        "Tickers (choisir au moins 6 actifs)",
        options=available_tickers,
        default=default_available,
        help="Les trois classes d'actifs minimales du projet sont Actions, Obligations US et matières premières.",
    )
    n_selected = len(selected_tickers)
    if n_selected:
        equal_weight = 100.0 / n_selected
        default_weights = [round(equal_weight, 4)] * n_selected
        default_weights[-1] = round(100.0 - sum(default_weights[:-1]), 4)
        default_shocks = []
        for ticker in selected_tickers:
            if ASSET_CLASSES.get(ticker, "") in ("Actions", "ETF actions", "Immobilier coté"):
                default_shocks.append(-15.0 if ticker != "SPY" else -12.0)
            elif ASSET_CLASSES.get(ticker, "") == "Obligations US":
                default_shocks.append(-5.0)
            elif ticker == "GLD":
                default_shocks.append(0.0)
            else:
                default_shocks.append(-12.0)
        input_df = pd.DataFrame({
            "Ticker": selected_tickers,
            "Classe d'actifs": [ASSET_CLASSES.get(t, "Non renseignée") for t in selected_tickers],
            "Poids (%)": default_weights,
            "Choc hypothétique (%)": default_shocks,
        })
        edited_alloc = st.data_editor(
            input_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="alloc_" + "_".join(selected_tickers),
            disabled=["Ticker", "Classe d'actifs"],
            column_config={
                "Poids (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
                "Choc hypothétique (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=1.0, format="%.1f"),
            },
        )
    else:
        edited_alloc = pd.DataFrame(columns=["Ticker", "Classe d'actifs", "Poids (%)", "Choc hypothétique (%)"])

    st.divider()
    st.header("3. Paramètres de risque")
    portfolio_value = st.number_input("Valeur du portefeuille (€)", min_value=1_000.0, max_value=1_000_000_000_000.0, value=1_000_000.0, step=50_000.0)
    risk_free_rate = st.number_input("Taux sans risque annualisé (%)", min_value=-10.0, max_value=30.0, value=3.0, step=0.25) / 100.0
    focus_confidence_label = st.selectbox("Niveau de confiance mis en avant", options=["95 %", "99 %"], index=1)
    focus_confidence = 0.95 if focus_confidence_label == "95 %" else 0.99
    focus_horizon = st.selectbox("Horizon mis en avant", options=[1, 10], index=0, format_func=lambda x: f"{x} jour(s)")
    lookback = st.slider("Fenêtre d'estimation du backtest (jours)", min_value=250, max_value=1_000, value=500, step=50)
    n_simulations = st.select_slider("Simulations Monte Carlo", options=[2_000, 5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)
    seed = st.number_input("Seed reproductible", min_value=0, max_value=2_147_483_647, value=42, step=1)
    st.divider()
    st.header("4. Paramètres des stress tests")
    event_name = st.selectbox("Épisode historique", options=list(HISTORICAL_EVENTS.keys()), index=1)
    target_correlation = st.slider("Corrélation stressée entre actifs", min_value=0.80, max_value=1.00, value=0.90, step=0.05)
    volatility_multiplier = st.slider("Multiplicateur de volatilité", min_value=1.5, max_value=3.0, value=2.0, step=0.25)
    reverse_loss_pct = st.number_input("Seuil de perte du reverse stress (%)", min_value=1.0, max_value=90.0, value=20.0, step=1.0)

metadata = read_metadata()
if metadata.get("kind") != "yahoo_finance":
    st.warning(
        "**Mode démonstration hors ligne :** le CSV fourni contient des prix synthétiques, pas des cours de marché observés. "
        "Les résultats illustrent le fonctionnement du moteur et ne doivent pas être interprétés comme des estimations réelles. "
        "Clique sur « Télécharger les cours ajustés Yahoo Finance » pour remplacer le CSV par les données réelles."
    )
else:
    st.success(
        f"Source : {metadata.get('source', 'Yahoo Finance')} · champ : {metadata.get('price_field', 'Cours ajustés')} · "
        f"période : {metadata.get('start_date', '—')} au {metadata.get('end_date', '—')} · "
        f"{len(all_prices):,} lignes".replace(",", " ")
    )
    st.caption("Les estimations reposent sur des historiques ; elles ne prédisent pas les pertes futures.")

if n_selected < 6:
    st.error("Le cahier des charges demande au moins 6 tickers. Sélectionne au moins 6 actifs dans la barre latérale.")
    st.stop()
selected_classes = {ASSET_CLASSES.get(t, "Non renseignée") for t in selected_tickers}
if len(selected_classes) < 3:
    st.error("Le cahier des charges demande au moins 3 classes d'actifs distinctes. Ajoute des obligations et/ou des matières premières.")
    st.stop()
if any(ticker not in all_prices.columns for ticker in selected_tickers):
    st.error("Certains tickers sélectionnés ne sont pas présents dans le CSV. Télécharge à nouveau les cours Yahoo Finance.")
    st.stop()

try:
    weight_percent = pd.to_numeric(edited_alloc["Poids (%)"], errors="coerce").to_numpy(dtype=float)
    shocks_percent = pd.to_numeric(edited_alloc["Choc hypothétique (%)"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weight_percent).all() or not np.isfinite(shocks_percent).all():
        raise ValueError("Les poids et les chocs doivent être renseignés par des nombres valides.")
    weights = weight_percent / 100.0
    if (weights < 0).any():
        raise ValueError("Les poids négatifs ne sont pas pris en charge.")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Les poids totalisent {weights.sum():.2%}. Ajuste la colonne Poids (%) pour atteindre exactement 100 %.")
except ValueError as exc:
    st.error(str(exc))
    st.stop()

prices = all_prices[selected_tickers].copy().ffill().dropna(how="any")
asset_returns = simple_returns(prices)
if len(asset_returns) < 252:
    st.error(f"Historique trop court : seulement {len(asset_returns)} rendements communs. Il faut au moins 252 jours, idéalement 5 ans ou plus.")
    st.stop()

period_start = asset_returns.index.min().strftime("%Y-%m-%d")
period_end = asset_returns.index.max().strftime("%Y-%m-%d")
p_return = portfolio_returns(asset_returns, weights)
port_summary = portfolio_metrics(p_return, risk_free_rate)
asset_summary = asset_metrics(asset_returns, risk_free_rate)

with st.container():
    st.subheader("Synthèse du portefeuille")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement annualisé", pct(port_summary["Rendement annualisé"]))
    c2.metric("Volatilité annualisée", pct(port_summary["Volatilité annualisée"]))
    c3.metric("Sharpe", f"{port_summary['Sharpe']:.2f}")
    c4.metric("Drawdown maximal", pct(port_summary["Drawdown maximal"]))
    st.caption(f"Cours couvrant {period_start} à {period_end} · {len(asset_returns):,} rendements communs · taux sans risque {risk_free_rate:.2%} · seed {int(seed)}".replace(",", " "))

# The tabs separate portfolio analytics, tail risk, stress tests and backtesting.
tab_overview, tab_risk, tab_stress, tab_backtest, tab_data = st.tabs([
    "Vue d'ensemble", "VaR & Expected Shortfall", "Stress tests", "Backtesting", "Données & méthode"
])

with tab_overview:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Valeur normalisée des actifs")
        normalized = prices / prices.iloc[0] * 100.0
        fig_prices = px.line(normalized, x=normalized.index, y=normalized.columns, labels={"value": "Base 100", "variable": "Ticker", "Date": "Date"})
        fig_prices.update_layout(legend_title_text="Actif", height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_prices, use_container_width=True)
    with right:
        st.markdown("#### Statistiques par actif")
        metrics_display = asset_summary.copy()
        for col in ["Rendement annualisé", "Volatilité annualisée", "Drawdown maximal"]:
            metrics_display[col] = metrics_display[col].map(lambda x: pct(x, 2))
        metrics_display["Sharpe"] = metrics_display["Sharpe"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
        st.dataframe(metrics_display, use_container_width=True)
        if not asset_summary.empty:
            riskiest = asset_summary["Volatilité annualisée"].idxmax()
            st.info(f"Actif le plus volatil sur cette période : **{riskiest}** ({pct(asset_summary.loc[riskiest, 'Volatilité annualisée'])} annualisé).")

    st.markdown("#### Corrélations")
    corr_left, corr_right = st.columns(2)
    with corr_left:
        st.caption("Période complète")
        corr_all = asset_returns.corr()
        fig_corr = px.imshow(corr_all, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        fig_corr.update_layout(height=450, margin=dict(l=10, r=10, t=25, b=10), coloraxis_colorbar_title="ρ")
        st.plotly_chart(fig_corr, use_container_width=True)
    with corr_right:
        st.caption("Épisode de crise Covid-19 (19 février–30 avril 2020)")
        crisis_returns = asset_returns.loc["2020-02-19":"2020-04-30"]
        if len(crisis_returns) >= 10:
            crisis_corr = crisis_returns.corr()
            fig_crisis = px.imshow(crisis_corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
            fig_crisis.update_layout(height=450, margin=dict(l=10, r=10, t=25, b=10), coloraxis_colorbar_title="ρ")
            st.plotly_chart(fig_crisis, use_container_width=True)
        else:
            st.info("La période Covid-19 n'est pas disponible dans l'historique sélectionné.")

    with st.expander("Contributions au risque de volatilité"):
        contributions = risk_contributions(asset_returns, weights)
        contributions["Poids (%)"] = contributions["Poids (%)"].map(lambda x: f"{x:.2f}%")
        contributions["Contribution au risque (%)"] = contributions["Contribution au risque (%)"].map(lambda x: f"{x:.2f}%")
        contributions["Contribution vol. quotidienne"] = contributions["Contribution vol. quotidienne"].map(lambda x: f"{x:.4%}")
        st.dataframe(contributions, use_container_width=True, hide_index=True)
        st.caption("Une contribution peut être négative si un actif réduit la volatilité totale grâce à sa covariance avec le portefeuille.")

with tab_risk:
    st.markdown("#### Tableau de risque attendu dans le projet")
    st.caption(f"Valeur du portefeuille : {euro(portfolio_value)} · source : {metadata.get('source', 'CSV local')} · période : {period_start}–{period_end}.")
    with st.spinner("Calcul des quantiles historiques, paramétriques et Monte Carlo…"):
        risk_table = compute_risk_table(
            asset_returns=asset_returns,
            weights=weights,
            portfolio_value=float(portfolio_value),
            horizons=(1, 10),
            confidence_levels=(0.95, 0.99),
            n_simulations=int(n_simulations),
            seed=int(seed),
        )
    st.dataframe(formatted_risk_table(risk_table), use_container_width=True, hide_index=True)
    focus_row = risk_table[(risk_table["Horizon (jours)"] == focus_horizon) & (risk_table["Méthode"] == "Historique")].iloc[0]
    focus_var_col = f"VaR {int(focus_confidence * 100)} % (€)"
    focus_es_col = f"ES {int(focus_confidence * 100)} % (€)"
    a, b, c = st.columns(3)
    a.metric(f"VaR historique {focus_confidence_label} · {focus_horizon} j", euro(float(focus_row[focus_var_col])))
    b.metric(f"ES historique {focus_confidence_label} · {focus_horizon} j", euro(float(focus_row[focus_es_col])))
    c.metric("Simulations Monte Carlo", f"{int(n_simulations):,}".replace(",", " "))
    hist_dist = -p_return.to_numpy() * float(portfolio_value)
    fig_hist = px.histogram(x=hist_dist, nbins=60, labels={"x": "Perte quotidienne (€)", "count": "Nombre de jours"}, title="Distribution historique des pertes (1 jour)")
    hist_var99 = float(risk_table[(risk_table["Horizon (jours)"] == 1) & (risk_table["Méthode"] == "Historique")]["VaR 99 % (€)"].iloc[0])
    hist_es99 = float(risk_table[(risk_table["Horizon (jours)"] == 1) & (risk_table["Méthode"] == "Historique")]["ES 99 % (€)"].iloc[0])
    fig_hist.add_vline(x=hist_var99, line_dash="dash", annotation_text="VaR historique 99 %")
    fig_hist.add_vline(x=hist_es99, line_dash="dot", annotation_text="ES historique 99 %")
    fig_hist.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown(
        "**Interprétation :** la VaR est un seuil de perte associé au niveau de confiance ; ce n'est pas une perte maximale. "
        "L'Expected Shortfall mesure la perte moyenne au-delà du seuil VaR. Les résultats à 10 jours utilisent des rendements composés "
        "pour la méthode historique et une mise à l'échelle en √temps pour la méthode paramétrique."
    )

with tab_stress:
    st.markdown("#### Comparaison des scénarios de stress")
    st.caption("Pour les stress de corrélation et de volatilité, la colonne perte correspond à une VaR 99 % paramétrique stressée à un jour. Les ratios utilisent les VaR/ES historiques 99 % de référence.")

    baseline_var99 = float(risk_table[(risk_table["Horizon (jours)"] == 1) & (risk_table["Méthode"] == "Historique")]["VaR 99 % (€)"].iloc[0])
    baseline_es99 = float(risk_table[(risk_table["Horizon (jours)"] == 1) & (risk_table["Méthode"] == "Historique")]["ES 99 % (€)"].iloc[0])
    stress_rows: list[dict[str, object]] = []
    impact_tables: dict[str, pd.DataFrame] = {}

    event_returns, actual_event_start, actual_event_end = historical_event_returns(prices, selected_tickers, event_name)
    if event_returns is not None:
        event_contributions = weights * event_returns.to_numpy(dtype=float)
        event_port_return = float(event_contributions.sum())
        event_loss = -event_port_return * portfolio_value
        worst_idx = int(np.argmin(event_contributions))
        stress_rows.append({
            "Scénario": event_name,
            "Nature": f"Historique ({actual_event_start}–{actual_event_end})",
            "Perte stressée (€)": event_loss,
            "Perte (% valeur)": -event_port_return * 100,
            "Perte / VaR historique 99 %": ratio(event_loss, baseline_var99),
            "Perte / ES historique 99 %": ratio(event_loss, baseline_es99),
            "Actif le plus pénalisant": selected_tickers[worst_idx],
        })
        impact_tables[event_name] = pd.DataFrame({
            "Ticker": selected_tickers,
            "Poids (%)": weights * 100,
            "Rendement de l'épisode (%)": event_returns.to_numpy(dtype=float) * 100,
            "Contribution portefeuille (%)": event_contributions * 100,
            "Contribution (€)": event_contributions * portfolio_value,
        })
    else:
        st.warning(f"L'épisode « {event_name} » n'est pas couvert par l'historique actuellement chargé. Télécharge un historique qui inclut les dates concernées.")

    custom_shocks = shocks_percent / 100.0
    custom_contribs = weights * custom_shocks
    custom_port_return = float(custom_contribs.sum())
    custom_loss = -custom_port_return * portfolio_value
    custom_worst_idx = int(np.argmin(custom_contribs))
    stress_rows.append({
        "Scénario": "Chocs hypothétiques personnalisés",
        "Nature": "Chocs saisis par actif",
        "Perte stressée (€)": custom_loss,
        "Perte (% valeur)": -custom_port_return * 100,
        "Perte / VaR historique 99 %": ratio(custom_loss, baseline_var99),
        "Perte / ES historique 99 %": ratio(custom_loss, baseline_es99),
        "Actif le plus pénalisant": selected_tickers[custom_worst_idx],
    })
    impact_tables["Chocs hypothétiques personnalisés"] = pd.DataFrame({
        "Ticker": selected_tickers,
        "Poids (%)": weights * 100,
        "Choc saisi (%)": shocks_percent,
        "Contribution portefeuille (%)": custom_contribs * 100,
        "Contribution (€)": custom_contribs * portfolio_value,
    })

    corr_result = correlation_stress(asset_returns, weights, float(portfolio_value), target_correlation=target_correlation, confidence=0.99, horizon=1)
    stress_rows.append({
        "Scénario": f"Stress de corrélation (ρ={target_correlation:.2f})",
        "Nature": "VaR 99 % paramétrique stressée (1 j)",
        "Perte stressée (€)": corr_result["VaR (€)"],
        "Perte (% valeur)": corr_result["VaR (€)"] / portfolio_value * 100,
        "Perte / VaR historique 99 %": ratio(corr_result["VaR (€)"], baseline_var99),
        "Perte / ES historique 99 %": ratio(corr_result["VaR (€)"], baseline_es99),
        "Actif le plus pénalisant": "Corrélations",
    })
    st.caption(f"Après stress de corrélation, volatilité annualisée estimée : {pct(corr_result['Volatilité annualisée'])} ; VaR 99 % : {euro(corr_result['VaR (€)'])} ; ES 99 % : {euro(corr_result['ES (€)'])}.")

    vol_result = volatility_stress(asset_returns, weights, float(portfolio_value), multiplier=volatility_multiplier, confidence=0.99, horizon=1)
    stress_rows.append({
        "Scénario": f"Stress de volatilité (×{volatility_multiplier:.2f})",
        "Nature": "VaR 99 % paramétrique stressée (1 j)",
        "Perte stressée (€)": vol_result["VaR (€)"],
        "Perte (% valeur)": vol_result["VaR (€)"] / portfolio_value * 100,
        "Perte / VaR historique 99 %": ratio(vol_result["VaR (€)"], baseline_var99),
        "Perte / ES historique 99 %": ratio(vol_result["VaR (€)"], baseline_es99),
        "Actif le plus pénalisant": "Volatilité globale",
    })
    st.caption(f"Après stress de volatilité, volatilité annualisée estimée : {pct(vol_result['Volatilité annualisée'])} ; VaR 99 % : {euro(vol_result['VaR (€)'])} ; ES 99 % : {euro(vol_result['ES (€)'])}.")

    try:
        reverse = reverse_stress(selected_tickers, weights, target_loss_fraction=float(reverse_loss_pct) / 100.0)
        reverse_loss = float(reverse_loss_pct) / 100.0 * portfolio_value
        reverse_shocks = reverse["Chocs actifs (%)"]
        reverse_contribs = weights * reverse_shocks.to_numpy(dtype=float) / 100.0
        reverse_worst = selected_tickers[int(np.argmin(reverse_contribs))]
        stress_rows.append({
            "Scénario": f"Reverse stress (perte cible {reverse_loss_pct:.1f} %)",
            "Nature": "Choc marché via bêtas illustratifs",
            "Perte stressée (€)": reverse_loss,
            "Perte (% valeur)": reverse_loss_pct,
            "Perte / VaR historique 99 %": ratio(reverse_loss, baseline_var99),
            "Perte / ES historique 99 %": ratio(reverse_loss, baseline_es99),
            "Actif le plus pénalisant": reverse_worst,
        })
        impact_tables["Reverse stress"] = pd.DataFrame({
            "Ticker": selected_tickers,
            "Poids (%)": weights * 100,
            "Bêta illustratif": [MARKET_BETAS.get(ticker, 1.0) for ticker in selected_tickers],
            "Choc de l'actif (%)": reverse_shocks.reindex(selected_tickers).to_numpy(),
            "Contribution portefeuille (%)": reverse_contribs * 100,
        })
        st.info(f"Reverse stress : un choc de marché d'environ {pct(float(reverse['Choc marché (%)']) / 100.0)} atteint la perte cible, avec un bêta de portefeuille de {float(reverse['Bêta effectif']):.2f}. Les bêtas utilisés sont des approximations illustratives, pas des estimations de régression.")
    except ValueError as exc:
        st.warning(f"Reverse stress non calculé : {exc}")

    stress_df = pd.DataFrame(stress_rows)
    if not stress_df.empty:
        formatted_stress = stress_df.copy()
        formatted_stress["Perte stressée (€)"] = formatted_stress["Perte stressée (€)"].map(euro)
        formatted_stress["Perte (% valeur)"] = formatted_stress["Perte (% valeur)"].map(lambda x: pct(x / 100.0))
        formatted_stress["Perte / VaR historique 99 %"] = formatted_stress["Perte / VaR historique 99 %"].map(lambda x: f"{x:.2f}x" if np.isfinite(x) else "—")
        formatted_stress["Perte / ES historique 99 %"] = formatted_stress["Perte / ES historique 99 %"].map(lambda x: f"{x:.2f}x" if np.isfinite(x) else "—")
        st.dataframe(formatted_stress, use_container_width=True, hide_index=True)
        worst_scenario_idx = int(stress_df["Perte stressée (€)"].astype(float).idxmax())
        st.markdown(f"**Scénario de perte le plus élevé dans ce tableau :** {stress_df.loc[worst_scenario_idx, 'Scénario']} ({euro(float(stress_df.loc[worst_scenario_idx, 'Perte stressée (€)']))}).")
    for name, impact_df in impact_tables.items():
        with st.expander(f"Détail par actif — {name}"):
            shown = impact_df.copy()
            for col in shown.columns:
                if col.endswith("(€)") or col == "Contribution (€)":
                    shown[col] = shown[col].map(euro)
                elif "(%)" in col or col.endswith(" (%)"):
                    shown[col] = shown[col].map(lambda x: pct(float(x) / 100.0))
            st.dataframe(shown, use_container_width=True, hide_index=True)

with tab_backtest:
    st.markdown("#### Backtesting de la VaR historique à 99 %")
    st.write("Chaque jour testé est comparé à un quantile calculé sur les observations précédentes uniquement. Une exception survient si la perte réalisée dépasse la VaR estimée.")
    try:
        bt_df, bt_stats = rolling_historical_backtest(p_return, float(portfolio_value), lookback=int(lookback), confidence=0.99, test_days=250)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Exceptions", f"{bt_stats['Exceptions']} / {bt_stats['Observations']}")
        b2.metric("Exceptions attendues (99 %)", f"{bt_stats['Observations'] * 0.01:.2f}")
        b3.metric("p-value Kupiec", f"{bt_stats['p-value']:.4f}")
        b4.metric("Feu tricolore", str(bt_stats["Feu tricolore"]).split(" (")[0])
        if bt_stats["Couverture non rejetée à 5 %"]:
            st.success("Le test de Kupiec ne rejette pas l'hypothèse de couverture correcte au seuil de 5 %.")
        else:
            st.warning("Le test de Kupiec rejette l'hypothèse de couverture correcte au seuil de 5 %.")
        st.caption(f"Test hors échantillon : {bt_stats['Date de début du test']} à {bt_stats['Date de fin du test']} · fenêtre d'estimation {lookback} jours.")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["VaR historique (€)"], mode="lines", name="VaR historique 99 %"))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Perte réalisée (€)"], mode="lines", name="Perte réalisée"))
        exceptions = bt_df[bt_df["Exception"]]
        fig_bt.add_trace(go.Scatter(x=exceptions.index, y=exceptions["Perte réalisée (€)"], mode="markers", name="Exception", marker={"size": 9, "symbol": "x"}))
        fig_bt.update_layout(height=400, xaxis_title="Date", yaxis_title="Perte (€)", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_bt, use_container_width=True)
        with st.expander("Détail des observations du backtest"):
            st.dataframe(bt_df, use_container_width=True)
        st.caption("Feu tricolore indicatif bâlois pour un test de 250 jours à 99 % : vert 0–4 exceptions, orange 5–9, rouge à partir de 10. Le test de Kupiec évalue la fréquence des exceptions, mais pas leur indépendance ni leur regroupement dans le temps.")
    except ValueError as exc:
        st.error(f"Backtest indisponible : {exc}")

with tab_data:
    st.markdown("#### Source et période des données")
    st.json(metadata)
    st.write(f"Période effectivement utilisée : **{period_start} – {period_end}** ({len(asset_returns):,} rendements journaliers alignés).".replace(",", " "))
    st.download_button(
        label="Télécharger le CSV utilisé",
        data=prices.to_csv(index_label="Date", date_format="%Y-%m-%d").encode("utf-8"),
        file_name="portfolio_prices_selected.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("#### Hypothèses et limites")
    st.markdown(
        "- Rendements du portefeuille calculés comme somme pondérée des rendements journaliers simples, avec poids fixes.\n"
        "- VaR historique : quantile empirique des pertes ; ES : moyenne des pertes au-delà de la VaR.\n"
        "- VaR paramétrique : loi normale, moyenne et écart-type empiriques ; horizon de 10 jours mis à l'échelle en √temps.\n"
        "- Monte Carlo : rendements logarithmiques multivariés normaux estimés sur l'historique ; seed fixe pour la reproductibilité.\n"
        "- Le stress de corrélation préserve les volatilités individuelles et remplace les corrélations hors diagonale par la cible.\n"
        "- Les bêtas du reverse stress sont des proxys explicitement illustratifs. Les résultats dépendent fortement de la fenêtre et des hypothèses."
    )

st.divider()
st.caption("Projet pédagogique de gestion du risque de portefeuille. Les indicateurs historiques et les scénarios simulés ne constituent pas un conseil en investissement.")
