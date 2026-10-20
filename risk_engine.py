"""Reusable calculations for the Portfolio Risk Engine project.

Risk conventions
----------------
* Prices are adjusted close prices from Yahoo Finance when downloaded live.
* Daily portfolio returns are computed with fixed user-supplied weights.
* VaR and Expected Shortfall (ES) are reported as positive losses where the
  loss distribution is positive in the left tail; a negative result is retained
  if the data imply that the selected quantile is a gain.
* The parametric model assumes normally distributed daily portfolio returns.
* Monte Carlo uses a multivariate normal fit to historical asset log returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

TRADING_DAYS = 252
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TLT", "IEF", "GLD"]
ASSET_CLASSES: dict[str, str] = {
    "AAPL": "Actions",
    "MSFT": "Actions",
    "NVDA": "Actions",
    "AMZN": "Actions",
    "SPY": "ETF actions",
    "TLT": "Obligations US",
    "IEF": "Obligations US",
    "BND": "Obligations US",
    "GLD": "Or",
    "SLV": "Argent",
    "VNQ": "Immobilier coté",
}
MARKET_BETAS: dict[str, float] = {
    "AAPL": 1.20,
    "MSFT": 1.10,
    "NVDA": 1.70,
    "AMZN": 1.25,
    "SPY": 1.00,
    "TLT": -0.10,
    "IEF": -0.08,
    "BND": 0.00,
    "GLD": 0.10,
    "SLV": 0.65,
    "VNQ": 1.10,
}
HISTORICAL_EVENTS: dict[str, tuple[str, str]] = {
    "Crise financière 2008 (sept. 2008–mars 2009)": ("2008-09-01", "2009-03-31"),
    "Covid-19 (19 fév.–23 mars 2020)": ("2020-02-19", "2020-03-23"),
    "Choc inflation/taux (janv.–oct. 2022)": ("2022-01-03", "2022-10-12"),
}


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a wide price CSV with a Date column and ticker columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de cours introuvable : {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Le fichier de cours est vide : {path}")
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=[date_col]).set_index(date_col)
    frame.index.name = "Date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.ffill().dropna(how="all")
    return frame


def download_yahoo_prices(
    tickers: Sequence[str],
    start: str | pd.Timestamp = "2006-01-01",
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance.

    `auto_adjust=True` makes the Close field adjusted for dividends and splits.
    The requested end date is inclusive from the user's perspective; Yahoo's
    API uses an exclusive end date, so one day is added before download.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on user's environment
        raise ImportError("yfinance n'est pas installé. Lancez `pip install -r requirements.txt`.") from exc

    clean_tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not clean_tickers:
        raise ValueError("Sélectionnez au moins un ticker.")
    start_ts = pd.Timestamp(start).normalize()
    end_ts = (pd.Timestamp(end).normalize() + pd.Timedelta(days=1)) if end is not None else (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    if start_ts >= end_ts:
        raise ValueError("La date de début doit être antérieure à la date de fin.")

    raw = yf.download(
        tickers=clean_tickers,
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance n'a renvoyé aucune donnée. Vérifiez la connexion et les tickers.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Close" in level1:
            prices = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("La réponse Yahoo Finance ne contient pas la colonne Close ajustée attendue.")
    else:
        if "Close" not in raw.columns:
            raise ValueError("La réponse Yahoo Finance ne contient pas de cours de clôture.")
        prices = raw[["Close"]].copy()
        prices.columns = [clean_tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=clean_tickers[0])
    prices.columns = [str(c).strip().upper() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
    prices.index.name = "Date"
    missing = [ticker for ticker in clean_tickers if ticker not in prices.columns or prices[ticker].dropna().empty]
    if missing:
        raise ValueError("Aucun historique exploitable pour : " + ", ".join(missing))
    prices = prices.loc[:, clean_tickers].sort_index().ffill()
    # Do not backfill pre-listing data. Start on the first date available for all
    # selected assets so cross-asset returns use an aligned trading calendar.
    prices = prices.dropna(subset=clean_tickers, how="any")
    if prices.empty:
        raise ValueError("Aucune date commune exploitable pour les tickers sélectionnés.")
    return prices.astype(float)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns without forward-filling prices again."""
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns."""
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def validate_weights(weights: Sequence[float], n_assets: int | None = None, tolerance: float = 1e-6) -> np.ndarray:
    """Validate non-negative weights summing to one and return a NumPy array."""
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Les poids doivent former une liste non vide.")
    if n_assets is not None and arr.size != n_assets:
        raise ValueError("Le nombre de poids doit correspondre au nombre d'actifs.")
    if not np.isfinite(arr).all():
        raise ValueError("Les poids doivent être des nombres finis.")
    if (arr < 0).any():
        raise ValueError("Les poids négatifs ne sont pas pris en charge dans cette version.")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=tolerance):
        raise ValueError(f"La somme des poids doit être égale à 100 % (actuellement {total:.2%}).")
    return arr


def portfolio_returns(asset_returns: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    """Daily portfolio simple returns for fixed weights."""
    weight_array = validate_weights(weights, n_assets=asset_returns.shape[1])
    result = asset_returns.to_numpy(dtype=float) @ weight_array
    return pd.Series(result, index=asset_returns.index, name="Portfolio")


def _loss_var_es(losses: np.ndarray | pd.Series, confidence: float) -> tuple[float, float]:
    arr = np.asarray(losses, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    var = float(np.quantile(arr, confidence, method="linear"))
    tail = arr[arr >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def historical_var_es(
    daily_portfolio_returns: pd.Series,
    portfolio_value: float,
    horizon: int = 1,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Historical VaR/ES from overlapping compounded portfolio returns."""
    if horizon < 1:
        raise ValueError("L'horizon doit être un entier positif.")
    if portfolio_value <= 0:
        raise ValueError("La valeur du portefeuille doit être strictement positive.")
    if len(daily_portfolio_returns) < horizon:
        raise ValueError("Historique insuffisant pour cet horizon.")
    rolling_returns = (1.0 + daily_portfolio_returns).rolling(horizon).apply(np.prod, raw=True) - 1.0
    losses = (-rolling_returns.dropna().to_numpy()) * portfolio_value
    return _loss_var_es(losses, confidence)


def parametric_var_es(
    daily_portfolio_returns: pd.Series,
    portfolio_value: float,
    horizon: int = 1,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Normal parametric VaR/ES using daily simple-return mean and sample std."""
    if horizon < 1 or portfolio_value <= 0:
        raise ValueError("Horizon et valeur de portefeuille invalides.")
    arr = daily_portfolio_returns.dropna().to_numpy(dtype=float)
    if arr.size < 2:
        raise ValueError("Historique insuffisant pour estimer mu et sigma.")
    mu_daily = float(np.mean(arr))
    sigma_daily = float(np.std(arr, ddof=1))
    z = float(norm.ppf(confidence))
    sigma_h = sigma_daily * np.sqrt(horizon)
    mu_h = mu_daily * horizon
    var = (z * sigma_h - mu_h) * portfolio_value
    es = (-mu_h + sigma_h * norm.pdf(z) / (1.0 - confidence)) * portfolio_value
    return float(var), float(es)


def monte_carlo_loss_distribution(
    asset_returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    horizon: int = 1,
    n_simulations: int = 20_000,
    seed: int = 42,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Generate a reproducible Monte Carlo sample of portfolio losses."""
    if horizon < 1 or portfolio_value <= 0:
        raise ValueError("Horizon et valeur de portefeuille invalides.")
    if n_simulations < 1:
        raise ValueError("Le nombre de simulations doit être positif.")
    weight_array = validate_weights(weights, n_assets=asset_returns.shape[1])
    valid_returns = asset_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    lr = np.log1p(valid_returns.to_numpy(dtype=float))
    if lr.shape[0] < 2:
        raise ValueError("Historique insuffisant pour la simulation Monte Carlo.")
    mean_vec = lr.mean(axis=0)
    cov = np.cov(lr, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    cov = (cov + cov.T) / 2.0
    # Small numerical negative eigenvalues can occur in a nearly singular sample covariance.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    cov = (eigenvectors * np.clip(eigenvalues, 1e-12, None)) @ eigenvectors.T
    rng = np.random.default_rng(seed)
    all_losses: list[np.ndarray] = []
    remaining = int(n_simulations)
    while remaining > 0:
        batch = min(batch_size, remaining)
        simulated_log_returns = rng.multivariate_normal(mean_vec, cov, size=(batch, horizon), check_valid="warn")
        simulated_asset_returns = np.expm1(simulated_log_returns)
        daily_portfolio_returns = np.einsum("bhi,i->bh", simulated_asset_returns, weight_array)
        holding_returns = np.prod(1.0 + daily_portfolio_returns, axis=1) - 1.0
        all_losses.append(-holding_returns * portfolio_value)
        remaining -= batch
    return np.concatenate(all_losses)


def monte_carlo_var_es(
    asset_returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    horizon: int = 1,
    confidence: float = 0.99,
    n_simulations: int = 20_000,
    seed: int = 42,
    batch_size: int = 10_000,
) -> tuple[float, float]:
    """Monte Carlo VaR/ES from multivariate-normal daily asset log returns."""
    losses = monte_carlo_loss_distribution(
        asset_returns, weights, portfolio_value, horizon, n_simulations, seed, batch_size
    )
    return _loss_var_es(losses, confidence)


def compute_risk_table(
    asset_returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    horizons: Sequence[int] = (1, 10),
    confidence_levels: Sequence[float] = (0.95, 0.99),
    n_simulations: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return comparable Historical, Parametric and Monte Carlo VaR/ES rows.

    Each empirical or simulated loss sample is computed once per horizon, then
    queried at both confidence levels. This keeps interactive recalculations
    responsive without sacrificing identical Monte Carlo samples across levels.
    """
    weights_array = validate_weights(weights, n_assets=asset_returns.shape[1])
    p_returns = portfolio_returns(asset_returns, weights_array)
    rows: list[dict[str, float | int | str]] = []
    for horizon in horizons:
        rolling_returns = (1.0 + p_returns).rolling(int(horizon)).apply(np.prod, raw=True) - 1.0
        historical_losses = -rolling_returns.dropna().to_numpy() * portfolio_value
        p_arr = p_returns.dropna().to_numpy(dtype=float)
        mu_daily = float(np.mean(p_arr))
        sigma_daily = float(np.std(p_arr, ddof=1)) if len(p_arr) > 1 else 0.0
        mc_losses = monte_carlo_loss_distribution(
            asset_returns, weights_array, portfolio_value, int(horizon), n_simulations, seed + 100 * int(horizon)
        )
        method_losses: dict[str, np.ndarray | None] = {
            "Historique": historical_losses,
            "Paramétrique (normal)": None,
            "Monte Carlo": mc_losses,
        }
        method_rows: dict[str, dict[float, tuple[float, float]]] = {}
        z_alpha: dict[float, tuple[float, float]] = {}
        for alpha in confidence_levels:
            z = float(norm.ppf(float(alpha)))
            sigma_h = sigma_daily * np.sqrt(int(horizon))
            mu_h = mu_daily * int(horizon)
            param_var = (z * sigma_h - mu_h) * portfolio_value
            param_es = (-mu_h + sigma_h * norm.pdf(z) / (1.0 - float(alpha))) * portfolio_value
            z_alpha[float(alpha)] = (float(param_var), float(param_es))
        for method, losses in method_losses.items():
            method_rows[method] = {}
            for alpha in confidence_levels:
                if method == "Paramétrique (normal)":
                    method_rows[method][float(alpha)] = z_alpha[float(alpha)]
                else:
                    method_rows[method][float(alpha)] = _loss_var_es(losses, float(alpha))
        for method, estimates in method_rows.items():
            row: dict[str, float | int | str] = {"Méthode": method, "Horizon (jours)": int(horizon)}
            for alpha in confidence_levels:
                label = int(round(float(alpha) * 100))
                row[f"VaR {label} % (€)"] = estimates[float(alpha)][0]
                row[f"ES {label} % (€)"] = estimates[float(alpha)][1]
            rows.append(row)
    return pd.DataFrame(rows)

def asset_metrics(asset_returns: pd.DataFrame, risk_free_rate: float = 0.03) -> pd.DataFrame:
    """Annualized return/volatility, Sharpe and maximum drawdown per asset."""
    rows = []
    for ticker in asset_returns.columns:
        returns = asset_returns[ticker].dropna()
        if returns.empty:
            continue
        log_return = np.log1p(returns.clip(lower=-0.999999))
        annual_return = float(log_return.mean() * TRADING_DAYS)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else np.nan
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        rows.append({
            "Ticker": ticker,
            "Classe d'actifs": ASSET_CLASSES.get(ticker, "Non renseignée"),
            "Rendement annualisé": annual_return,
            "Volatilité annualisée": annual_vol,
            "Sharpe": float(sharpe),
            "Drawdown maximal": float(drawdown.min()),
        })
    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def portfolio_metrics(daily_portfolio_returns: pd.Series, risk_free_rate: float = 0.03) -> dict[str, float]:
    """Calculate return, annualized volatility, Sharpe and max drawdown."""
    returns = daily_portfolio_returns.dropna()
    if len(returns) < 2:
        raise ValueError("Au moins deux rendements sont nécessaires.")
    ann_return = float(np.log1p(returns.clip(lower=-0.999999)).mean() * TRADING_DAYS)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    wealth = (1.0 + returns).cumprod()
    max_drawdown = float((wealth / wealth.cummax() - 1.0).min())
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    return {
        "Rendement annualisé": ann_return,
        "Volatilité annualisée": ann_vol,
        "Sharpe": float(sharpe),
        "Drawdown maximal": max_drawdown,
    }


def risk_contributions(asset_returns: pd.DataFrame, weights: Sequence[float]) -> pd.DataFrame:
    """Euler contributions to portfolio volatility from the sample covariance."""
    w = validate_weights(weights, n_assets=asset_returns.shape[1])
    cov = asset_returns.cov().to_numpy(dtype=float)
    portfolio_variance = float(w @ cov @ w)
    portfolio_vol = np.sqrt(max(portfolio_variance, 0.0))
    if portfolio_vol == 0:
        contributions = np.zeros_like(w)
    else:
        contributions = w * (cov @ w) / portfolio_vol
    relative = contributions / portfolio_vol * 100.0 if portfolio_vol > 0 else np.zeros_like(w)
    return pd.DataFrame({
        "Ticker": asset_returns.columns,
        "Poids (%)": w * 100,
        "Contribution au risque (%)": relative,
        "Contribution vol. quotidienne": contributions,
    }).sort_values("Contribution au risque (%)", ascending=False, ignore_index=True)


def historical_event_returns(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    event_name: str,
) -> tuple[pd.Series | None, str | None, str | None]:
    """Return buy-and-hold asset returns for a named historical stress window."""
    if event_name not in HISTORICAL_EVENTS:
        raise ValueError(f"Épisode historique inconnu : {event_name}")
    start, end = HISTORICAL_EVENTS[event_name]
    available = [t for t in tickers if t in prices.columns]
    window = prices.loc[start:end, available].dropna(how="all")
    if len(window) < 2:
        return None, None, None
    # Require at least 80% of the window's days to contain usable data for each asset.
    coverage = window.notna().mean()
    if (coverage < 0.8).any():
        return None, None, None
    first = window.ffill().dropna(how="any").iloc[0]
    last = window.ffill().dropna(how="any").iloc[-1]
    return (last / first - 1.0), pd.Timestamp(window.index[0]).strftime("%Y-%m-%d"), pd.Timestamp(window.index[-1]).strftime("%Y-%m-%d")


def stressed_gaussian_metrics(
    daily_covariance: pd.DataFrame | np.ndarray,
    weights: Sequence[float],
    portfolio_value: float,
    mean_daily_return: float,
    confidence: float = 0.99,
    horizon: int = 1,
) -> dict[str, float]:
    """Parametric VaR/ES and annual volatility for a supplied daily covariance."""
    cov = np.asarray(daily_covariance, dtype=float)
    w = validate_weights(weights, n_assets=cov.shape[0])
    if cov.shape != (len(w), len(w)):
        raise ValueError("La matrice de covariance doit être carrée et correspondre aux poids.")
    variance = float(w @ cov @ w)
    sigma_daily = float(np.sqrt(max(variance, 0.0)))
    z = float(norm.ppf(confidence))
    sigma_h = sigma_daily * np.sqrt(horizon)
    mean_h = mean_daily_return * horizon
    var = (z * sigma_h - mean_h) * portfolio_value
    es = (-mean_h + sigma_h * norm.pdf(z) / (1.0 - confidence)) * portfolio_value
    return {
        "Volatilité quotidienne": sigma_daily,
        "Volatilité annualisée": sigma_daily * np.sqrt(TRADING_DAYS),
        "VaR (€)": float(var),
        "ES (€)": float(es),
    }


def correlation_stress(
    daily_returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    target_correlation: float = 0.9,
    confidence: float = 0.99,
    horizon: int = 1,
) -> dict[str, float]:
    """Stress all off-diagonal correlations to an equicorrelation target."""
    if not (0.0 <= target_correlation <= 1.0):
        raise ValueError("La corrélation cible doit être comprise entre 0 et 1.")
    cov = daily_returns.cov().to_numpy(dtype=float)
    sigmas = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = np.full((len(sigmas), len(sigmas)), target_correlation, dtype=float)
    np.fill_diagonal(corr, 1.0)
    stressed_cov = np.outer(sigmas, sigmas) * corr
    p_returns = portfolio_returns(daily_returns, weights)
    return stressed_gaussian_metrics(stressed_cov, weights, portfolio_value, float(p_returns.mean()), confidence, horizon)


def volatility_stress(
    daily_returns: pd.DataFrame,
    weights: Sequence[float],
    portfolio_value: float,
    multiplier: float = 2.0,
    confidence: float = 0.99,
    horizon: int = 1,
) -> dict[str, float]:
    """Stress all asset volatilities by multiplying the covariance by k squared."""
    if not (1.0 <= multiplier <= 3.0):
        raise ValueError("Le multiplicateur de volatilité doit être compris entre 1 et 3.")
    cov = daily_returns.cov().to_numpy(dtype=float) * (multiplier**2)
    p_returns = portfolio_returns(daily_returns, weights)
    return stressed_gaussian_metrics(cov, weights, portfolio_value, float(p_returns.mean()), confidence, horizon)


def reverse_stress(
    tickers: Sequence[str],
    weights: Sequence[float],
    target_loss_fraction: float = 0.20,
    betas: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Find a broad-market shock that reaches a selected portfolio loss limit.

    Asset-level shocks are approximated as beta_i times the market shock. Betas
    are deliberately simple illustrative proxies, not live regression estimates.
    """
    if not (0.0 < target_loss_fraction < 1.0):
        raise ValueError("Le seuil de perte doit être entre 0 % et 100 %.")
    w = validate_weights(weights, n_assets=len(tickers))
    beta_map = MARKET_BETAS if betas is None else betas
    beta_vec = np.array([float(beta_map.get(ticker, 1.0)) for ticker in tickers])
    effective_beta = float(w @ beta_vec)
    if abs(effective_beta) < 1e-8:
        raise ValueError("Bêta de portefeuille proche de zéro : le choc de marché inverse n'est pas identifiable.")
    market_shock = -target_loss_fraction / effective_beta
    asset_shocks = beta_vec * market_shock
    portfolio_return = float(w @ asset_shocks)
    return {
        "Bêta effectif": effective_beta,
        "Choc marché (%)": market_shock * 100,
        "Chocs actifs (%)": pd.Series(asset_shocks * 100, index=tickers, name="Choc (%)"),
        "Rendement portefeuille": portfolio_return,
        "Perte cible (%)": target_loss_fraction * 100,
    }


def kupiec_test(exceptions: int, observations: int, confidence: float = 0.99) -> dict[str, float | int | bool]:
    """Kupiec unconditional-coverage likelihood-ratio test (one degree freedom)."""
    x = int(exceptions)
    n = int(observations)
    if n <= 0 or x < 0 or x > n:
        raise ValueError("Nombre d'exceptions ou d'observations invalide.")
    p = 1.0 - confidence
    p_hat = x / n
    ll_null = (x * np.log(p) if x else 0.0) + ((n - x) * np.log(1.0 - p) if n > x else 0.0)
    ll_alt = (x * np.log(p_hat) if x else 0.0) + ((n - x) * np.log(1.0 - p_hat) if n > x else 0.0)
    lr = max(0.0, float(-2.0 * (ll_null - ll_alt)))
    p_value = float(chi2.sf(lr, df=1))
    return {
        "Exceptions": x,
        "Observations": n,
        "Fréquence observée": p_hat,
        "Fréquence attendue": p,
        "LR Kupiec": lr,
        "p-value": p_value,
        "Couverture non rejetée à 5 %": bool(p_value >= 0.05),
    }


def traffic_light_99(exceptions: int, observations: int) -> str:
    """Basel-style 99% VaR traffic light for a 250-observation test sample."""
    if observations != 250:
        return "Non applicable (échantillon différent de 250 jours)"
    if exceptions <= 4:
        return "Vert (0–4 exceptions)"
    if exceptions <= 9:
        return "Orange (5–9 exceptions)"
    return "Rouge (10 exceptions ou plus)"


def rolling_historical_backtest(
    daily_portfolio_returns: pd.Series,
    portfolio_value: float,
    lookback: int = 500,
    confidence: float = 0.99,
    test_days: int = 250,
) -> tuple[pd.DataFrame, dict[str, float | int | bool | str]]:
    """Out-of-sample rolling Historical VaR backtest on the last test_days."""
    series = daily_portfolio_returns.dropna().astype(float)
    if lookback < 50:
        raise ValueError("La fenêtre de backtest doit contenir au moins 50 jours.")
    if len(series) <= lookback:
        raise ValueError(f"Historique insuffisant : {len(series)} observations pour une fenêtre de {lookback} jours.")
    start_idx = max(lookback, len(series) - test_days)
    records: list[dict[str, object]] = []
    for idx in range(start_idx, len(series)):
        estimation_returns = series.iloc[idx - lookback:idx]
        var_loss = float(np.quantile(-estimation_returns.to_numpy() * portfolio_value, confidence, method="linear"))
        realized_loss = float(-series.iloc[idx] * portfolio_value)
        records.append({
            "Date": series.index[idx],
            "VaR historique (€)": var_loss,
            "Perte réalisée (€)": realized_loss,
            "Exception": bool(realized_loss > var_loss),
        })
    backtest = pd.DataFrame(records).set_index("Date")
    stats = kupiec_test(int(backtest["Exception"].sum()), len(backtest), confidence)
    stats["Feu tricolore"] = traffic_light_99(int(backtest["Exception"].sum()), len(backtest)) if confidence >= 0.99 else "Non calculé pour une confiance autre que 99 %"
    stats["Fenêtre d'estimation (jours)"] = int(lookback)
    stats["Date de début du test"] = pd.Timestamp(backtest.index.min()).strftime("%Y-%m-%d")
    stats["Date de fin du test"] = pd.Timestamp(backtest.index.max()).strftime("%Y-%m-%d")
    return backtest, stats
