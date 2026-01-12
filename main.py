
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volatility forecasting pipeline (NIFTY50 real data guaranteed via robust yfinance retries)

- Forces real data (no synthetic fallback unless you explicitly disable)
- Retries yfinance using both download() and Ticker.history() with start/end
- Physics-inspired features (spring-damper + jerk/snap) with no lookahead
- Stable Black-Scholes pricer (handles sigma==0 and scalar T broadcasting)
- Walk-forward expanding validation; baselines: persistence, EWMA, optional GARCH
- Metrics include price-level MAE/RMSE vs realized vol; plot saved for headless

Run (CLI): python main.py --ticker ^NSEI --period 10y --force-real --auto-adjust --save-plot
"""

import os
import sys
import math
import warnings
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings("ignore")

# Optional GARCH baseline
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


# ---------- Robust real-data downloader (NIFTY50 / ^NSEI) ----------
def _normalize_price_volume(df: pd.DataFrame, auto_adjust: bool) -> pd.DataFrame:
    """Normalize to columns: Price, Volume."""
    df = df.copy()
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c if x != ""]).strip("_") for c in df.columns]

    # Price selection: if auto_adjust=True, Close is already adjusted
    if auto_adjust:
        if "Close" in df.columns:
            df = df.rename(columns={"Close": "Price"})
        elif "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Price"})
        else:
            raise ValueError("Expected 'Close' or 'Adj Close' for price.")
    else:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Price"})
        elif "Close" in df.columns:
            df = df.rename(columns={"Close": "Price"})
        else:
            raise ValueError("Expected 'Close' or 'Adj Close' for price.")

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    # Clean volume: replace zeros, ffill, then constant
    df["Volume"] = df["Volume"].replace(0, np.nan).fillna(method="ffill").fillna(100000.0)

    return df[["Price", "Volume"]].dropna(how="all")


def _retry_sleep(sec: float):
    try:
        import time
        time.sleep(sec)
    except Exception:
        pass


def get_data_real(ticker: str = "^NSEI",
                  period: str = "10y",
                  auto_adjust: bool = True,
                  max_retries: int = 4) -> pd.DataFrame:
    """
    Robustly fetch REAL data for the given ticker using yfinance.
    Tries yf.download(period=...) -> yf.Ticker.history(start/end) -> yf.download(start/end).
    Raises on failure (no synthetic fallback).
    """
    print(f"[Data] Fetching REAL data for {ticker} (period={period}, auto_adjust={auto_adjust})")

    # First attempt: direct download by period
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker, period=period, interval="1d",
                progress=False, auto_adjust=auto_adjust, threads=True
            )
            df_norm = _normalize_price_volume(df, auto_adjust=auto_adjust)
            if len(df_norm) >= 250:  # require ~1 trading year minimum
                print(f"  -> yfinance.download() succeeded on attempt {attempt}: {len(df_norm)} rows")
                return df_norm
            else:
                print(f"  -> insufficient rows ({len(df_norm)}); retrying...")
        except Exception as e:
            print(f"  -> yfinance.download() attempt {attempt} failed: {e}")
        _retry_sleep(1.5 * attempt)

    # Second attempt: Ticker.history with explicit start/end
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=3650 if period.endswith("10y") else 1825)
    for attempt in range(1, max_retries + 1):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=auto_adjust
            )
            df_norm = _normalize_price_volume(df, auto_adjust=auto_adjust)
            if len(df_norm) >= 250:
                print(f"  -> Ticker.history() succeeded on attempt {attempt}: {len(df_norm)} rows")
                return df_norm
            else:
                print(f"  -> Ticker.history() insufficient rows ({len(df_norm)}); retrying...")
        except Exception as e:
            print(f"  -> Ticker.history() attempt {attempt} failed: {e}")
        _retry_sleep(1.5 * attempt)

    # Third attempt: download with explicit start/end (backup path)
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False, auto_adjust=auto_adjust, threads=True
            )
            df_norm = _normalize_price_volume(df, auto_adjust=auto_adjust)
            if len(df_norm) >= 250:
                print(f"  -> yfinance.download(start/end) succeeded on attempt {attempt}: {len(df_norm)} rows")
                return df_norm
            else:
                print(f"  -> download(start/end) insufficient rows ({len(df_norm)}); retrying...")
        except Exception as e:
            print(f"  -> download(start/end) attempt {attempt} failed: {e}")
        _retry_sleep(1.5 * attempt)

    raise RuntimeError(
        f"Failed to fetch REAL data for {ticker}. "
        "Please ensure internet connectivity, that 'yfinance' is up-to-date, and that the ticker is correct (^NSEI for NIFTY 50). "
        "You can also try --auto-adjust off or a shorter --period."
    )


# ---------- Feature engineering (physics-inspired) ----------
def compute_physics_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_10"] = df["Price"].ewm(span=10, adjust=False).mean()
    df["E_STD_10"] = df["Price"].ewm(span=10, adjust=False).std().fillna(0.0)

    df["Upper"] = df["EMA_10"] + 2 * df["E_STD_10"]
    df["Lower"] = df["EMA_10"] - 2 * df["E_STD_10"]
    df["Bandwidth"] = (df["Upper"] - df["Lower"]) / (df["EMA_10"].replace(0, np.nan))
    df["Bandwidth"] = df["Bandwidth"].fillna(df["Bandwidth"].median()).replace(0, 1e-9)
    df["k_raw"] = 1.0 / (df["Bandwidth"] + 1e-6)
    df["k"] = df["k_raw"] / (df["k_raw"].mean() + 1e-9)

    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["c"] = (df["Volume"] / (df["Vol_MA20"].replace(0, np.nan))).fillna(1.0) * 0.5

    df["Log_Ret"] = np.log(df["Price"] / df["Price"].shift(1)).fillna(0.0)
    forces = df["Log_Ret"].fillna(0).values * 100.0

    n = len(df)
    v = np.zeros(n)
    x = np.zeros(n)
    a = np.zeros(n)
    jerk = np.zeros(n)
    snap = np.zeros(n)

    k_vals = df["k"].fillna(df["k"].median()).values
    c_vals = df["c"].fillna(df["c"].median()).values
    m = 1.0
    dt = 1.0

    for i in range(1, n):
        F_net = forces[i] - c_vals[i] * v[i - 1] - k_vals[i] * x[i - 1]
        a[i] = F_net / m
        jerk[i] = (a[i] - a[i - 1]) / (dt + 1e-12)
        snap[i] = (jerk[i] - jerk[i - 1]) / (dt + 1e-12)
        v[i] = v[i - 1] + a[i] * dt
        x[i] = x[i - 1] + v[i] * dt

    df["Phys_Power"] = np.log1p(np.abs(forces * v))
    df["Signal_Jerk"] = pd.Series(jerk, index=df.index).rolling(3, min_periods=1).mean()
    df["Signal_Snap"] = pd.Series(snap, index=df.index).rolling(3, min_periods=1).mean()

    df[["k", "c", "Phys_Power", "Signal_Jerk", "Signal_Snap"]] = \
        df[["k", "c", "Phys_Power", "Signal_Jerk", "Signal_Snap"]].fillna(method="ffill").fillna(0.0)

    return df


def create_features_targets(df: pd.DataFrame, TRADING_DAYS: int = 21) -> pd.DataFrame:
    df = df.copy()
    df["Past_RV"] = df["Log_Ret"].rolling(window=TRADING_DAYS).std()
    df["Current_Vol"] = df["Past_RV"] * np.sqrt(252.0)

    df["Future_RV"] = df["Log_Ret"].shift(-1).rolling(window=TRADING_DAYS).std()
    df["Target_Vol"] = df["Future_RV"] * np.sqrt(252.0)

    df["Vol_Lag_1"] = df["Current_Vol"].shift(1)
    df["Ret_Lag_1"] = df["Log_Ret"].shift(1)

    df_model = df[[
        "Price", "Volume",
        "k", "Phys_Power", "Signal_Jerk", "Signal_Snap",
        "Current_Vol", "Vol_Lag_1", "Ret_Lag_1",
        "Target_Vol"
    ]].copy()

    return df_model.dropna().copy()


# ---------- Stable Black-Scholes pricer ----------
def bs_pricer(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    eps = 1e-6
    S, K, sigma, T = np.broadcast_arrays(S, K, sigma, T)

    small_sigma_mask = (sigma <= eps) | (T <= eps)

    sig = np.clip(sigma, eps, None)
    t = np.clip(T, eps, None)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    bs_prices = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    intrinsic = np.maximum(S - K * np.exp(-r * T), 0.0)
    return np.where(small_sigma_mask, intrinsic, bs_prices)


# ---------- Baselines ----------
def ewma_volatility(returns: pd.Series, span: int = 21) -> float:
    ret = returns.dropna()
    if len(ret) < 2:
        return float(np.nan)
    lam = 2.0 / (span + 1.0)
    ewma_var = (ret ** 2).ewm(alpha=lam, adjust=False).mean().iloc[-1]
    return float(np.sqrt(ewma_var) * np.sqrt(252.0))


def garch_forecast_variance(returns: pd.Series, horizon: int = 21) -> float:
    if not HAS_ARCH:
        return float(np.nan)
    try:
        am = arch_model(returns.dropna() * 100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        res = am.fit(disp="off", show_warning=False)
        fh = res.forecast(horizon=horizon, reindex=False)
        var_path = fh.variance.iloc[-1].values / (100.0 ** 2)
        var_avg = float(np.mean(var_path))
        return float(np.sqrt(var_avg) * np.sqrt(252.0))
    except Exception:
        return float(np.nan)


# ---------- Walk-forward ----------
def walk_forward_forecast(df_model: pd.DataFrame,
                          features: list,
                          target_name: str = "Target_Vol",
                          model=None,
                          initial_train_frac: float = 0.5,
                          test_window: int = None) -> pd.DataFrame:
    n = len(df_model)
    if test_window is None:
        test_window = max(1, int(0.15 * n))

    initial_train = max(10, int(initial_train_frac * n))
    indices = df_model.index

    preds = pd.Series(index=indices, dtype=float)
    baseline_persist = pd.Series(index=indices, dtype=float)
    baseline_ewma = pd.Series(index=indices, dtype=float)
    baseline_garch = pd.Series(index=indices, dtype=float)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model if model is not None else HistGradientBoostingRegressor(random_state=42))
    ])

    start = initial_train
    step = test_window

    while start < n:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, min(start + step, n))
        if len(test_idx) == 0:
            break

        train_df = df_model.iloc[train_idx]
        test_df = df_model.iloc[test_idx]

        pipe = clone(pipeline)
        if hasattr(pipe.named_steps["model"], "random_state"):
            pipe.named_steps["model"].random_state = 42
        pipe.fit(train_df[features], train_df[target_name])

        preds.iloc[test_idx] = pipe.predict(test_df[features])

        baseline_persist.iloc[test_idx] = test_df["Current_Vol"].values

        try:
            train_returns = train_df["Ret_Lag_1"].dropna()
            ewma_v = ewma_volatility(train_returns, span=21)
        except Exception:
            ewma_v = float(train_df["Current_Vol"].iloc[-1])
        baseline_ewma.iloc[test_idx] = ewma_v

        try:
            garch_v = garch_forecast_variance(train_df["Ret_Lag_1"].fillna(0.0), horizon=21)
        except Exception:
            garch_v = float(np.nan)
        baseline_garch.iloc[test_idx] = garch_v

        start += step

    results = df_model.copy()
    results["Predicted_Vol"] = preds
    results["Base_Persist_Vol"] = baseline_persist
    results["Base_EWMA_Vol"] = baseline_ewma
    results["Base_GARCH_Vol"] = baseline_garch

    return results.dropna(subset=["Predicted_Vol", "Base_Persist_Vol"])


# ---------- Metrics ----------
def evaluate_price_errors(test_df: pd.DataFrame, T_days: int = 30, r: float = 0.07) -> Dict[str, float]:
    T = T_days / 365.0
    S = test_df["Price"].values
    K = S
    target_sigma = test_df["Target_Vol"].values
    pred_sigma = test_df["Predicted_Vol"].values
    base_persist = test_df["Base_Persist_Vol"].values
    base_ewma = test_df.get("Base_EWMA_Vol", pd.Series(np.nan, index=test_df.index)).values
    base_garch = test_df.get("Base_GARCH_Vol", pd.Series(np.nan, index=test_df.index)).values

    price_ideal = bs_pricer(S, K, T, r, target_sigma)
    price_ai = bs_pricer(S, K, T, r, pred_sigma)
    price_pers = bs_pricer(S, K, T, r, base_persist)
    price_ewma = bs_pricer(S, K, T, r, base_ewma)
    price_garch = bs_pricer(S, K, T, r, base_garch)

    def mae(a, b): return float(np.nanmean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.nanmean((a - b) ** 2)))

    metrics = {
        "MAE_Persistence": mae(price_pers, price_ideal),
        "RMSE_Persistence": rmse(price_pers, price_ideal),
        "MAE_EWMA": mae(price_ewma, price_ideal),
        "RMSE_EWMA": rmse(price_ewma, price_ideal),
        "MAE_GARCH": mae(price_garch, price_ideal),
        "RMSE_GARCH": rmse(price_garch, price_ideal),
        "MAE_Model": mae(price_ai, price_ideal),
        "RMSE_Model": rmse(price_ai, price_ideal),
    }
    base_mae = metrics["MAE_Persistence"]
    metrics["MAE_Improvement_pct_Model_vs_Persistence"] = ((base_mae - metrics["MAE_Model"]) / base_mae * 100.0) if base_mae != 0 else float("nan")
    metrics["MAE_Improvement_pct_EWMA_vs_Persistence"] = ((base_mae - metrics["MAE_EWMA"]) / base_mae * 100.0) if base_mae != 0 else float("nan")
    metrics["MAE_Improvement_pct_GARCH_vs_Persistence"] = ((base_mae - metrics["MAE_GARCH"]) / base_mae * 100.0) if base_mae != 0 else float("nan")
    return metrics


def print_metrics(df_results: pd.DataFrame):
    y_true = df_results["Target_Vol"].values
    y_pred = df_results["Predicted_Vol"].values
    y_pers = df_results["Base_Persist_Vol"].values
    y_ewma = df_results["Base_EWMA_Vol"].values
    y_garch = df_results["Base_GARCH_Vol"].values

    def mae(a, b): return float(np.nanmean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.nanmean((a - b) ** 2)))

    print("Forecasting metrics (volatility):")
    print(f" - MAE (Model): {mae(y_true, y_pred):.6f}")
    print(f" - MAE (Persistence): {mae(y_true, y_pers):.6f}")
    print(f" - MAE (EWMA): {mae(y_true, y_ewma):.6f}")
    print(f" - MAE (GARCH): {mae(y_true, y_garch):.6f}")
    print(f" - RMSE (Model): {rmse(y_true, y_pred):.6f}")
    print(f" - RMSE (Persistence): {rmse(y_true, y_pers):.6f}")
    print(f" - RMSE (EWMA): {rmse(y_true, y_ewma):.6f}")
    print(f" - RMSE (GARCH): {rmse(y_true, y_garch):.6f}")


# ---------- Plot ----------
def plot_results(df_results: pd.DataFrame, title_suffix=""):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df_results.index, df_results["Price"], label="Price", color="white", alpha=0.6)
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left")

    axes[1].plot(df_results.index, df_results["Signal_Snap"], color="orange", label="Signal_Snap")
    axes[1].set_ylabel("Snap")
    axes[1].legend(loc="upper left")

    axes[2].plot(df_results.index, df_results["Target_Vol"], color="green", label="Target_Vol")
    axes[2].plot(df_results.index, df_results["Predicted_Vol"], color="cyan", linestyle="--", label="Predicted_Vol")
    axes[2].plot(df_results.index, df_results["Base_Persist_Vol"], color="gray", linestyle="-.", label="Persistence")
    axes[2].plot(df_results.index, df_results["Base_EWMA_Vol"], color="magenta", linestyle=":", label="EWMA")
    axes[2].plot(df_results.index, df_results["Base_GARCH_Vol"], color="yellow", linestyle=":", label="GARCH")
    axes[2].set_ylabel("Vol (ann.)")
    axes[2].legend(loc="upper left")

    fig.suptitle(f"Vol Forecasting Diagnostics {title_suffix}")
    plt.tight_layout()
    out_name = f"vol_forecast_diag{title_suffix.replace(' ', '_')}.png" if title_suffix else "vol_forecast_diag.png"
    fig.savefig(out_name, dpi=150)
    print(f"Saved plot to: {out_name}")
    try:
        plt.show()
    except Exception:
        pass


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="NIFTY50 volatility forecasting pipeline (real data)")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Ticker symbol (NIFTY50 is ^NSEI on Yahoo Finance)")
    parser.add_argument("--period", type=str, default="10y", help="Download period (e.g., 5y, 10y, max)")
    parser.add_argument("--force-real", action="store_true", help="Force real download (no synthetic fallback)")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to file")
    parser.add_argument("--t-days", type=int, default=30, help="Option maturity in days for price-level eval")
    parser.add_argument("--rate", type=float, default=0.07, help="Risk-free rate for BS pricer")
    parser.add_argument("--trading-days", type=int, default=21, help="Rolling window for realized vol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--auto-adjust", action="store_true", help="Use yfinance auto_adjust=True (recommended)")
    # Notebook-safe:
    args, _unknown = parser.parse_known_args()
    # Default to force real + auto_adjust unless explicitly disabled
    if not args.force_real:
        args.force_real = True
    if not args.auto_adjust:
        args.auto_adjust = True
    return args


def main():
    args = parse_args()

    # Reproducibility
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(args.seed)

    # === REAL DATA DOWNLOAD ===
    df_raw = get_data_real(
        ticker=args.ticker,
        period=args.period,
        auto_adjust=args.auto_adjust,
        max_retries=4
    )
    print(f"[Data] Retrieved {len(df_raw)} rows of REAL data for {args.ticker}")

    # Compute features
    df_phys = compute_physics_signals(df_raw)
    df_model = create_features_targets(df_phys, TRADING_DAYS=args.trading_days)
    print(f"[Modeling] Prepared dataset with {len(df_model)} rows.")

    # Ensure enough data
    if len(df_model) < 50:
        print("[Modeling] Not enough data to run walk-forward evaluation. Exiting.")
        return

    # Features and model
    features = ['k', 'Phys_Power', 'Signal_Jerk', 'Signal_Snap', 'Current_Vol', 'Vol_Lag_1', 'Ret_Lag_1']
    base_model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=500, max_depth=6,
        loss='absolute_error', random_state=args.seed
    )

    # Walk-forward
    test_window = max(10, int(0.1 * len(df_model)))
    df_results = walk_forward_forecast(
        df_model, features, target_name="Target_Vol",
        model=base_model, initial_train_frac=0.6, test_window=test_window
    )
    print(f"[Validation] Walk-forward produced {len(df_results)} evaluated rows.")

    # Metrics
    print_metrics(df_results)
    stats_price = evaluate_price_errors(df_results, T_days=args.t_days, r=args.rate)
    print("\nPrice-level evaluation vs realized future vol (Target):")
    for k, v in stats_price.items():
        print(f" - {k}: {v:.4f}")

    # Plot
    title_suffix = f" ({args.ticker})"
    plot_results(df_results, title_suffix=title_suffix if args.save_plot else "")

    # Feature importances
    final_train_end = int(0.85 * len(df_model))
    final_train = df_model.iloc[:final_train_end]
    final_test = df_model.iloc[final_train_end:]
    pipe_full = Pipeline([("scaler", StandardScaler()), ("model", clone(base_model))])
    pipe_full.fit(final_train[features], final_train["Target_Vol"])
    try:
        importances = pipe_full.named_steps["model"].feature_importances_
        print("\nFeature importances (trained on 85% of data):")
        for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            print(f" - {f}: {imp:.4f}")
    except Exception:
        print("Could not extract feature importances from model.")

    # Final held-out price evaluation
    if len(final_test) > 0:
        final_test_preds = pipe_full.predict(final_test[features])
        final_test = final_test.copy()
        final_test["Predicted_Vol"] = final_test_preds
        final_test["Base_Persist_Vol"] = final_test["Current_Vol"]
        try:
            final_ewma = ewma_volatility(final_train["Ret_Lag_1"].dropna(), span=21)
        except Exception:
            final_ewma = float(final_train["Current_Vol"].iloc[-1])
        final_test["Base_EWMA_Vol"] = final_ewma
        try:
            final_garch = garch_forecast_variance(final_train["Ret_Lag_1"].fillna(0.0), horizon=21)
        except Exception:
            final_garch = float(np.nan)
        final_test["Base_GARCH_Vol"] = final_garch

        stats_full = evaluate_price_errors(final_test, T_days=args.t_days, r=args.rate)
        print("\nFinal held-out test price MAE/RMSE:")
        for k, v in stats_full.items():
            print(f" - {k}: {v:.4f}")
    else:
        print("No held-out final test slice available to evaluate.")

    print("\nDone.")


if __name__ == "__main__":
    main()

