
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust, reproducible volatility forecasting pipeline with:
- Safe data ingestion (optionally fail instead of silently using synthetic)
- Physics-inspired features (spring-damper + jerk/snap) (no lookahead)
- Stable Black-Scholes pricer that guards against sigma==0 and scalar T broadcasting
- Walk-forward (rolling / expanding) time-series validation
- Baselines: persistence (Current_Vol), EWMA; optional GARCH (if arch installed)
- Feature scaling pipeline and reproducible training
- Metrics, simple price-based evaluation vs realized (Target) vol

Run: python main.py
Requirements: numpy, pandas, yfinance, matplotlib, scikit-learn, scipy
Optional: arch (for GARCH baseline)

Notebook-friendly: uses argparse.parse_known_args() to ignore Jupyter's -f argument.
"""

import os
import sys
import math
import warnings
import argparse
from typing import Tuple, Dict

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
RNG = np.random.default_rng(42)  # reproducible synthetic data

# Try optional arch package for a GARCH baseline
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


def get_data(ticker: str = "^NSEI", period: str = "10y", force_real: bool = False, auto_adjust: bool = False) -> Tuple[pd.DataFrame, bool]:
    """
    Download price data, normalize column names, and return DataFrame with Price & Volume.
    If download fails and force_real is False, return synthetic demo data and flag True for synthetic.
    If force_real is True, raise the exception instead of returning synthetic data.
    """
    print(f"Downloading {ticker} for period={period} (auto_adjust={auto_adjust}) ...")
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=auto_adjust)

        # Flatten MultiIndex (robust)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(x) for x in c if x != ""]).strip("_") for c in df.columns]

        # Normalize names to 'Price' and 'Volume'
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Price"})
        elif "Close" in df.columns:
            df = df.rename(columns={"Close": "Price"})
        else:
            # If auto_adjust=True, yfinance already returns 'Close' adjusted
            if "Price" not in df.columns:
                raise ValueError("Expected 'Close' or 'Adj Close' in yfinance output.")

        if "Volume" not in df.columns:
            df["Volume"] = np.nan

        # Fill volume sensible defaults: forward-fill then a safe constant
        df["Volume"] = df["Volume"].replace(0, np.nan).fillna(method="ffill").fillna(100000.0)

        # Ensure enough rows
        if len(df) < 50:
            raise ValueError(f"Downloaded data too short ({len(df)} rows).")

        print(f" -> Download succeeded: {len(df)} rows")
        return df[["Price", "Volume"]].copy(), False

    except Exception as e:
        print(f" -> Download failed: {e}")
        if force_real:
            raise
        print(" -> Generating synthetic demo data (not for production evaluation).")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=1500, freq="B")  # business days
        prices = [10000.0]
        for _ in range(1, len(dates)):
            change = RNG.normal(0.0, 25.0)
            prices.append(max(50.0, prices[-1] + change))
        df_fake = pd.DataFrame(
            {"Price": prices, "Volume": 100000.0 + RNG.integers(-20000, 20000, size=len(dates))},
            index=dates
        )
        return df_fake, True


def compute_physics_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physics-inspired features using only past information.
    Returns a DataFrame extended with the new features.
    """
    df = df.copy()
    # Short EMA & rolling std (fast reaction)
    df["EMA_10"] = df["Price"].ewm(span=10, adjust=False).mean()
    df["E_STD_10"] = df["Price"].ewm(span=10, adjust=False).std().fillna(0.0)

    # Bands and normalized stiffness k
    df["Upper"] = df["EMA_10"] + 2 * df["E_STD_10"]
    df["Lower"] = df["EMA_10"] - 2 * df["E_STD_10"]
    df["Bandwidth"] = (df["Upper"] - df["Lower"]) / (df["EMA_10"].replace(0, np.nan))
    df["Bandwidth"] = df["Bandwidth"].fillna(df["Bandwidth"].median()).replace(0, 1e-9)
    df["k_raw"] = 1.0 / (df["Bandwidth"] + 1e-6)
    df["k"] = df["k_raw"] / (df["k_raw"].mean() + 1e-9)

    # Damping c from volume relative to a rolling average
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["c"] = (df["Volume"] / (df["Vol_MA20"].replace(0, np.nan))).fillna(1.0) * 0.5

    # Basic kinematics from returns (no future data)
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

    # integrate forward using only past values to compute kinematics
    for i in range(1, n):
        F_net = forces[i] - c_vals[i] * v[i - 1] - k_vals[i] * x[i - 1]
        a[i] = F_net / m
        jerk[i] = (a[i] - a[i - 1]) / (dt + 1e-12)
        snap[i] = (jerk[i] - jerk[i - 1]) / (dt + 1e-12)
        v[i] = v[i - 1] + a[i] * dt
        x[i] = x[i - 1] + v[i] * dt

    # Phys power; log-transform to reduce scale issues
    df["Phys_Power"] = np.log1p(np.abs(forces * v))
    df["Signal_Jerk"] = pd.Series(jerk, index=df.index).rolling(3, min_periods=1).mean()
    df["Signal_Snap"] = pd.Series(snap, index=df.index).rolling(3, min_periods=1).mean()

    # Cleanup / fill any remaining NaNs
    df[["k", "c", "Phys_Power", "Signal_Jerk", "Signal_Snap"]] = \
        df[["k", "c", "Phys_Power", "Signal_Jerk", "Signal_Snap"]].fillna(method="ffill").fillna(0.0)

    return df


def create_features_targets(df: pd.DataFrame, TRADING_DAYS: int = 21) -> pd.DataFrame:
    """
    Build features and the forecasting target:
    - Current_Vol: past realized vol over TRADING_DAYS (annualized)
    - Target_Vol: realized vol over the *future* TRADING_DAYS (annualized) aligned to current row
    """
    df = df.copy()

    # Past realized volatility (rolling window) — uses only past returns
    df["Past_RV"] = df["Log_Ret"].rolling(window=TRADING_DAYS).std()  # daily std
    df["Current_Vol"] = df["Past_RV"] * np.sqrt(252.0)

    # Future realized vol: compute rolling std of returns and shift backward so that the value
    # representing volatility over t+1..t+TRADING_DAYS is placed at index t
    df["Future_RV"] = df["Log_Ret"].shift(-1).rolling(window=TRADING_DAYS).std()
    df["Target_Vol"] = df["Future_RV"] * np.sqrt(252.0)

    # Add simple lagged features (no lookahead)
    df["Vol_Lag_1"] = df["Current_Vol"].shift(1)
    df["Ret_Lag_1"] = df["Log_Ret"].shift(1)

    # Keep features of interest
    df_model = df[[
        "Price",
        "Volume",
        "k", "Phys_Power", "Signal_Jerk", "Signal_Snap",
        "Current_Vol", "Vol_Lag_1", "Ret_Lag_1",
        "Target_Vol"
    ]].copy()

    # Drop rows with NaN in the target or main features
    df_model = df_model.dropna().copy()
    return df_model


def bs_pricer(S, K, T, r, sigma):
    """
    Black-Scholes call price, stable against sigma == 0 and scalar T broadcasting.
    If sigma is extremely small or T <= 0, returns intrinsic present value (approx).
    S, K, sigma, T can be numpy arrays or scalars; we broadcast to a common shape.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    eps = 1e-6

    # Broadcast all to a common shape to avoid indexing errors with scalar T
    S, K, sigma, T = np.broadcast_arrays(S, K, sigma, T)

    # Masks
    small_sigma_mask = (sigma <= eps) | (T <= eps)

    # Normal BS path
    sig = np.clip(sigma, eps, None)
    t = np.clip(T, eps, None)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    bs_prices = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    # Intrinsic (discounted) for degenerate cases
    intrinsic = np.maximum(S - K * np.exp(-r * T), 0.0)

    # Combine
    prices = np.where(small_sigma_mask, intrinsic, bs_prices)
    return prices


def ewma_volatility(returns: pd.Series, span: int = 21) -> float:
    """
    Return one-step EWMA volatility estimate (annualized) computed on the provided returns series.
    """
    ret = returns.dropna()
    if len(ret) < 2:
        return float(np.nan)
    lam = 2.0 / (span + 1.0)  # alpha
    ewma_var = (ret ** 2).ewm(alpha=lam, adjust=False).mean().iloc[-1]
    return float(np.sqrt(ewma_var) * np.sqrt(252.0))


def garch_forecast_variance(returns: pd.Series, horizon: int = 21) -> float:
    """
    Fit a simple GARCH(1,1) on returns (if arch available) and forecast annualized vol for the given horizon.
    Returns sqrt(variance)*sqrt(252).
    If arch is not available or fit fails, returns np.nan.
    """
    if not HAS_ARCH:
        return float(np.nan)
    try:
        am = arch_model(returns.dropna() * 100.0, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
        res = am.fit(disp="off", show_warning=False)
        fh = res.forecast(horizon=horizon, reindex=False)
        # Average the path to avoid excessive end-horizon noise
        var_path = fh.variance.iloc[-1].values / (100.0 ** 2)
        var_avg = float(np.mean(var_path))
        return float(np.sqrt(var_avg) * np.sqrt(252.0))
    except Exception:
        return float(np.nan)


def walk_forward_forecast(df_model: pd.DataFrame,
                          features: list,
                          target_name: str = "Target_Vol",
                          model=None,
                          initial_train_frac: float = 0.5,
                          test_window: int = None) -> pd.DataFrame:
    """
    Perform walk-forward forecasting with expanding training window and fixed-size test windows.
    Returns test DataFrame with added prediction columns (Predicted_Vol, Base_Persist_Vol, Base_EWMA_Vol, Base_GARCH_Vol).
    """
    n = len(df_model)
    if test_window is None:
        test_window = max(1, int(0.15 * n))

    initial_train = max(10, int(initial_train_frac * n))
    indices = df_model.index

    # Prepare container for predictions
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

        # Fit pipeline on train
        pipe = clone(pipeline)
        if hasattr(pipe.named_steps["model"], "random_state"):
            pipe.named_steps["model"].random_state = 42
        pipe.fit(train_df[features], train_df[target_name])

        # Store predicted vols
        preds.iloc[test_idx] = pipe.predict(test_df[features])

        # Baseline: persistence (Current_Vol)
        baseline_persist.iloc[test_idx] = test_df["Current_Vol"].values

        # Baseline: EWMA computed using returns up to train end (one forecast per test window)
        try:
            train_returns = train_df["Ret_Lag_1"].dropna()
            ewma_v = ewma_volatility(train_returns, span=21)
        except Exception:
            # fallback: use last Current_Vol from train
            ewma_v = float(train_df["Current_Vol"].iloc[-1])
        baseline_ewma.iloc[test_idx] = ewma_v

        # Baseline: GARCH (optional)
        try:
            garch_v = garch_forecast_variance(train_df["Ret_Lag_1"].fillna(0.0), horizon=21)
        except Exception:
            garch_v = float(np.nan)
        baseline_garch.iloc[test_idx] = garch_v

        start += step

    # Assemble into DataFrame
    results = df_model.copy()
    results["Predicted_Vol"] = preds
    results["Base_Persist_Vol"] = baseline_persist
    results["Base_EWMA_Vol"] = baseline_ewma
    results["Base_GARCH_Vol"] = baseline_garch

    # Drop rows where we couldn't predict (head training portion)
    results = results.dropna(subset=["Predicted_Vol", "Base_Persist_Vol"])
    return results


def evaluate_price_errors(test_df: pd.DataFrame, T_days: int = 30, r: float = 0.07) -> Dict[str, float]:
    """
    Evaluate mean absolute price error of different vol sources against the 'ideal' using Target_Vol.
    Returns dict of MAE/RMSE for Baselines and Model, plus % improvement vs persistence.
    """
    T = T_days / 365.0
    S = test_df["Price"].values
    K = S  # ATM
    target_sigma = test_df["Target_Vol"].values
    pred_sigma = test_df["Predicted_Vol"].values
    base_persist = test_df["Base_Persist_Vol"].values
    base_ewma = test_df.get("Base_EWMA_Vol", pd.Series(np.nan, index=test_df.index)).values
    base_garch = test_df.get("Base_GARCH_Vol", pd.Series(np.nan, index=test_df.index)).values

    price_ideal = bs_pricer(S, K, T, r, target_sigma)
    price_ai    = bs_pricer(S, K, T, r, pred_sigma)
    price_pers  = bs_pricer(S, K, T, r, base_persist)
    price_ewma  = bs_pricer(S, K, T, r, base_ewma)
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
    # % improvement vs persistence (MAE)
    base = metrics["MAE_Persistence"]
    metrics["MAE_Improvement_pct_Model_vs_Persistence"] = ((base - metrics["MAE_Model"]) / base * 100.0) if base != 0 else float("nan")
    metrics["MAE_Improvement_pct_EWMA_vs_Persistence"] = ((base - metrics["MAE_EWMA"]) / base * 100.0) if base != 0 else float("nan")
    metrics["MAE_Improvement_pct_GARCH_vs_Persistence"] = ((base - metrics["MAE_GARCH"]) / base * 100.0) if base != 0 else float("nan")
    return metrics


def print_metrics(df_results: pd.DataFrame):
    """
    Print evaluation metrics comparing model vs baselines on volatility prediction and price MAE.
    Uses numpy with nan-safe metrics.
    """
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

    # Price-level evaluation
    price_stats = evaluate_price_errors(df_results)
    print("\nPrice-level evaluation vs realized future vol (Target):")
    for k, v in price_stats.items():
        print(f" - {k}: {v:.4f}")
    print()


def plot_results(df_results: pd.DataFrame, title_suffix=""):
    """
    Quick visual diagnostics. Saves to PNG and attempts to show.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Prices and Snap
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


def parse_args():
    parser = argparse.ArgumentParser(description="Volatility forecasting pipeline")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Ticker symbol (e.g., ^NSEI, ^GSPC, SPY)")
    parser.add_argument("--period", type=str, default="10y", help="Download period (e.g., 5y, 10y, max)")
    parser.add_argument("--force-real", action="store_true", help="Fail if download fails (no synthetic fallback)")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to file")
    parser.add_argument("--t-days", type=int, default=30, help="Option maturity in days for price-level eval")
    parser.add_argument("--rate", type=float, default=0.07, help="Risk-free rate for BS pricer")
    parser.add_argument("--trading-days", type=int, default=21, help="Rolling window for realized vol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--auto-adjust", action="store_true", help="Use yfinance auto_adjust=True")

    # NOTE: use parse_known_args to avoid Jupyter's -f arg crash
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    # Reproducibility
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(args.seed)

    # Get data
    df_raw, is_synthetic = get_data(ticker=args.ticker, period=args.period, force_real=args.force_real, auto_adjust=args.auto_adjust)
    if is_synthetic:
        print("Warning: using synthetic data — results are for demo only.")

    # Compute features
    df_phys = compute_physics_signals(df_raw)
    df_model = create_features_targets(df_phys, TRADING_DAYS=args.trading_days)
    print(f"Prepared dataset with {len(df_model)} rows for modeling.")

    # Define features used for forecasting
    features = ['k', 'Phys_Power', 'Signal_Jerk', 'Signal_Snap', 'Current_Vol', 'Vol_Lag_1', 'Ret_Lag_1']

    # Ensure enough data
    if len(df_model) < 50:
        print("Not enough data to run walk-forward evaluation. Exiting.")
        return

    # Model
    base_model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=500, max_depth=6,
        loss='absolute_error', random_state=args.seed
    )

    # Walk-forward forecasting
    test_window = max(10, int(0.1 * len(df_model)))
    df_results = walk_forward_forecast(
        df_model, features, target_name="Target_Vol",
        model=base_model, initial_train_frac=0.6, test_window=test_window
    )
    print(f"Walk-forward produced {len(df_results)} evaluated rows.")

    # Print primary metrics
    print_metrics(df_results)

    # Show diagnostic plot
    title_suffix = f" ({args.ticker})"
    plot_results(df_results, title_suffix=title_suffix if args.save_plot else "")

    # Optionally: inspect feature importances by training on full training portion
    final_train_end = int(0.85 * len(df_model))
    final_train = df_model.iloc[:final_train_end]
    final_test = df_model.iloc[final_train_end:]
    pipe_full = Pipeline([("scaler", StandardScaler()), ("model", clone(base_model))])
    pipe_full.fit(final_train[features], final_train["Target_Vol"])
    try:
        importances = pipe_full.named_steps["model"].feature_importances_
        print("Feature importances (trained on 85% of data):")
        for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
            print(f" - {f}: {imp:.4f}")
    except Exception:
        print("Could not extract feature importances from model.")

    # Final price-level evaluation on the final test slice
    if len(final_test) > 0:
        final_test_preds = pipe_full.predict(final_test[features])
        final_test = final_test.copy()
        final_test["Predicted_Vol"] = final_test_preds
        final_test["Base_Persist_Vol"] = final_test["Current_Vol"]
        # Recompute EWMA/GARCH baselines for final slice (using entire final_train history)
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

    print("Done.")


if __name__ == "__main__":
    main()
