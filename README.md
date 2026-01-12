# SnapSigma — Physics‑Inspired Volatility Engine (NIFTY)

SnapSigma is an experimental pipeline that combines physics‑inspired signals (spring/damper dynamics and higher derivatives like jerk/snap) with a tree‑based regressor to forecast future realized volatility for the Indian National Stock Exchange index (NIFTY / `^NSEI`). The predicted volatility is used in a Black–Scholes pricer to compare model option prices against a realized "ideal" price and simple baselines.

This repository is a research/prototype project — it is not production-ready. See "Limitations & Warnings" below.

---

Table of contents
- Overview
- Key features
- Quickstart
- Installation
- Usage
- Configuration options
- How it works (high-level)
- Evaluation & validation
- Suggestions & next steps
- Limitations & warnings
- Contributing
- License
- Contact

---

Overview
--------
SnapSigma explores whether responsive, physics-motivated features derived from price & volume dynamics can improve short-term realized volatility forecasts versus classical baselines (persistence, EWMA, optional GARCH). The project includes:
- robust data ingestion with a synthetic-data safe fallback (for demo runs),
- physics signal construction (EMA, stiffness k, damping c, force, acceleration, jerk, snap),
- a stable Black–Scholes pricer that guards against sigma ≈ 0,
- walk‑forward (expanding) time‑series validation with baselines,
- simple plotting and metrics (MAE/RMSE on vol and price-level MAE vs realized future vol).

Key features
------------
- Physics-inspired inputs: stiffness (k), Phys_Power, Signal_Jerk, Signal_Snap
- Target: future realized volatility over a TRADING_DAYS window (annualized)
- Model: HistGradientBoostingRegressor inside an sklearn Pipeline with StandardScaler
- Walk-forward evaluation to avoid naive train/test leakage
- Baselines: persistence (current realized vol), EWMA, and optional GARCH (requires `arch`)
- Numerically stable Black–Scholes pricer (handles sigma → 0 and T �� 0)
- Reproducible synthetic data mode for demos (explicit flag if you require real data)

Quickstart
----------
1. Clone the repo:
   git clone https://github.com/manishsh-collab/AI_NationalStockExIndiaPhysicsVolitalityModel.git
2. Create a virtual environment and install dependencies (see below).
3. Run the main script:
   python main.py

Installation
------------
It is recommended to run inside a virtual environment.

pip install -r requirements.txt

If a requirements file is not available, install the minimum required packages:

pip install numpy pandas yfinance matplotlib scikit-learn scipy

Optional (for GARCH baseline):
pip install arch

Usage
-----
The primary entrypoint is `main.py`. Running it will:
- download NIFTY (`^NSEI`) price & volume (10y by default) via `yfinance` (if available),
- compute physics features,
- build features and target (future realized vol),
- run walk-forward forecasting + baselines,
- print metrics and show diagnostic plots,
- optionally compute feature importances and final held-out price MAE.

Important runtime notes:
- If `yfinance` download fails the script will generate synthetic demo data. This synthetic mode is suitable for testing the pipeline but not for real evaluation. To force the script to error if real data cannot be downloaded, change `get_data(..., force_real=True)` in `main.py`.
- Default option/config parameters are placed in `main()` of `main.py` (e.g., TRADING_DAYS, T_days for option tenor, walk-forward settings). Tweak as needed.

Configuration options
---------------------
Key variables in `main.py` you may want to tune:
- ticker (default `^NSEI`)
- period (default `"10y"`) passed to yfinance
- TRADING_DAYS (default 21) — window for realized vol
- T_days (default 30) — option tenor used by the price comparison
- walk-forward settings: `initial_train_frac`, `test_window` inside `walk_forward_forecast`

How it works (high-level)
-------------------------
1. Data ingestion:
   - Download price & volume (or produce synthetic data).
   - Normalize column names and fill missing volume sensibly.
2. Physics signals:
   - Short EMA and EMA-based std produce bands → compute a normalized stiffness `k`.
   - Damping `c` from volume relative to a 20-day MA.
   - Convert log returns to a "force" and integrate a simple discrete spring-mass-damper forward to obtain acceleration, velocity, jerk, snap, and Phys_Power.
   - All signals are computed using only historical/past information to avoid lookahead.
3. Targets:
   - Target_Vol = realized volatility measured over the next TRADING_DAYS (annualized).
   - Current_Vol = realized volatility over the previous TRADING_DAYS (annualized).
4. Modeling & evaluation:
   - Train a HistGradientBoostingRegressor inside a Pipeline with StandardScaler using an expanding (walk-forward) training window and fixed test windows.
   - Baselines: persistence (Current_Vol), EWMA (one-step EWMA applied as naive forecast), optional GARCH via `arch`.
   - Evaluate both vol forecasts (MAE/RMSE) and price-level MAE where predicted vols are plugged into Black–Scholes and compared to the BS price using realized future vol (the "ideal" price).

Evaluation & validation
-----------------------
- The provided evaluation is a prototype: MAE/RMSE on vol and price MAE vs realized. To claim effectiveness you should:
  - Use time-series cross-validation and multiple non-overlapping test slices,
  - Add statistical tests and confidence intervals,
  - Perform a proper options P&L/backtest (requires implied vol or option price data),
  - Run hyperparameter search with time-series aware CV,
  - Avoid synthetic-data results when presenting effectiveness.

Suggestions & next steps
------------------------
- Replace synthetic fallback with an explicit error for production use (or keep synthetic only for demos).
- Add expanding-window hyperparameter tuning (e.g., GridSearchCV with custom time-series splits).
- Collect market-implied vol or option price data to evaluate a trading/backtest strategy meaningfully.
- Add unit tests that assert no lookahead (small concrete examples where target alignment is checked).
- Try probabilistic models (quantile regression forests, conformal prediction) for uncertainty around vol forecasts.

Limitations & warnings
----------------------
- This is research‑grade prototype code. Do not use it for live trading without extended validation.
- Synthetic fallback data is for development/testing only and must not be used to claim out-of-sample performance.
- Forecasting realized volatility is hard and noisy — expected predictive power is limited on short horizons.
- Numerical edge cases are guarded in the BS pricer, but ensure predicted sigma values are reasonable before deploying.

Contributing
------------
Contributions and improvements are welcome. Suggested areas:
- Add time-series unit tests to guarantee no lookahead,
- Integrate market-implied vols and option market data for a trading backtest,
- Add hyperparameter optimization and robust model selection,
- Add benchmarking against more baselines (GARCH, stochastic volatility models).

License
-------
MIT License — see LICENSE file (or add one if absent).

Contact
-------
For questions, issues or collaboration ideas, open an issue on the repository or contact the maintainer.

---

If you want, I can:
- Create a polished `requirements.txt` and `environment.yml`.
- Add a sample `LICENSE` (MIT) file.
- Produce a short `CONTRIBUTING.md` with PR template and testing instructions.
- Produce a small unit-test suite that asserts target alignment and no lookahead.

Which of the above would you like next?
