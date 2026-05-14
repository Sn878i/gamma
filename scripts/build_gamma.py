"""
build_gamma.py
Fetches SPX (^SPX) option chain via yfinance, computes call wall / put wall / zero gamma,
converts to NQ-equivalent using live SPX/NQ ratio, writes data/gamma.json.

Free-source caveats:
- yfinance OI updates overnight; pre-market run captures yesterday's close OI.
- IV is yfinance's snapshot; we recompute gamma with Black-Scholes for consistency.
- No vega/charm — just gamma, which is all the spec calls for.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ---------- config ----------
SPX_TICKER = "^SPX"
NQ_TICKER  = "NQ=F"          # front-month NQ continuous
RISK_FREE  = 0.045           # rough 3M T-bill; refine later if you want
DTE_MAX    = 45              # ignore far-dated tail
STRIKE_WINDOW = 0.10         # +/- 10% of spot
OUT_PATH   = Path("data/gamma.json")
HIST_DIR   = Path("data/history")


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def fetch_chain(ticker_symbol: str, spot: float) -> pd.DataFrame:
    """Pull all expiries within DTE_MAX, return flat DataFrame of calls+puts."""
    t = yf.Ticker(ticker_symbol)
    today = datetime.now(timezone.utc).date()
    rows = []

    for exp_str in t.options:
        exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp - today).days
        if dte <= 0 or dte > DTE_MAX:
            continue

        chain = t.option_chain(exp_str)
        T = max(dte, 1) / 365.0

        for df, kind in [(chain.calls, "C"), (chain.puts, "P")]:
            sub = df[["strike", "openInterest", "impliedVolatility"]].copy()
            sub = sub.dropna()
            sub = sub[sub["openInterest"] > 0]
            # filter to reasonable strike window
            sub = sub[(sub["strike"] > spot * (1 - STRIKE_WINDOW)) &
                      (sub["strike"] < spot * (1 + STRIKE_WINDOW))]
            sub["kind"] = kind
            sub["dte"] = dte
            sub["T"] = T
            rows.append(sub)

    if not rows:
        raise RuntimeError("No option chain rows fetched.")
    return pd.concat(rows, ignore_index=True)


def compute_levels(chain: pd.DataFrame, spot: float) -> dict:
    """Returns {call_wall, put_wall, zero_gamma} as float strikes."""
    # signed dealer GEX per row (dealers short calls = +gamma exposure long for them when
    # they hedge; convention here: +calls, -puts, common dealer-net-short assumption)
    chain["gamma"] = chain.apply(
        lambda r: bs_gamma(spot, r["strike"], r["T"], RISK_FREE, r["impliedVolatility"]),
        axis=1,
    )
    chain["gex"] = chain["gamma"] * chain["openInterest"] * 100 * spot * spot * 0.01
    chain.loc[chain["kind"] == "P", "gex"] *= -1

    # aggregate by strike
    by_strike = chain.groupby("strike").agg(
        call_gex=("gex", lambda s: s[chain.loc[s.index, "kind"] == "C"].sum()),
        put_gex =("gex", lambda s: s[chain.loc[s.index, "kind"] == "P"].sum()),
        net_gex =("gex", "sum"),
    ).reset_index().sort_values("strike")

    # call wall: max positive call_gex strike above spot
    above = by_strike[by_strike["strike"] > spot]
    call_wall = float(above.loc[above["call_gex"].idxmax(), "strike"]) if not above.empty else None

    # put wall: max abs put_gex strike below spot (put_gex is negative, want most negative)
    below = by_strike[by_strike["strike"] < spot]
    put_wall = float(below.loc[below["put_gex"].idxmin(), "strike"]) if not below.empty else None

    # zero gamma: cumulative net_gex sign change strike
    by_strike["cum_gex"] = by_strike["net_gex"].cumsum()
    sign = np.sign(by_strike["cum_gex"].values)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips) > 0:
        # linear interp between bracketing strikes
        i = flips[0]
        s0, s1 = by_strike["strike"].iloc[i], by_strike["strike"].iloc[i + 1]
        g0, g1 = by_strike["cum_gex"].iloc[i], by_strike["cum_gex"].iloc[i + 1]
        zero_gamma = float(s0 + (s1 - s0) * (-g0) / (g1 - g0))
    else:
        zero_gamma = None

    return {"call_wall": call_wall, "put_wall": put_wall, "zero_gamma": zero_gamma}


def spx_to_nq(spx_level: float, ratio: float) -> float:
    """Convert SPX strike to NQ-equivalent price using current SPX/NQ ratio."""
    if spx_level is None:
        return None
    return round(spx_level / ratio, 2)


def main():
    spx = yf.Ticker(SPX_TICKER).history(period="1d")
    nq  = yf.Ticker(NQ_TICKER).history(period="1d")
    if spx.empty or nq.empty:
        print("ERROR: spot fetch failed", file=sys.stderr)
        sys.exit(1)

    spx_spot = float(spx["Close"].iloc[-1])
    nq_spot  = float(nq["Close"].iloc[-1])
    ratio    = spx_spot / nq_spot   # ~0.28 typically

    print(f"SPX={spx_spot:.2f}  NQ={nq_spot:.2f}  ratio={ratio:.5f}")

    chain  = fetch_chain(SPX_TICKER, spx_spot)
    levels = compute_levels(chain, spx_spot)
    print("SPX levels:", levels)

    nq_levels = {k: spx_to_nq(v, ratio) for k, v in levels.items()}
    print("NQ levels:", nq_levels)

    now = datetime.now(timezone.utc)
    payload = {
        "version": "1.0",
        "generated_at": now.isoformat(),
        "trade_date": now.date().isoformat(),
        "source": "yfinance/SPX",
        "spx_spot": round(spx_spot, 2),
        "nq_spot": round(nq_spot, 2),
        "spx_nq_ratio": round(ratio, 5),
        "spx": levels,
        "nq": nq_levels,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    # snapshot for history / backtest
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    (HIST_DIR / f"{payload['trade_date']}.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
