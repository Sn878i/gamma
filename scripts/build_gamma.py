"""
build_gamma.py
Fetches SPY option chain via yfinance (denser OI than ^SPX), computes call wall /
put wall / zero gamma, converts SPY -> SPX-equivalent -> NQ-equivalent, writes
data/gamma.json.

Why SPY instead of ^SPX:
- SPY has 10x more strikes traded and denser OI distribution.
- ^SPX often has gaps that cause put_wall / zero_gamma to compute as None.
- SPY price ~= SPX / 10, so we multiply SPY strikes by 10 to get SPX equivalents.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ---------- config ----------
SPY_TICKER = "SPY"
NQ_TICKER  = "NQ=F"
SPY_TO_SPX = 10.0
RISK_FREE  = 0.045
DTE_MAX    = 45
STRIKE_WINDOW = 0.15
OUT_PATH   = Path("data/gamma.json")
HIST_DIR   = Path("data/history")


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def fetch_chain(ticker_symbol: str, spot: float) -> pd.DataFrame:
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
            sub = sub[(sub["strike"] > spot * (1 - STRIKE_WINDOW)) &
                      (sub["strike"] < spot * (1 + STRIKE_WINDOW))]
            sub["kind"] = kind
            sub["dte"] = dte
            sub["T"] = T
            rows.append(sub)

    if not rows:
        raise RuntimeError("No option chain rows fetched.")
    out = pd.concat(rows, ignore_index=True)
    print(f"Chain: {len(out)} rows, "
          f"strikes {out['strike'].min():.2f}-{out['strike'].max():.2f}, "
          f"OI total {int(out['openInterest'].sum()):,}")
    return out


def compute_levels(chain: pd.DataFrame, spot: float) -> dict:
    chain["gamma"] = chain.apply(
        lambda r: bs_gamma(spot, r["strike"], r["T"], RISK_FREE, r["impliedVolatility"]),
        axis=1,
    )
    chain["gex"] = chain["gamma"] * chain["openInterest"] * 100 * spot * spot * 0.01
    chain.loc[chain["kind"] == "P", "gex"] *= -1

    by_strike = chain.groupby("strike").agg(
        call_gex=("gex", lambda s: s[chain.loc[s.index, "kind"] == "C"].sum()),
        put_gex =("gex", lambda s: s[chain.loc[s.index, "kind"] == "P"].sum()),
        net_gex =("gex", "sum"),
    ).reset_index().sort_values("strike")

    above = by_strike[by_strike["strike"] > spot]
    call_wall = float(above.loc[above["call_gex"].idxmax(), "strike"]) if not above.empty else None

    below = by_strike[by_strike["strike"] < spot]
    put_wall = float(below.loc[below["put_gex"].idxmin(), "strike"]) if not below.empty else None

    by_strike["cum_gex"] = by_strike["net_gex"].cumsum()
    sign = np.sign(by_strike["cum_gex"].values)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips) > 0:
        i = flips[0]
        s0, s1 = by_strike["strike"].iloc[i], by_strike["strike"].iloc[i + 1]
        g0, g1 = by_strike["cum_gex"].iloc[i], by_strike["cum_gex"].iloc[i + 1]
        zero_gamma = float(s0 + (s1 - s0) * (-g0) / (g1 - g0))
    else:
        zero_gamma = None

    return {"call_wall": call_wall, "put_wall": put_wall, "zero_gamma": zero_gamma}


def scale_to_spx(spy_levels: dict) -> dict:
    return {k: (round(v * SPY_TO_SPX, 2) if v is not None else None)
            for k, v in spy_levels.items()}


def spx_to_nq(spx_levels: dict, ratio: float) -> dict:
    return {k: (round(v / ratio, 2) if v is not None else None)
            for k, v in spx_levels.items()}


def main():
    spy = yf.Ticker(SPY_TICKER).history(period="1d")
    nq  = yf.Ticker(NQ_TICKER).history(period="1d")
    if spy.empty or nq.empty:
        print("ERROR: spot fetch failed", file=sys.stderr)
        sys.exit(1)

    spy_spot = float(spy["Close"].iloc[-1])
    nq_spot  = float(nq["Close"].iloc[-1])
    spx_spot_est = spy_spot * SPY_TO_SPX
    ratio = spx_spot_est / nq_spot

    print(f"SPY={spy_spot:.2f}  SPX~={spx_spot_est:.2f}  NQ={nq_spot:.2f}  SPX/NQ ratio={ratio:.5f}")

    chain = fetch_chain(SPY_TICKER, spy_spot)
    spy_levels = compute_levels(chain, spy_spot)
    spx_levels = scale_to_spx(spy_levels)
    nq_levels  = spx_to_nq(spx_levels, ratio)

    print(f"SPY levels: {spy_levels}")
    print(f"SPX levels: {spx_levels}")
    print(f"NQ  levels: {nq_levels}")

    missing = [k for k, v in spx_levels.items() if v is None]
    if missing:
        print(f"WARNING: missing levels: {missing} -- "
              f"consider widening STRIKE_WINDOW or checking yfinance data quality",
              file=sys.stderr)

    now = datetime.now(timezone.utc)
    payload = {
        "version": "1.0",
        "generated_at": now.isoformat(),
        "trade_date": now.date().isoformat(),
        "source": "yfinance/SPY",
        "spy_spot": round(spy_spot, 2),
        "spx_spot_est": round(spx_spot_est, 2),
        "nq_spot": round(nq_spot, 2),
        "spx_nq_ratio": round(ratio, 5),
        "spy": spy_levels,
        "spx": spx_levels,
        "nq": nq_levels,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    (HIST_DIR / f"{payload['trade_date']}.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
