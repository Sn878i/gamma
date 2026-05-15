"""
build_gamma.py <instrument>
  <instrument> = "nq" (uses QQQ chain) or "es" (uses SPY chain)

Computes gamma_flip, put_wall, call_wall, hvl, vol_trigger for the futures
instrument by analyzing the corresponding ETF option chain and scaling levels
to futures price via the live ETF/future ratio.

Schema (matches IUXX/Barchart convention):
{
  "instrument": "nq",
  "symbol": "$IUXX",
  "date": "...",
  "fetched_at": "...",
  "gamma_flip": ...,
  "put_wall": ...,
  "call_wall": ...,
  "hvl": ...,            # synonym for gamma_flip
  "vol_trigger": ...,    # last positive-gamma strike below spot
  "source": "yfinance_gex",
  "stale": false
}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ---------- instrument config ----------
INSTRUMENTS = {
    "nq": {
        "etf_ticker": "QQQ",
        "future_ticker": "NQ=F",
        "symbol": "$IUXX",
    },
    "es": {
        "etf_ticker": "SPY",
        "future_ticker": "ES=F",
        "symbol": "$ISPX",
    },
}

# ---------- math config ----------
RISK_FREE     = 0.045
DTE_MAX       = 60
STRIKE_WINDOW = 0.20

# ---------- output paths ----------
OUT_DIR  = Path("data/gamma")
HIST_DIR = Path("data/history")


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
    """Returns ETF-scale strikes for all 5 levels."""
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
    ).reset_index().sort_values("strike").reset_index(drop=True)

    # call wall = strike above spot with max call GEX
    above = by_strike[by_strike["strike"] > spot]
    call_wall = float(above.loc[above["call_gex"].idxmax(), "strike"]) if not above.empty else None

    # put wall = strike below spot with most negative put GEX
    below = by_strike[by_strike["strike"] < spot]
    put_wall = float(below.loc[below["put_gex"].idxmin(), "strike"]) if not below.empty else None

    # gamma_flip / hvl = interpolated zero crossing of cumulative net GEX
    by_strike["cum_gex"] = by_strike["net_gex"].cumsum()
    sign = np.sign(by_strike["cum_gex"].values)
    flips = np.where(np.diff(sign) != 0)[0]
    if len(flips) > 0:
        i = flips[0]
        s0, s1 = by_strike["strike"].iloc[i], by_strike["strike"].iloc[i + 1]
        g0, g1 = by_strike["cum_gex"].iloc[i], by_strike["cum_gex"].iloc[i + 1]
        gamma_flip = float(s0 + (s1 - s0) * (-g0) / (g1 - g0))
    else:
        gamma_flip = None

    # vol_trigger = highest strike below spot where net_gex is still positive
    below_pos = by_strike[(by_strike["strike"] < spot) & (by_strike["net_gex"] > 0)]
    vol_trigger = float(below_pos["strike"].max()) if not below_pos.empty else None

    return {
        "call_wall":   call_wall,
        "put_wall":    put_wall,
        "gamma_flip":  gamma_flip,
        "hvl":         gamma_flip,    # synonym
        "vol_trigger": vol_trigger,
    }


def scale_to_future(etf_levels: dict, ratio: float) -> dict:
    """ratio = future_spot / etf_spot. Apply to all 5 levels."""
    return {k: (round(v * ratio, 2) if v is not None else None)
            for k, v in etf_levels.items()}


def build(instrument: str):
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {instrument}. Use 'nq' or 'es'.")
    cfg = INSTRUMENTS[instrument]

    etf    = yf.Ticker(cfg["etf_ticker"]).history(period="1d")
    future = yf.Ticker(cfg["future_ticker"]).history(period="1d")
    if etf.empty or future.empty:
        print(f"ERROR: spot fetch failed for {instrument}", file=sys.stderr)
        sys.exit(1)

    etf_spot    = float(etf["Close"].iloc[-1])
    future_spot = float(future["Close"].iloc[-1])
    ratio       = future_spot / etf_spot

    print(f"[{instrument.upper()}] {cfg['etf_ticker']}={etf_spot:.2f}  "
          f"{cfg['future_ticker']}={future_spot:.2f}  ratio={ratio:.4f}")

    chain         = fetch_chain(cfg["etf_ticker"], etf_spot)
    etf_levels    = compute_levels(chain, etf_spot)
    future_levels = scale_to_future(etf_levels, ratio)

    print(f"[{instrument.upper()}] ETF levels:    {etf_levels}")
    print(f"[{instrument.upper()}] Future levels: {future_levels}")

    missing = [k for k, v in future_levels.items() if v is None]
    stale   = len(missing) > 0
    if missing:
        print(f"WARNING [{instrument}]: missing levels: {missing} -- marked stale=true",
              file=sys.stderr)

    now = datetime.now(timezone.utc)
    payload = {
        "instrument":  instrument,
        "symbol":      cfg["symbol"],
        "date":        now.date().isoformat(),
        "fetched_at":  now.strftime("%Y-%m-%dT%H%M.%f")[:-3],
        "gamma_flip":  future_levels["gamma_flip"],
        "put_wall":    future_levels["put_wall"],
        "call_wall":   future_levels["call_wall"],
        "hvl":         future_levels["hvl"],
        "vol_trigger": future_levels["vol_trigger"],
        "source":      "yfinance_gex",
        "stale":       stale,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{instrument}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    day_dir = HIST_DIR / now.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"{instrument}.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: build_gamma.py <nq|es>", file=sys.stderr)
        sys.exit(2)
    build(sys.argv[1].lower())


if __name__ == "__main__":
    main()
