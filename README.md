Gamma Levels Pipeline
Daily SPX gamma levels (call wall / put wall / zero gamma) computed from free
options data, converted to NQ-equivalent, published as JSON, consumed by AlphaFlowX.
Repo layout
```
.github/workflows/build-gamma.yml    # cron @ 6:30 AM ET
scripts/build_gamma.py               # fetcher + calculator
data/gamma.json                      # latest (overwritten daily)
data/gamma.sample.json               # schema reference
data/history/YYYY-MM-DD.json         # daily snapshots (nice for backtest later)
```
Setup checklist
Create public repo (must be public so AlphaFlowX can fetch without a token).
`steelytraders/gamma-levels` or similar.
Drop these files in:
```
   .github/workflows/build-gamma.yml
   scripts/build_gamma.py
   data/.gitkeep
   data/history/.gitkeep
   ```
Push. The Action will not run automatically until next scheduled tick — use the
"Run workflow" button on the Actions tab to test it now.
Once `data/gamma.json` exists, grab its raw URL:
`https://raw.githubusercontent.com/<user>/<repo>/main/data/gamma.json`
Paste that into `GAMMA_URL` in `AlphaFlowX_GammaFetcher.cs`.
Things to verify on first run
Action logs show `SPX=… NQ=… ratio=…` and sensible level values.
`data/gamma.json` committed with today's `trade_date`.
Manually hit the raw URL in browser → JSON loads.
Compile AlphaFlowX → load chart → check NT8 output window for `[Gamma] loaded …`.
Known limitations of free data
yfinance OI is end-of-prior-day; 6:30 AM run reflects last close's positioning. For
daily levels that's fine — gamma walls don't shift intraday off-flow that much.
IV from yfinance is snapshot; gamma is recomputed via Black-Scholes for consistency.
No dealer-positioning model — we assume the standard "dealers short calls, short puts"
convention. If SpotGamma's numbers don't match, this is why.
DTE_MAX=45 ignores LEAPS; if you want longer-dated influence, bump it.
Upgrade paths later
Add vanna/charm (need different greeks calc, same chain data).
Switch source to CBOE direct (more accurate OI) — paid tier.
Add Discord webhook step to ping #gamma-levels with the day's numbers.
Add a midday refresh at 12:00 ET if you want intraday updates.
Cost
Zero. GitHub Actions free tier is 2000 min/month for public repos, this job uses ~1 min/day.
