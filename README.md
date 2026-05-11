# Quaternionic Black-Scholes

An experiment lifting the Black-Scholes option pricing formula from ℝ into ℍ (the quaternions), encoding four market dimensions simultaneously:

| Component | Dimension |
|-----------|-----------|
| `t` (real) | standard log-price / option value |
| `p` (i) | momentum / funding-rate-adjusted price |
| `f` (j) | liquidity-adjusted price |
| `l` (k) | emergent cross-effect (funding × liquidity) |

When all imaginary components are zero the formulas reduce exactly to classical 1973 Black-Scholes.

## What's here

- **`src/quaternion/`** — quaternion math (multiply, exp, log, norm, conjugate)
- **`src/bs-model/`** — quaternionic BS pricing, Greeks, and implied vol inversion
- **`src/okx/`** — OKX API client for spot candles and perpetual funding rates
- **`scripts/fetch-prices.ts`** — fetches BTC/USDT daily OHLCV + funding rates into `data/`
- **`scripts/backtest.ts`** — backtests four models against 30-day ATM call payoffs

## Backtest results

30-day ATM calls on BTC/USDT, 240 observations (2025–2026).

| Model | n | MAE | RMSE | Bias |
|-------|---|-----|------|------|
| Classical BS | 240 | 3503 | 3941 | +1503 |
| EWMA vol (λ=0.94) | 240 | 3572 | 4023 | +1516 |
| Carry-adjusted (Merton) | 62 | 3627 | 4470 | −1083 |
| Quaternionic | 62 | 3608 | 4567 | −2487 |

Classical and carry-adjusted are essentially tied. EWMA is slightly worse — BTC vol is mean-reverting enough that equal-weighted historical vol beats an exponentially weighted estimator. The quaternionic model is a clear loser.

## Why it didn't work

The quaternionic extension has no principled financial justification. BS derives from GBM + no-arbitrage + continuous hedging; lifting to quaternions doesn't relax any of those assumptions. In practice, encoding momentum and funding rate as imaginary spot components produces cross-terms in the Hamilton product that subtract from the call price when the market is bullish — the opposite of what you'd want.

The dominant source of error across all models is vol forecasting: backward-looking 30-day realized vol is a poor predictor of vol over the next 30 days regardless of how you weight it. Beating classical BS here would require actual market-implied vol (Deribit/OKX options surface) or a calibrated stochastic vol model (Heston, SABR).

## What is interesting

The imaginary output components — `call.p`, `call.f`, `call.l` — are genuine first-order price sensitivities to funding and liquidity dimensions. They don't improve the scalar price, but they're a novel way to decompose option risk beyond the standard Greeks.

## Stack

Deno + TypeScript. No external dependencies beyond the Deno standard library.

```sh
# fetch data
deno run --allow-net --allow-write scripts/fetch-prices.ts

# run backtest
deno run --allow-read --allow-write scripts/backtest.ts
```
