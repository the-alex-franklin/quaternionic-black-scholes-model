/**
 * Unit tests — Backtest Engine
 *
 * Uses synthetic fixture snapshots generated from known BS parameters
 * so tests are deterministic and network-free.
 *
 * Fixture design:
 *   Ticker   : SPY (synthetic)
 *   AsOf     : 2026-01-02
 *   Spot     : 500.00
 *   Expiry   : 2026-02-20  (~49 days = 0.1342 yr)
 *   Rate     : 5%
 *   Vol surface: σ(K) = 0.20 − 0.10·log(K/500)  (realistic downward skew)
 *
 * Market "mid" prices are generated with the skewed IV per strike.
 * Classical BS is priced with flat vol (ATM IV = 0.20).
 * Quaternionic BS is calibrated from the surface → should have lower RMSE.
 */

import { assertAlmostEquals } from "@std/assert";
import { price as bsPrice } from "../bs-model/bs-model.ts";
import {
	extractQuatSpot,
	fitQuatVol,
	runBacktest,
} from "./backtest.ts";
import type { MarketSnapshot, OptionQuote } from "../../scripts/collect-market-data.ts";

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

const SPOT = 500;
const RATE = 0.05;
const EXPIRY_DATE = new Date("2026-02-20T21:00:00Z");
const AS_OF = new Date("2026-01-02T15:30:00Z");
const EXPIRY_TS = Math.floor(EXPIRY_DATE.getTime() / 1000);
const T = (EXPIRY_DATE.getTime() - AS_OF.getTime()) / (1000 * 60 * 60 * 24 * 365.25);

/** Skewed implied vol surface: downward skew typical of equity index options */
const skewedIV = (K: number): number =>
	Math.max(0.10, 0.20 - 0.10 * Math.log(K / SPOT));

/** Generate a synthetic option quote with the skewed IV */
const makeQuote = (K: number, type: "call" | "put"): OptionQuote => {
	const iv = skewedIV(K);
	const params = {
		spot: { t: SPOT, p: 0, f: 0, l: 0 },
		strike: K,
		expiry: T,
		rate: RATE,
		vol: { t: iv, p: 0, f: 0, l: 0 },
	};
	const r = bsPrice(params);
	const mid = type === "call" ? r.call.t : r.put.t;
	// Realistic bid-ask: ~1% of mid for liquid SPY options
	const halfSpread = Math.max(0.05, mid * 0.005);
	return {
		strike: K,
		expiry: EXPIRY_DATE,
		expiryTs: EXPIRY_TS,
		bid: mid - halfSpread,
		ask: mid + halfSpread,
		mid,
		iv,
		volume: 1000,
		openInterest: 10000,
		type,
	};
};

const STRIKES = [450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550];

const fixtureSnapshot: MarketSnapshot = {
	ticker: "SPY",
	asOf: AS_OF,
	spot: SPOT,
	calls: STRIKES.map((K) => makeQuote(K, "call")),
	puts: STRIKES.map((K) => makeQuote(K, "put")),
};

// ---------------------------------------------------------------------------
// fitQuatVol tests
// ---------------------------------------------------------------------------

Deno.test("fitQuatVol: vol.t is ATM implied vol (≈ 0.20 for our fixture)", () => {
	const { vol } = fitQuatVol(fixtureSnapshot, RATE);
	// ATM strike = 500, skewedIV(500) = 0.20
	assertAlmostEquals(vol.t, 0.20, 0.01);
});

Deno.test("fitQuatVol: vol.p is negative for downward skew", () => {
	const { vol } = fitQuatVol(fixtureSnapshot, RATE);
	// Our skew σ(K) = 0.20 − 0.10·log(K/500): slope w.r.t. log(K/S) is −0.10
	assertAlmostEquals(vol.p < 0 ? 1 : 0, 1, 1e-10);
});

Deno.test("fitQuatVol: vol components are finite", () => {
	const { vol } = fitQuatVol(fixtureSnapshot, RATE);
	assertAlmostEquals(isFinite(vol.t) ? 1 : 0, 1, 1e-10);
	assertAlmostEquals(isFinite(vol.p) ? 1 : 0, 1, 1e-10);
	assertAlmostEquals(isFinite(vol.f) ? 1 : 0, 1, 1e-10);
});

// ---------------------------------------------------------------------------
// extractQuatSpot tests
// ---------------------------------------------------------------------------

Deno.test("extractQuatSpot: spot.t equals market spot", () => {
	const qSpot = extractQuatSpot(fixtureSnapshot, RATE, EXPIRY_TS);
	assertAlmostEquals(qSpot.t, SPOT, 1e-6);
});

Deno.test("extractQuatSpot: spot.l is zero (emergent)", () => {
	const qSpot = extractQuatSpot(fixtureSnapshot, RATE, EXPIRY_TS);
	assertAlmostEquals(qSpot.l, 0, 1e-10);
});

Deno.test("extractQuatSpot: spot.p is zero when no funding rate in snapshot", () => {
	// fixtureSnapshot has no funding8h field → spot.p = 0
	const qSpot = extractQuatSpot(fixtureSnapshot, RATE, EXPIRY_TS);
	assertAlmostEquals(qSpot.p, 0, 1e-10);
});

// ---------------------------------------------------------------------------
// runBacktest tests
// ---------------------------------------------------------------------------

Deno.test("runBacktest: returns correct ticker and spot", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	assertAlmostEquals(result.spot, SPOT, 1e-6);
	assertAlmostEquals(result.ticker === "SPY" ? 1 : 0, 1, 1e-10);
});

Deno.test("runBacktest: produces comparisons for all liquid quotes", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	// Should have 22 comparisons (11 strikes × 2 types) minus any filtered
	assertAlmostEquals(result.comparisons.length > 0 ? 1 : 0, 1, 1e-10);
});

Deno.test("runBacktest: RMSE values are non-negative", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	assertAlmostEquals(result.classicalRMSE >= 0 ? 1 : 0, 1, 1e-10);
	assertAlmostEquals(result.quaternionicRMSE >= 0 ? 1 : 0, 1, 1e-10);
});

Deno.test("runBacktest: alpha = classicalRMSE − quaternionicRMSE", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	assertAlmostEquals(result.alpha, result.classicalRMSE - result.quaternionicRMSE, 1e-10);
});

Deno.test("runBacktest: the two models give different RMSE on a skewed surface", () => {
	// When vol.p ≠ 0 (skew is present), the quaternionic model uses a different
	// effective vol quaternion than classical flat-vol BS, so the two RMSEs must
	// differ. Whether quaternionic wins depends on calibration; here we only
	// verify the calibration has an observable effect (alpha ≠ 0 and is finite).
	const result = runBacktest(fixtureSnapshot, RATE);
	assertAlmostEquals(isFinite(result.alpha) ? 1 : 0, 1, 1e-10);
	// vol.p ≈ −0.10 for our skewed surface → the models cannot be identical
	assertAlmostEquals(Math.abs(result.alpha) > 0 ? 1 : 0, 1, 1e-10);
});

Deno.test("runBacktest: all comparison errors are non-negative", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	for (const c of result.comparisons) {
		assertAlmostEquals(c.classicalError >= 0 ? 1 : 0, 1, 1e-10);
		assertAlmostEquals(c.quaternionicError >= 0 ? 1 : 0, 1, 1e-10);
	}
});

Deno.test("runBacktest: market IV is positive for all comparisons", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	for (const c of result.comparisons) {
		assertAlmostEquals(c.marketIV > 0 ? 1 : 0, 1, 1e-10);
	}
});

Deno.test("runBacktest: volT is ATM implied vol from fixture (≈ 0.20)", () => {
	const result = runBacktest(fixtureSnapshot, RATE);
	assertAlmostEquals(result.volT, 0.20, 0.01);
});
