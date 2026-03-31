/**
 * Backtest Engine
 *
 * Pure functions — no network I/O.  Feed in a MarketSnapshot from
 * market-data.ts and get back a structured comparison of classical BS
 * vs quaternionic BS pricing errors.
 *
 * Calibration strategy
 * ────────────────────
 * Four market-observable quantities map cleanly to the four quaternion
 * dimensions of vol and spot:
 *
 *   vol.t   — ATM implied vol (classical σ)
 *   vol.p   — vol skew: slope of IV vs log-moneyness (funding vol)
 *   vol.f   — vol term structure: slope of IV vs T^{1/2} (liquidity vol)
 *   vol.l   — 0 (emergent; let the model generate it via Σ²)
 *
 *   spot.t  — market price of the underlying
 *   spot.p  — put–call parity residual / spot (funding pressure)
 *   spot.f  — mean normalised bid–ask spread × spot (liquidity pressure)
 *   spot.l  — 0 (emergent)
 *
 * The hypothesis: quaternionic RMSE < classical RMSE because vol.p/vol.f
 * capture the implied vol smile that classical flat-vol BS ignores.
 */

import { type OptionQuote, type MarketSnapshot } from "./market-data.ts";
import { impliedVol as bsImpliedVol, price as bsPrice, type BSParams } from "./bs-model.ts";
import type { Quaternion } from "./quaternion.ts";
import { extractVolCurve } from "./fft.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PricingComparison = {
	strike: number;
	expiry: Date;
	type: "call" | "put";
	marketMid: number;
	classicalPrice: number;
	classicalError: number;       // |classical − market|
	quaternionicPrice: number;
	quaternionicError: number;    // |quaternionic − market|
	marketIV: number;             // implied vol from market mid price
};

export type BacktestResult = {
	ticker: string;
	asOf: Date;
	spot: number;
	rate: number;
	volT: number;                 // ATM implied vol (vol.t) — DFT DC mode
	volP: number;                 // skew (vol.p) — DFT first-mode derivative
	volF: number;                 // term structure (vol.f) — OLS vs √T
	volL: number;                 // curvature (vol.l) — DFT second-mode
	spotP: number;                // funding pressure (spot.p)
	spotF: number;                // liquidity pressure (spot.f)
	comparisons: PricingComparison[];
	classicalRMSE: number;
	quaternionicRMSE: number;
	/** classicalRMSE − quaternionicRMSE: positive = quaternionic wins */
	alpha: number;
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Simple OLS: returns { slope, intercept } for y ~ slope*x + intercept */
const ols = (xs: number[], ys: number[]): { slope: number; intercept: number } => {
	const n = xs.length;
	if (n < 2) return { slope: 0, intercept: ys[0] ?? 0 };
	const mx = xs.reduce((s, x) => s + x, 0) / n;
	const my = ys.reduce((s, y) => s + y, 0) / n;
	const num = xs.reduce((s, x, i) => s + (x - mx) * ((ys[i] ?? my) - my), 0);
	const den = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
	const slope = den === 0 ? 0 : num / den;
	return { slope, intercept: my - slope * mx };
};

const rmse = (errors: number[]): number =>
	Math.sqrt(errors.reduce((s, e) => s + e ** 2, 0) / errors.length);

/** Fraction of time to expiry remaining in years */
const timeToExpiry = (asOf: Date, expiry: Date): number =>
	Math.max((expiry.getTime() - asOf.getTime()) / (1000 * 60 * 60 * 24 * 365.25), 1 / 365);

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/** Filter options that are liquid enough to trust */
const liquidQuotes = (quotes: OptionQuote[]): OptionQuote[] =>
	quotes.filter((q) =>
		q.bid > 0 &&
		q.ask > 0 &&
		q.mid > 0 &&
		q.ask > q.bid &&
		q.iv > 0.01 && q.iv < 5 &&    // sanity bounds on IV
		(q.volume > 0 || q.openInterest > 0)
	);

/**
 * Fit quaternionic vol from the options surface.
 *
 *   vol.t — smoothed ATM implied vol (DFT DC mode at log-moneyness = 0)
 *   vol.p — skew: dIV/d(log-moneyness) at ATM (DFT first-mode derivative)
 *   vol.f — term-structure slope: OLS of mean-IV vs √T across expiries
 *   vol.l — smile curvature: d²IV/d(log-moneyness)² at ATM (DFT second-mode)
 *
 * vol.t/p/l are extracted via DFT-based low-pass filtering of the IV curve,
 * which separates the smooth smile signal from noise in illiquid/stale quotes.
 * vol.f is kept on OLS since it operates along a separate axis (√T).
 */
export const fitQuatVol = (
	snapshot: MarketSnapshot,
	rate: number,
): { vol: Quaternion; atm: number } => {
	const allQuotes = liquidQuotes([...snapshot.calls, ...snapshot.puts]);
	const S = snapshot.spot;

	// vol.t, vol.p, vol.l — DFT of IV vs log-moneyness
	const moneyness = allQuotes.map((q) => Math.log(q.strike / S));
	const ivs = allQuotes.map((q) => q.iv);
	const { volT, volP, volL } = extractVolCurve(moneyness, ivs);

	// vol.f — OLS of mean IV vs √T across expiry groups
	const expiryGroups = new Map<number, number[]>();
	for (const q of allQuotes) {
		const t = timeToExpiry(snapshot.asOf, q.expiry);
		if (!expiryGroups.has(t)) expiryGroups.set(t, []);
		expiryGroups.get(t)!.push(q.iv);
	}
	const tsPoints: Array<[number, number]> = [...expiryGroups.entries()].map(
		([t, ivList]) => [Math.sqrt(t), ivList.reduce((s, x) => s + x, 0) / ivList.length] as [number, number],
	);
	const { slope: volF } = ols(tsPoints.map(([t]) => t), tsPoints.map(([, iv]) => iv));

	const vol: Quaternion = { t: volT, p: volP, f: volF, l: volL };
	return { vol, atm: volT };
};

/**
 * Extract quaternionic spot from market data.
 *
 *   spot.t — market price
 *   spot.p — put–call parity residual / S (funding pressure)
 *   spot.f — mean normalised bid–ask spread × S (liquidity pressure)
 *   spot.l — 0 (emergent)
 */
export const extractQuatSpot = (
	snapshot: MarketSnapshot,
	rate: number,
	expiryTs: number,
): Quaternion => {
	const S = snapshot.spot;
	const calls = liquidQuotes(snapshot.calls.filter((q) => q.expiryTs === expiryTs));
	const puts = liquidQuotes(snapshot.puts.filter((q) => q.expiryTs === expiryTs));
	const T = timeToExpiry(snapshot.asOf, new Date(expiryTs * 1000));
	const discount = Math.exp(-rate * T);

	// ATM put–call parity residual
	const atmStrike = calls.reduce(
		(best, q) => Math.abs(q.strike - S) < Math.abs(best.strike - S) ? q : best,
		calls[0] ?? { strike: S } as OptionQuote,
	).strike;
	const atmCall = calls.find((q) => q.strike === atmStrike);
	const atmPut = puts.find((q) => q.strike === atmStrike);
	const spotP = (atmCall && atmPut)
		? (atmCall.mid - atmPut.mid - S + atmStrike * discount) / S
		: 0;

	// Liquidity pressure: mean normalised bid–ask spread of near-ATM options × S.
	// We restrict to within ±10% of spot because deep OTM spreads are wide by
	// construction and tell us nothing about underlying spot liquidity.
	// Also capped at 3% of spot: quatN is a second-order Taylor expansion valid
	// only for small imaginary perturbations (|v(d1)| < ~0.5), which requires
	// the imaginary components of spot to stay below ~3% of the real part.
	const atmLiquid = liquidQuotes([...calls, ...puts]).filter(
		(q) => Math.abs(q.strike / S - 1) < 0.10,
	);
	const meanSpread = atmLiquid.length > 0
		? atmLiquid.reduce((s, q) => s + Math.min((q.ask - q.bid) / q.mid, 1.0), 0) / atmLiquid.length
		: 0;
	const spotF = Math.min(meanSpread * S, 0.03 * S);

	return { t: S, p: spotP * S, f: spotF, l: 0 };
};

// ---------------------------------------------------------------------------
// Backtest runner
// ---------------------------------------------------------------------------

/**
 * Compare classical vs quaternionic BS pricing across an options snapshot.
 *
 * Only options with liquid markets (positive bid/ask, sane IV) are included.
 * Market mid price is used as the reference; bid-ask midpoint is standard.
 *
 * @param snapshot  Live or historical snapshot from fetchOptionsChain()
 * @param rate      Risk-free rate (e.g. 0.05 for 5%)
 */
export const runBacktest = (snapshot: MarketSnapshot, rate: number): BacktestResult => {
	const S = snapshot.spot;
	const { vol, atm: volT } = fitQuatVol(snapshot, rate);

	// Pick the nearest expiry with at least MIN_DTE days to go.
	// Very short-dated options have tiny sqrt(T) denominators that amplify the
	// imaginary components of d1 beyond the Taylor regime of quatN.
	const MIN_DTE = 7;
	const minTs = snapshot.asOf.getTime() / 1000 + MIN_DTE * 24 * 60 * 60;
	const firstExpiry = [...snapshot.calls, ...snapshot.puts]
		.map((q) => q.expiryTs)
		.filter((ts) => ts >= minTs)
		.sort((a, b) => a - b)[0];

	const firstExpiryDate = firstExpiry
		? new Date(firstExpiry * 1000)
		: new Date(snapshot.asOf.getTime() + 30 * 24 * 60 * 60 * 1000);
	const T = timeToExpiry(snapshot.asOf, firstExpiryDate);
	const discount = Math.exp(-rate * T);

	const quatSpot = firstExpiry
		? extractQuatSpot(snapshot, rate, firstExpiry)
		: { t: S, p: 0, f: 0, l: 0 };

	const realSpot: Quaternion = { t: S, p: 0, f: 0, l: 0 };
	const realVol: Quaternion = { t: vol.t, p: 0, f: 0, l: 0 };

	const allQuotes = liquidQuotes([
		...snapshot.calls.filter((q) => q.expiryTs === firstExpiry),
		...snapshot.puts.filter((q) => q.expiryTs === firstExpiry),
	]);

	const comparisons: PricingComparison[] = [];

	for (const q of allQuotes) {
		// Skip if market mid is too small to invert reliably
		if (q.mid < 0.01) continue;

		const params: BSParams = {
			spot: realSpot,
			strike: q.strike,
			expiry: T,
			rate,
			vol: realVol,
		};

		// Classical price
		let classicalPrice = 0;
		try {
			const cf = bsPrice(params);
			classicalPrice = q.type === "call" ? cf.call.t : cf.put.t;
		} catch {
			continue;
		}

		// Quaternionic price
		const quatParams: BSParams = {
			...params,
			spot: quatSpot,
			vol,
		};
		let quaternionicPrice = 0;
		try {
			const qf = bsPrice(quatParams);
			quaternionicPrice = q.type === "call" ? qf.call.t : qf.put.t;
		} catch {
			continue;
		}

		// Market implied vol (for reference)
		let marketIV = q.iv;
		try {
			marketIV = bsImpliedVol(q.mid, { ...params, vol: { t: vol.t, p: 0, f: 0, l: 0 } });
		} catch {
			// fall back to Yahoo's IV estimate
		}

		comparisons.push({
			strike: q.strike,
			expiry: q.expiry,
			type: q.type,
			marketMid: q.mid,
			classicalPrice,
			classicalError: Math.abs(classicalPrice - q.mid),
			quaternionicPrice,
			quaternionicError: Math.abs(quaternionicPrice - q.mid),
			marketIV,
		});
	}

	const classicalRMSE = comparisons.length > 0
		? rmse(comparisons.map((c) => c.classicalError))
		: 0;
	const quaternionicRMSE = comparisons.length > 0
		? rmse(comparisons.map((c) => c.quaternionicError))
		: 0;

	return {
		ticker: snapshot.ticker,
		asOf: snapshot.asOf,
		spot: S,
		rate,
		volT,
		volP: vol.p,
		volF: vol.f,
		volL: vol.l,
		spotP: quatSpot.p,
		spotF: quatSpot.f,
		comparisons,
		classicalRMSE,
		quaternionicRMSE,
		alpha: classicalRMSE - quaternionicRMSE,
	};
};
