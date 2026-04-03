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
 *   vol.t   — ATM implied vol (quadratic OLS intercept at m=0)
 *   vol.p   — skew: dIV/d(log-moneyness) at ATM (quadratic OLS linear term)
 *   vol.f   — term structure: OLS slope of mean-IV vs √T
 *   vol.l   — curvature: d²IV/d(log-moneyness)² at ATM (2× quadratic OLS term)
 *
 *   spot.t  — market price of the underlying
 *   spot.p  — perpetual funding rate cost: funding_8h × spot (dollars)
 *   spot.f  — 0 (reserved)
 *   spot.l  — 0 (emergent)
 *
 * The hypothesis: quaternionic RMSE < classical RMSE because encoding
 * funding rate pressure and vol surface shape in the imaginary dimensions
 * captures cross-market effects that classical flat-vol BS ignores.
 */

import { type OptionQuote, type MarketSnapshot } from "../deribit/deribit.ts";
import { impliedVol as bsImpliedVol, price as bsPrice, type BSParams } from "../bs-model/bs-model.ts";
import type { Quaternion } from "../quaternion/quaternion.ts";

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
	volT: number;                 // ATM implied vol (vol.t)
	volP: number;                 // skew (vol.p)
	volF: number;                 // term structure (vol.f) — OLS vs √T
	volL: number;                 // curvature (vol.l)
	spotP: number;                // funding pressure: funding_8h × spot (spot.p)
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

/**
 * Fit IV(m) = a + b·m + c·m² via quadratic OLS (m = log-moneyness).
 *
 *   volT = a   — ATM implied vol (value at m = 0)
 *   volP = b   — skew slope at ATM (first derivative)
 *   volL = 2c  — smile curvature at ATM (second derivative)
 */
const fitVolCurve = (
	moneyness: number[],
	ivs: number[],
): { volT: number; volP: number; volL: number } => {
	const mean = ivs.reduce((s, v) => s + v, 0) / (ivs.length || 1);
	const fallback = { volT: mean, volP: 0, volL: 0 };
	if (moneyness.length < 3) return fallback;

	// Accumulate sums for the 3×3 normal equations: Σ[1, m, m²]ᵀ[1, m, m²] a = Σ[1, m, m²]ᵀ iv
	let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
	let y0 = 0, y1 = 0, y2 = 0;
	for (let i = 0; i < moneyness.length; i++) {
		const m = moneyness[i]!, iv = ivs[i]!;
		const m2 = m * m;
		s0 += 1; s1 += m; s2 += m2; s3 += m2 * m; s4 += m2 * m2;
		y0 += iv; y1 += m * iv; y2 += m2 * iv;
	}

	// Cramer's rule for the 3×3 system
	const det3 = (r: number[][]): number =>
		r[0]![0]! * (r[1]![1]! * r[2]![2]! - r[1]![2]! * r[2]![1]!) -
		r[0]![1]! * (r[1]![0]! * r[2]![2]! - r[1]![2]! * r[2]![0]!) +
		r[0]![2]! * (r[1]![0]! * r[2]![1]! - r[1]![1]! * r[2]![0]!);

	const A = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
	const D = det3(A);
	if (Math.abs(D) < 1e-12) return fallback;

	const rhs = [y0, y1, y2];
	const cramer = (col: number): number =>
		det3(A.map((row, i) => row.map((v, j) => (j === col ? rhs[i]! : v)))) / D;

	const a = cramer(0), b = cramer(1), c = cramer(2);
	if (!isFinite(a) || !isFinite(b) || !isFinite(c)) return fallback;
	return { volT: Math.max(0.01, a), volP: b, volL: 2 * c };
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
export const liquidQuotes = (quotes: OptionQuote[]): OptionQuote[] =>
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
 *   vol.t — ATM implied vol (quadratic OLS intercept at log-moneyness = 0)
 *   vol.p — skew: dIV/d(log-moneyness) at ATM (quadratic OLS linear term)
 *   vol.f — term-structure slope: OLS of mean-IV vs √T across expiries
 *   vol.l — smile curvature: d²IV/d(log-moneyness)² at ATM (2× quadratic term)
 *
 * vol.t/p/l are extracted by fitting IV(m) = a + b·m + c·m² via OLS.
 * vol.f uses linear OLS since it operates along a separate axis (√T).
 */
export const fitQuatVol = (
	snapshot: MarketSnapshot,
	rate: number,
): { vol: Quaternion; atm: number } => {
	const allQuotes = liquidQuotes([...snapshot.calls, ...snapshot.puts]);
	const S = snapshot.spot;

	// vol.t, vol.p, vol.l — quadratic OLS on IV vs log-moneyness
	const moneyness = allQuotes.map((q) => Math.log(q.strike / S));
	const ivs = allQuotes.map((q) => q.iv);
	const { volT, volP, volL } = fitVolCurve(moneyness, ivs);

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

	// Cap imaginary components: |v(vol)| ≤ MAX_IMAG_VOL_RATIO * vol.t
	// Prevents imaginary squares from reducing the Itô drift enough to depress
	// option prices across the board (vol².t = vol.t² − vol.p² − vol.f² − vol.l²).
	const MAX_IMAG_VOL_RATIO = 0.20;
	const imagNorm = Math.sqrt(volP ** 2 + volF ** 2 + volL ** 2);
	const capFactor =
		imagNorm > MAX_IMAG_VOL_RATIO * volT
			? (MAX_IMAG_VOL_RATIO * volT) / imagNorm
			: 1;

	const vol: Quaternion = {
		t: volT,
		p: volP * capFactor,
		f: volF * capFactor,
		l: volL * capFactor,
	};
	return { vol, atm: volT };
};

/**
 * Extract quaternionic spot from market data.
 *
 *   spot.t — market price
 *   spot.p — perpetual funding rate cost in dollars (funding_8h × spot)
 *   spot.f — 0 (reserved; bid-ask proxy was too noisy to be useful)
 *   spot.l — 0 (emergent)
 */
export const extractQuatSpot = (
	snapshot: MarketSnapshot,
	_rate: number,
	_expiryTs: number,
): Quaternion => {
	const S = snapshot.spot;

	// Funding pressure: actual 8h perp funding rate × spot, in dollars.
	// Positive = longs pay shorts (contango); negative = shorts pay longs (backwardation).
	// Directly measured from the perpetual swap; no option-chain inference needed.
	const spotP = (snapshot.funding8h ?? 0) * S;

	return { t: S, p: spotP, f: 0, l: 0 };
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
		comparisons,
		classicalRMSE,
		quaternionicRMSE,
		alpha: classicalRMSE - quaternionicRMSE,
	};
};
