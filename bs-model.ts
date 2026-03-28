/**
 * Quaternionic Black-Scholes Model
 *
 * Classical Black-Scholes (1973) operates in ℝ.
 * This module lifts it into ℍ (the quaternions), capturing four
 * market dimensions simultaneously:
 *
 *   t — real part         → standard log-price / option value
 *   p — i component       → funding-rate dimension
 *   f — j component       → liquidity dimension
 *   l — k component       → emergent cross-effect (funding × liquidity)
 *
 * Volatility and spot are quaternions; strike, expiry, and rate stay real.
 *
 * When all imaginary components are zero the formulas reduce exactly
 * to the classical 1973 Black-Scholes values.
 *
 * Pricing formula
 * ───────────────
 *   d₁_Q = [ log(S_Q) − log(K)  +  (r·1 + Σ²/2)·T ]  /  (|Σ| √T)
 *   d₂_Q = d₁_Q  −  Σ²·√T / |Σ|
 *   C_Q  = S_Q · N_Q(d₁)  −  K·e^{−rT} · N_Q(d₂)
 *   P_Q  = C_Q  −  S_Q  +  K·e^{−rT}·1
 *
 * where Σ² = Σ·Σ (quaternion product) and N_Q is the first-order
 * quaternionic extension of the normal CDF (see `quatN`).
 */

import {
	add,
	log as qlog,
	multiply,
	norm,
	type Quaternion,
	scale,
	subtract,
} from "./quaternion.ts";

// ---------------------------------------------------------------------------
// Normal distribution helpers
// ---------------------------------------------------------------------------

/**
 * Standard normal CDF.
 * Abramowitz & Stegun 26.2.17 — max absolute error ≈ 7.5 × 10⁻⁸.
 */
export const normalCDF = (x: number): number => {
	const t = 1 / (1 + 0.2316419 * Math.abs(x));
	const poly = t *
		(0.319381530 +
			t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
	const tail = (Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)) * poly;
	return x >= 0 ? 1 - tail : tail;
};

/** Standard normal PDF */
export const normalPDF = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/**
 * First-order quaternionic extension of N(·).
 *
 * For q = q.t + Im(q):
 *   N_Q(q) ≈ N(q.t) + n(q.t) · Im(q)
 *
 * The imaginary components of the result carry first-order sensitivities
 * of the cumulative probability to movements in the funding and liquidity
 * dimensions — they vanish when Im(q) = 0, recovering the scalar CDF.
 */
export const quatN = (q: Quaternion): Quaternion => {
	const n0 = normalCDF(q.t);
	const n1 = normalPDF(q.t);
	return { t: n0, p: n1 * q.p, f: n1 * q.f, l: n1 * q.l };
};

// ---------------------------------------------------------------------------
// Black-Scholes types
// ---------------------------------------------------------------------------

export type BSParams = {
	/**
	 * Quaternionic spot price.
	 *  t = market price
	 *  p = funding-rate-adjusted price component (i)
	 *  f = liquidity-adjusted price component (j)
	 *  l = emergent cross-term (k) — set to 0 to let the model generate it
	 */
	spot: Quaternion;

	/** Strike price (real-valued, > 0) */
	strike: number;

	/** Time to expiry in years */
	expiry: number;

	/** Continuously compounded risk-free rate */
	rate: number;

	/**
	 * Volatility quaternion.
	 *  t = base annualised vol  (σ in the classical model)
	 *  p = funding-rate vol component
	 *  f = liquidity vol component
	 *  l = cross / emergent vol
	 *
	 * |vol| must be > 0.
	 */
	vol: Quaternion;
};

export type BSResult = {
	/**
	 * Quaternionic d₁ and d₂.
	 *  Real parts recover the classical values.
	 *  Imaginary parts carry first-order corrections from funding and liquidity.
	 */
	d1: Quaternion;
	d2: Quaternion;

	/**
	 * Quaternionic option prices.
	 *  call.t / put.t  = classical BS call / put premium
	 *  Imaginary parts = first-order price sensitivities along each market dimension
	 */
	call: Quaternion;
	put: Quaternion;
};

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

/**
 * Quaternionic Black-Scholes pricing.
 *
 * Setting all imaginary components of `spot` and `vol` to zero yields
 * the exact classical 1973 Black-Scholes prices in the real components.
 */
export const price = (params: BSParams): BSResult => {
	const { spot, strike, expiry: T, rate: r, vol } = params;

	if (strike <= 0) throw new Error("BSParams: strike must be positive");
	if (T <= 0) throw new Error("BSParams: expiry must be positive");

	// Quaternionic log of the spot
	const logSpot = qlog(spot);

	// Log-moneyness: log(S_Q) − log(K)   [log(K) is real, subtracts from t only]
	const logMoneyness: Quaternion = { ...logSpot, t: logSpot.t - Math.log(strike) };

	// Σ² = vol · vol  (full quaternion product — captures cross-dimension interactions)
	const volSq = multiply(vol, vol);

	// Drift quaternion: r·1 + Σ²/2
	const drift = add({ t: r, p: 0, f: 0, l: 0 }, scale(volSq, 0.5));

	// Numerator of d₁: logMoneyness + drift·T
	const numer = add(logMoneyness, scale(drift, T));

	// Denominator of d₁ (real scalar): |Σ|·√T
	const volAbs = norm(vol);
	const denom = volAbs * Math.sqrt(T);
	if (denom === 0) throw new Error("BSParams: |vol|·√T must be non-zero");

	// d₁_Q and d₂_Q
	//   d₁_Q − d₂_Q = Σ²·√T / |Σ|   (recovers σ√T when vol is real)
	const d1 = scale(numer, 1 / denom);
	const d2 = subtract(d1, scale(volSq, Math.sqrt(T) / volAbs));

	// N_Q(d₁) and N_Q(d₂)
	const Nd1 = quatN(d1);
	const Nd2 = quatN(d2);

	// C_Q = S_Q · N_Q(d₁)  −  K·e^{−rT} · N_Q(d₂)
	const discount = Math.exp(-r * T);
	const call = subtract(multiply(spot, Nd1), scale(Nd2, strike * discount));

	// Quaternionic put-call parity: P_Q = C_Q − S_Q + K·e^{−rT}·1
	const put = add(subtract(call, spot), { t: strike * discount, p: 0, f: 0, l: 0 });

	return { d1, d2, call, put };
};
