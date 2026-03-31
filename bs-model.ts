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
 * Second-order quaternionic extension of N(·).
 *
 * For q = q_t + v  (v = p·i + f·j + l·k, purely imaginary):
 *
 *   N_Q(q) = [N(q_t) − N''(q_t)/2 · |v|²]
 *           + [n(q_t) − N'''(q_t)/6 · |v|²] · v
 *           + O(|v|⁴)
 *
 * Using N''(x) = −x·n(x)  and  N'''(x) = (x²−1)·n(x):
 *
 *   real part:       N(q_t) + x·n(x)/2 · |v|²
 *   imaginary scale: n(x) · [1 − (x²−1)/6 · |v|²]
 *
 * Reduces exactly to the scalar CDF when Im(q) = 0.
 * The first-order result is recovered when the |v|² correction terms are dropped.
 */
export const quatN = (q: Quaternion): Quaternion => {
	const n0 = normalCDF(q.t);
	const n1 = normalPDF(q.t);
	const vNormSq = q.p ** 2 + q.f ** 2 + q.l ** 2;
	// Real correction: −N''(t)/2 · |v|² = (t·n(t)/2) · |v|²
	const realCorrection = (q.t * n1 / 2) * vNormSq;
	// Imaginary scale correction: −N'''(t)/6 · |v|² = −(t²−1)·n(t)/6 · |v|²
	const imagScale = n1 * (1 - (q.t ** 2 - 1) / 6 * vNormSq);
	return {
		t: Math.max(0, Math.min(1, n0 + realCorrection)),
		p: imagScale * q.p,
		f: imagScale * q.f,
		l: imagScale * q.l,
	};
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
	const rawCall = subtract(multiply(spot, Nd1), scale(Nd2, strike * discount));

	// Clamp call.t to its no-arbitrage lower bound (intrinsic value).
	// The imaginary cross-terms in S_Q · N_Q(d₁) can shave a few dollars off
	// call.t, which is negligible on an ATM call but enough to push a nearly
	// worthless put negative via parity. Parity is then restored by recomputing
	// put from the clamped call rather than clamping both sides independently.
	const intrinsicCall = Math.max(spot.t - strike * discount, 0);
	const call = { ...rawCall, t: Math.max(rawCall.t, intrinsicCall) };

	// Quaternionic put-call parity: P_Q = C_Q − S_Q + K·e^{−rT}·1
	// Recomputed from the (possibly clamped) call so parity holds exactly.
	const put = add(subtract(call, spot), { t: strike * discount, p: 0, f: 0, l: 0 });

	return { d1, d2, call, put };
};

// ---------------------------------------------------------------------------
// Greeks
// ---------------------------------------------------------------------------

export type BSGreeks = {
	/**
	 * Delta — ∂C_Q/∂S_Q ≈ N_Q(d₁), ∂P_Q/∂S_Q = delta.call − 1
	 *
	 * Real part = classical Δ = N(d₁) for calls, N(d₁)−1 for puts.
	 * Imaginary parts = first-order Δ sensitivities to funding/liquidity dimensions.
	 */
	delta: { call: Quaternion; put: Quaternion };

	/**
	 * Gamma — ∂²C/∂S² (real scalar: second-order, same for call and put).
	 *
	 * Γ = n(d₁) / (S·|Σ|·√T)
	 *
	 * Lifted to a scalar because second-order quaternionic cross-terms would
	 * require the full 4×4 Hessian; the real part is the actionable value.
	 */
	gamma: number;

	/**
	 * Vega — ∂C_Q/∂|Σ| (same for call and put by put-call parity).
	 *
	 * Real part = classical ν = S·n(d₁)·√T.
	 * Imaginary parts propagate spot's imaginary components through n(d₁)·√T,
	 * reflecting how funding/liquidity-adjusted notional drives vol sensitivity.
	 */
	vega: Quaternion;

	/**
	 * Theta — ∂C_Q/∂(−T), time decay per year.
	 *
	 * Classical:  Θ_call = −S·n(d₁)·σ/(2√T) − r·K·e^{−rT}·N(d₂)
	 *             Θ_put  = Θ_call + r·K·e^{−rT}   (from parity)
	 */
	theta: { call: Quaternion; put: Quaternion };

	/**
	 * Rho — ∂C_Q/∂r.
	 *
	 * Classical:  ρ_call =  K·T·e^{−rT}·N(d₂)
	 *             ρ_put  = −K·T·e^{−rT}·N(1−d₂) = ρ_call − K·T·e^{−rT}
	 */
	rho: { call: Quaternion; put: Quaternion };
};

/**
 * Compute the five standard quaternionic Greeks.
 *
 * All real parts recover the classical 1973 Black-Scholes values exactly
 * when spot and vol are purely real.
 */
export const greeks = (params: BSParams): BSGreeks => {
	const { spot, strike, expiry: T, rate: r, vol } = params;
	const { d1, d2 } = price(params);

	const volAbs = norm(vol);
	const sqrtT = Math.sqrt(T);
	const discount = Math.exp(-r * T);
	const n1 = normalPDF(d1.t); // n(d₁.t)

	// --- delta ---
	// ∂C_Q/∂S_Q ≈ N_Q(d₁)  (leading term; the S-dependence inside d₁ cancels classically)
	const deltaCall = quatN(d1);
	const deltaPut = subtract(deltaCall, { t: 1, p: 0, f: 0, l: 0 });

	// --- gamma ---
	// Real scalar:  n(d₁) / (S·|Σ|·√T)
	const gamma = n1 / (spot.t * volAbs * sqrtT);

	// --- vega ---
	// ∂C_Q/∂|Σ| = S_Q · n(d₁) · √T
	// Imaginary parts: funding/liquidity notionals experience the same vol sensitivity
	const vega = scale(spot, n1 * sqrtT);

	// --- theta ---
	// Θ_call = −S_Q·n(d₁)·|Σ|/(2√T)  −  r·K·e^{−rT}·N_Q(d₂)
	const thetaDecay = scale(spot, -n1 * volAbs / (2 * sqrtT));
	const thetaCarry = scale(quatN(d2), -r * strike * discount);
	const thetaCall = add(thetaDecay, thetaCarry);
	// Θ_put = Θ_call + r·K·e^{−rT}  (put-call parity differentiated w.r.t. T)
	const thetaPut = add(thetaCall, { t: r * strike * discount, p: 0, f: 0, l: 0 });

	// --- rho ---
	// ρ_call =  K·T·e^{−rT}·N_Q(d₂)
	const rhoCall = scale(quatN(d2), strike * T * discount);
	// ρ_put  = ρ_call − K·T·e^{−rT}·1   (parity: ∂/∂r [K·e^{−rT}] = −K·T·e^{−rT})
	const rhoPut = subtract(rhoCall, { t: strike * T * discount, p: 0, f: 0, l: 0 });

	return {
		delta: { call: deltaCall, put: deltaPut },
		gamma,
		vega,
		theta: { call: thetaCall, put: thetaPut },
		rho: { call: rhoCall, put: rhoPut },
	};
};

// ---------------------------------------------------------------------------
// Implied volatility inversion
// ---------------------------------------------------------------------------

export type ImpliedVolOptions = {
	/**
	 * Convergence tolerance on |C(σ) − marketPrice|.
	 * Default: 1e-8.
	 */
	tol?: number;

	/**
	 * Maximum Newton-Raphson iterations.
	 * Default: 100.
	 */
	maxIter?: number;
};

/**
 * Implied volatility inversion via Newton-Raphson.
 *
 * Finds the real scalar σ* such that the real part of the quaternionic call
 * price equals `marketPrice`, holding all imaginary vol components fixed at
 * their values in `params.vol`.
 *
 * The initial guess is `params.vol.t`; pass a vol quaternion with a
 * reasonable `.t` to seed the search (e.g. 0.2 for equity options).
 *
 * Returns the calibrated real vol σ*.  To reconstruct the full quaternionic
 * vol use: `{ ...params.vol, t: impliedVol(...) }`.
 *
 * Throws if the market price lies outside the no-arbitrage bounds
 * [max(S − K·e^{−rT}, 0), S] or if Newton-Raphson fails to converge.
 */
export const impliedVol = (
	marketPrice: number,
	params: BSParams,
	opts: ImpliedVolOptions = {},
): number => {
	const { spot, strike, expiry: T, rate: r } = params;
	const { tol = 1e-8, maxIter = 100 } = opts;

	// No-arbitrage bounds on call price
	const discount = Math.exp(-r * T);
	const lowerBound = Math.max(spot.t - strike * discount, 0);
	const upperBound = spot.t;
	if (marketPrice < lowerBound - tol || marketPrice > upperBound + tol) {
		throw new Error(
			`impliedVol: market price ${marketPrice} outside no-arbitrage bounds [${
				lowerBound.toFixed(4)
			}, ${upperBound.toFixed(4)}]`,
		);
	}

	// Newton-Raphson on vol.t, imaginary components held fixed
	let sigma = params.vol.t;

	for (let i = 0; i < maxIter; i++) {
		const p: BSParams = { ...params, vol: { ...params.vol, t: sigma } };
		const callPrice = price(p).call.t;
		const vegaT = greeks(p).vega.t; // ∂call.t/∂σ

		const error = callPrice - marketPrice;
		if (Math.abs(error) < tol) return sigma;

		if (Math.abs(vegaT) < 1e-14) {
			throw new Error(`impliedVol: vega collapsed to zero at σ=${sigma}`);
		}

		// Newton step, clamped to keep σ positive
		sigma = Math.max(sigma - error / vegaT, 1e-8);
	}

	throw new Error(`impliedVol: failed to converge after ${maxIter} iterations`);
};
