/**
 * Quaternionic Monte Carlo Pricer
 *
 * Simulates the quaternionic GBM exact solution:
 *
 *   S_T^Q = S_0^Q · exp_Q[ (r·1 − Σ²/2)·T  +  Σ·√T·Z ],   Z ~ N(0,1)
 *
 * — The Itô correction −Σ²/2 is the quaternion product, not a scalar.
 * — exp_Q is the quaternion exponential (cross-dimension interactions).
 * — When all imaginary components are zero this is exactly classical GBM
 *   and the MC price converges to the classical Black-Scholes formula.
 *
 * The call payoff is projected to ℝ via the real component:
 *   payoff = max(S_T^Q.t − K, 0)
 *
 * Purpose: empirically validate the closed-form price() formula.
 * If both agree within statistical error, the model is internally consistent.
 */

import { add, exp as qexp, multiply, scale, subtract } from "./quaternion.ts";
import { price as bsPrice, type BSParams } from "./bs-model.ts";

// ---------------------------------------------------------------------------
// Seeded PRNG (Mulberry32)
// ---------------------------------------------------------------------------

/**
 * Mulberry32 — simple high-quality 32-bit seeded PRNG.
 * Returns a factory function producing uniform [0, 1) values.
 * Use this in tests for fully reproducible paths.
 */
export const mulberry32 = (seed: number): (() => number) => {
	let s = seed | 0;
	return () => {
		s = (s + 0x6D2B79F5) | 0;
		let t = Math.imul(s ^ (s >>> 15), 1 | s);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
};

// ---------------------------------------------------------------------------
// Box-Muller normal sampler
// ---------------------------------------------------------------------------

/** Polar form Box-Muller: two uniforms → one standard normal (avoids log(0)) */
const stdNormal = (rng: () => number): number => {
	let u: number, v: number, s: number;
	do {
		u = rng() * 2 - 1;
		v = rng() * 2 - 1;
		s = u * u + v * v;
	} while (s >= 1 || s === 0);
	return u * Math.sqrt(-2 * Math.log(s) / s);
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MCOptions = {
	/** Number of simulated paths. Default: 100_000. */
	paths?: number;
	/**
	 * Uniform [0,1) RNG. Defaults to Math.random.
	 * Pass mulberry32(seed) for reproducible results.
	 */
	rng?: () => number;
};

export type MCResult = {
	/** MC estimate of the call price (real component, discounted) */
	call: number;
	/** MC estimate of the put price (real component, discounted) */
	put: number;
	/** Standard error of the call estimate (1σ) */
	callStdErr: number;
	/** Standard error of the put estimate (1σ) */
	putStdErr: number;
	/** Number of paths actually simulated */
	paths: number;
	/** Closed-form prices for comparison */
	closedForm: { call: number; put: number };
};

// ---------------------------------------------------------------------------
// Pricer
// ---------------------------------------------------------------------------

/**
 * Quaternionic Monte Carlo pricer.
 *
 * Returns MC call/put estimates alongside their standard errors and the
 * closed-form prices.  Check convergence with:
 *
 *   Math.abs(result.call - result.closedForm.call) < 3 * result.callStdErr
 */
export const monteCarlo = (params: BSParams, opts: MCOptions = {}): MCResult => {
	const { spot, strike, expiry: T, rate: r, vol } = params;
	const paths = opts.paths ?? 100_000;
	const rng = opts.rng ?? Math.random;

	const discount = Math.exp(-r * T);
	const sqrtT = Math.sqrt(T);

	// Precompute drift quaternion: r·1 − Σ²/2  (Itô correction)
	const volSq = multiply(vol, vol);
	const drift = subtract({ t: r, p: 0, f: 0, l: 0 }, scale(volSq, 0.5));

	let sumCall = 0;
	let sumCall2 = 0;
	let sumPut = 0;
	let sumPut2 = 0;

	for (let i = 0; i < paths; i++) {
		const Z = stdNormal(rng);

		// exp_Q argument: drift·T + Σ·√T·Z
		const arg = add(scale(drift, T), scale(vol, sqrtT * Z));

		// Terminal quaternionic spot: S_0^Q · exp_Q(arg)
		const sT = multiply(spot, qexp(arg));

		// Project real component for payoff
		const callPayoff = Math.max(sT.t - strike, 0);
		const putPayoff = Math.max(strike - sT.t, 0);

		sumCall += callPayoff;
		sumCall2 += callPayoff * callPayoff;
		sumPut += putPayoff;
		sumPut2 += putPayoff * putPayoff;
	}

	const callMean = sumCall / paths;
	const putMean = sumPut / paths;

	// Unbiased variance estimate, SE = sqrt(var / N)
	const callVar = sumCall2 / paths - callMean * callMean;
	const putVar = sumPut2 / paths - putMean * putMean;

	const { call, put } = bsPrice(params);

	return {
		call: discount * callMean,
		put: discount * putMean,
		callStdErr: discount * Math.sqrt(callVar / paths),
		putStdErr: discount * Math.sqrt(putVar / paths),
		paths,
		closedForm: { call: call.t, put: put.t },
	};
};
