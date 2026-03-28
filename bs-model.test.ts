/**
 * Unit tests — Quaternionic Black-Scholes Model
 * Run with: deno test bs-model.test.ts
 */

import { assertAlmostEquals, assertThrows } from "@std/assert";
import { type BSParams, normalCDF, normalPDF, price, quatN } from "./bs-model.ts";
import type { Quaternion } from "./quaternion.ts";

const near = (a: number, b: number, eps = 1e-6) => assertAlmostEquals(a, b, eps);

const qNear = (a: Quaternion, b: Quaternion, eps = 1e-6) => {
	near(a.t, b.t, eps);
	near(a.p, b.p, eps);
	near(a.f, b.f, eps);
	near(a.l, b.l, eps);
};

// ---------------------------------------------------------------------------
// normalCDF
// ---------------------------------------------------------------------------

Deno.test("normalCDF(0) = 0.5", () => {
	near(normalCDF(0), 0.5);
});

Deno.test("normalCDF symmetry: N(−x) = 1 − N(x)", () => {
	for (const x of [0.1, 0.5, 1, 1.96, 2.5, 3]) {
		near(normalCDF(-x), 1 - normalCDF(x));
	}
});

Deno.test("normalCDF(1.96) ≈ 0.97500", () => {
	near(normalCDF(1.96), 0.97500, 1e-4);
});

Deno.test("normalCDF(−1.645) ≈ 0.05000", () => {
	near(normalCDF(-1.645), 0.05, 2e-4);
});

// ---------------------------------------------------------------------------
// normalPDF
// ---------------------------------------------------------------------------

Deno.test("normalPDF(0) = 1/√(2π)", () => {
	near(normalPDF(0), 1 / Math.sqrt(2 * Math.PI));
});

// ---------------------------------------------------------------------------
// quatN
// ---------------------------------------------------------------------------

Deno.test("quatN with zero imaginary part recovers scalar CDF", () => {
	const q: Quaternion = { t: 0.5, p: 0, f: 0, l: 0 };
	const result = quatN(q);
	near(result.t, normalCDF(0.5));
	near(result.p, 0);
	near(result.f, 0);
	near(result.l, 0);
});

Deno.test("quatN imaginary parts scale with normalPDF", () => {
	const q: Quaternion = { t: 0.35, p: 0.1, f: 0.2, l: 0 };
	const result = quatN(q);
	near(result.t, normalCDF(0.35));
	near(result.p, normalPDF(0.35) * 0.1);
	near(result.f, normalPDF(0.35) * 0.2);
	near(result.l, 0);
});

// ---------------------------------------------------------------------------
// Classical Black-Scholes recovery
//
// Benchmark: S=100, K=100, T=1yr, r=5%, σ=20%
//   d₁ = (0 + 0.07) / 0.20 = 0.35
//   d₂ = 0.35 - 0.20       = 0.15
//   C  ≈ 10.4506
// ---------------------------------------------------------------------------

const classic: BSParams = {
	spot: { t: 100, p: 0, f: 0, l: 0 },
	strike: 100,
	expiry: 1,
	rate: 0.05,
	vol: { t: 0.20, p: 0, f: 0, l: 0 },
};

Deno.test("classical BS: d₁ = 0.35, d₂ = 0.15", () => {
	const { d1, d2 } = price(classic);
	near(d1.t, 0.35, 1e-6);
	near(d2.t, 0.15, 1e-6);
	// No imaginary leakage when inputs are real
	near(d1.p, 0);
	near(d1.f, 0);
	near(d1.l, 0);
	near(d2.p, 0);
	near(d2.f, 0);
	near(d2.l, 0);
});

Deno.test("classical BS: call price ≈ 10.4506", () => {
	const { call } = price(classic);
	near(call.t, 10.4506, 1e-3);
	near(call.p, 0);
	near(call.f, 0);
	near(call.l, 0);
});

Deno.test("classical BS: put-call parity (real)", () => {
	const { call, put } = price(classic);
	const parity = classic.spot.t - classic.strike * Math.exp(-classic.rate * classic.expiry);
	near(call.t - put.t, parity);
});

Deno.test("classical BS: put price ≈ 5.5735 (via parity)", () => {
	const { put } = price(classic);
	// P = C − S + K e^{−rT} ≈ 10.4506 − 100 + 100·e^{−0.05} ≈ 5.57
	near(put.t, 10.4506 - 100 + 100 * Math.exp(-0.05), 1e-3);
});

// ---------------------------------------------------------------------------
// Deep ITM / OTM edge cases (classical)
// ---------------------------------------------------------------------------

Deno.test("deep ITM call approaches intrinsic value S − K·e^{−rT}", () => {
	const params: BSParams = { ...classic, spot: { t: 200, p: 0, f: 0, l: 0 } };
	const { call } = price(params);
	const intrinsic = 200 - 100 * Math.exp(-0.05);
	near(call.t, intrinsic, 1);
});

Deno.test("deep OTM call approaches zero", () => {
	const params: BSParams = { ...classic, spot: { t: 10, p: 0, f: 0, l: 0 } };
	const { call } = price(params);
	near(call.t, 0, 0.01);
});

// ---------------------------------------------------------------------------
// Quaternionic extension — non-zero imaginary inputs
// ---------------------------------------------------------------------------

Deno.test("funding-vol perturbation: real component stays near classical", () => {
	// Adding a small funding-rate vol component (5%) shouldn't move the classical
	// call price by more than a few ticks in the real dimension.
	const params: BSParams = { ...classic, vol: { t: 0.20, p: 0.05, f: 0, l: 0 } };
	const { call } = price(params);
	near(call.t, 10.4506, 1); // within $1
});

Deno.test("funding-vol perturbation: p-dimension of call is non-zero", () => {
	const params: BSParams = { ...classic, vol: { t: 0.20, p: 0.05, f: 0, l: 0 } };
	const { call } = price(params);
	// The funding-rate dimension must carry a non-trivial sensitivity
	assertAlmostEquals(
		Math.abs(call.p) > 0 ? 1 : 0,
		1,
		1e-10,
		"expected |call.p| > 0 for non-zero funding vol",
	);
});

Deno.test("quaternionic put-call parity holds in all four dimensions", () => {
	// C_Q − P_Q = S_Q − K·e^{−rT}·1
	const params: BSParams = {
		spot: { t: 100, p: 2.0, f: 1.0, l: 0.5 },
		strike: 100,
		expiry: 1,
		rate: 0.05,
		vol: { t: 0.20, p: 0.05, f: 0.03, l: 0 },
	};
	const { call, put } = price(params);
	const discount = Math.exp(-params.rate * params.expiry);
	const diff = {
		t: call.t - put.t,
		p: call.p - put.p,
		f: call.f - put.f,
		l: call.l - put.l,
	};
	// By construction of P_Q = C_Q − S_Q + K·e^{−rT}·1
	qNear(diff, {
		t: params.spot.t - params.strike * discount,
		p: params.spot.p,
		f: params.spot.f,
		l: params.spot.l,
	});
});

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

Deno.test("throws on non-positive strike", () => {
	assertThrows(() => price({ ...classic, strike: 0 }), Error, "strike");
});

Deno.test("throws on non-positive expiry", () => {
	assertThrows(() => price({ ...classic, expiry: 0 }), Error, "expiry");
});

Deno.test("throws on zero vol", () => {
	assertThrows(
		() => price({ ...classic, vol: { t: 0, p: 0, f: 0, l: 0 } }),
		Error,
	);
});
