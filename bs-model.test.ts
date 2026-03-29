/**
 * Unit tests — Quaternionic Black-Scholes Model
 * Run with: deno test bs-model.test.ts
 */

import { assertAlmostEquals, assertThrows } from "@std/assert";
import { type BSParams, greeks, impliedVol, normalCDF, normalPDF, price, quatN } from "./bs-model.ts";
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

// ---------------------------------------------------------------------------
// Greeks — classical recovery (real inputs)
//
// Benchmark: S=100, K=100, T=1, r=5%, σ=20%
//   d₁ = 0.35, d₂ = 0.15
//   Δ_call = N(0.35) ≈ 0.6368
//   Γ      = n(0.35)/(100·0.20·1) ≈ 0.01862
//   ν      = 100·n(0.35)·1 ≈ 37.524
//   Θ_call = −100·n(0.35)·0.20/2 − 0.05·100·e^{−0.05}·N(0.15)
//   ρ_call = 100·1·e^{−0.05}·N(0.15)
// ---------------------------------------------------------------------------

const d1t = 0.35;
const d2t = 0.15;

Deno.test("Greeks: delta.call.t = N(d₁), no imaginary leakage", () => {
	const { delta } = greeks(classic);
	near(delta.call.t, normalCDF(d1t), 1e-6);
	near(delta.call.p, 0);
	near(delta.call.f, 0);
	near(delta.call.l, 0);
});

Deno.test("Greeks: delta.put.t = N(d₁) − 1", () => {
	const { delta } = greeks(classic);
	near(delta.put.t, normalCDF(d1t) - 1, 1e-6);
});

Deno.test("Greeks: delta call + |delta put| = 1", () => {
	const { delta } = greeks(classic);
	near(delta.call.t - delta.put.t, 1);
});

Deno.test("Greeks: gamma = n(d₁)/(S·σ·√T)", () => {
	const { gamma } = greeks(classic);
	const expected = normalPDF(d1t) / (100 * 0.20 * 1);
	near(gamma, expected);
});

Deno.test("Greeks: vega.t = S·n(d₁)·√T", () => {
	const { vega } = greeks(classic);
	const expected = 100 * normalPDF(d1t) * 1;
	near(vega.t, expected);
	// Real inputs → no imaginary leakage
	near(vega.p, 0); near(vega.f, 0); near(vega.l, 0);
});

Deno.test("Greeks: vega same for call and put (put-call parity)", () => {
	// ∂C/∂σ − ∂P/∂σ = ∂/∂σ[S − K·e^{−rT}] = 0
	// So vega is identical for calls and puts — the single quaternion covers both.
	const { vega } = greeks(classic);
	near(vega.t, 100 * normalPDF(d1t));
});

Deno.test("Greeks: theta.call.t = classical formula", () => {
	const { theta } = greeks(classic);
	const discount = Math.exp(-0.05);
	const expected = -(100 * normalPDF(d1t) * 0.20) / 2 - 0.05 * 100 * discount * normalCDF(d2t);
	near(theta.call.t, expected, 1e-5);
});

Deno.test("Greeks: theta put-call parity Θ_put = Θ_call + r·K·e^{−rT}", () => {
	const { theta } = greeks(classic);
	const discount = Math.exp(-0.05);
	near(theta.put.t - theta.call.t, 0.05 * 100 * discount, 1e-6);
});

Deno.test("Greeks: rho.call.t = K·T·e^{−rT}·N(d₂)", () => {
	const { rho } = greeks(classic);
	const discount = Math.exp(-0.05);
	const expected = 100 * 1 * discount * normalCDF(d2t);
	near(rho.call.t, expected, 1e-5);
});

Deno.test("Greeks: rho put-call parity ρ_put = ρ_call − K·T·e^{−rT}", () => {
	const { rho } = greeks(classic);
	const discount = Math.exp(-0.05);
	near(rho.call.t - rho.put.t, 100 * 1 * discount, 1e-6);
});

// ---------------------------------------------------------------------------
// Greeks — quaternionic extension (non-zero imaginary inputs)
// ---------------------------------------------------------------------------

const quatParams: BSParams = {
	spot: { t: 100, p: 2.0, f: 1.0, l: 0 },
	strike: 100,
	expiry: 1,
	rate: 0.05,
	vol: { t: 0.20, p: 0.05, f: 0, l: 0 },
};

Deno.test("Greeks: quaternionic delta.call carries non-zero imaginary parts", () => {
	const { delta } = greeks(quatParams);
	// quatN propagates imaginary parts of d₁ through normalPDF
	assertAlmostEquals(Math.abs(delta.call.p) > 0 ? 1 : 0, 1, 1e-10);
});

Deno.test("Greeks: quaternionic vega imaginary parts match spot imaginary × n(d₁)", () => {
	const { vega } = greeks(quatParams);
	const { d1 } = price(quatParams);
	// vega = scale(spot, n(d₁.t)·√T)
	near(vega.p, quatParams.spot.p * normalPDF(d1.t));
	near(vega.f, quatParams.spot.f * normalPDF(d1.t));
});

Deno.test("Greeks: quaternionic gamma stays positive and real", () => {
	const { gamma } = greeks(quatParams);
	assertAlmostEquals(gamma > 0 ? 1 : 0, 1, 1e-10);
});

// ---------------------------------------------------------------------------
// Implied vol inversion
// ---------------------------------------------------------------------------

Deno.test("impliedVol: round-trip recovers classical vol", () => {
	// Price with σ=0.30, then invert
	const params: BSParams = { ...classic, vol: { t: 0.30, p: 0, f: 0, l: 0 } };
	const callPrice = price(params).call.t;
	const iv = impliedVol(callPrice, { ...classic, vol: { t: 0.20, p: 0, f: 0, l: 0 } });
	near(iv, 0.30, 1e-7);
});

Deno.test("impliedVol: round-trip from standard vol", () => {
	const callPrice = price(classic).call.t;
	const iv = impliedVol(callPrice, classic);
	near(iv, 0.20, 1e-7);
});

Deno.test("impliedVol: round-trip with OTM option", () => {
	const params: BSParams = {
		spot: { t: 100, p: 0, f: 0, l: 0 },
		strike: 110,
		expiry: 0.5,
		rate: 0.05,
		vol: { t: 0.25, p: 0, f: 0, l: 0 },
	};
	const callPrice = price(params).call.t;
	const iv = impliedVol(callPrice, { ...params, vol: { t: 0.20, p: 0, f: 0, l: 0 } });
	near(iv, 0.25, 1e-7);
});

Deno.test("impliedVol: round-trip with ITM option", () => {
	const params: BSParams = {
		spot: { t: 100, p: 0, f: 0, l: 0 },
		strike: 90,
		expiry: 0.25,
		rate: 0.02,
		vol: { t: 0.35, p: 0, f: 0, l: 0 },
	};
	const callPrice = price(params).call.t;
	const iv = impliedVol(callPrice, { ...params, vol: { t: 0.20, p: 0, f: 0, l: 0 } });
	near(iv, 0.35, 1e-7);
});

Deno.test("impliedVol: imaginary vol components held fixed during inversion", () => {
	// Only vol.t should move; vol.p/f/l stay as supplied
	const params: BSParams = { ...classic, vol: { t: 0.30, p: 0.05, f: 0.02, l: 0 } };
	const callPrice = price(params).call.t;
	const seed: BSParams = { ...classic, vol: { t: 0.20, p: 0.05, f: 0.02, l: 0 } };
	const iv = impliedVol(callPrice, seed);
	near(iv, 0.30, 1e-6);
});

Deno.test("impliedVol: throws on price below intrinsic", () => {
	assertThrows(
		() => impliedVol(-1, classic),
		Error,
		"no-arbitrage",
	);
});

Deno.test("impliedVol: throws on price above spot", () => {
	assertThrows(
		() => impliedVol(200, classic),
		Error,
		"no-arbitrage",
	);
});
