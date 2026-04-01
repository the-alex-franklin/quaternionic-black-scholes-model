/**
 * Unit tests — Quaternion Math Layer
 * Run with: deno test quaternion.test.ts
 */

import { assertAlmostEquals, assertEquals } from "@std/assert";
import {
	add,
	conjugate,
	exp,
	inverse,
	log,
	multiply,
	norm,
	project,
	type Quaternion,
	scale,
	subtract,
} from "./quaternion.ts";

const q1: Quaternion = { t: 1, p: 2, f: 3, l: 4 };
const q2: Quaternion = { t: 5, p: 6, f: 7, l: 8 };

const near = (a: number, b: number) => assertAlmostEquals(a, b, 1e-10);
const qNear = (a: Quaternion, b: Quaternion) => {
	near(a.t, b.t);
	near(a.p, b.p);
	near(a.f, b.f);
	near(a.l, b.l);
};

Deno.test("add", () => {
	qNear(add(q1, q2), { t: 6, p: 8, f: 10, l: 12 });
});

Deno.test("subtract", () => {
	qNear(subtract(q1, q2), { t: -4, p: -4, f: -4, l: -4 });
});

Deno.test("scale", () => {
	qNear(scale(q1, 2), { t: 2, p: 4, f: 6, l: 8 });
});

Deno.test("conjugate", () => {
	qNear(conjugate(q1), { t: 1, p: -2, f: -3, l: -4 });
});

Deno.test("norm", () => {
	near(norm({ t: 1, p: 0, f: 0, l: 0 }), 1);
	near(norm({ t: 0, p: 1, f: 0, l: 0 }), 1);
	near(norm(q1), Math.sqrt(1 + 4 + 9 + 16));
});

// Core quaternion identities — these must hold or the mul table is wrong
Deno.test("i² = -1", () => {
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	qNear(multiply(i, i), { t: -1, p: 0, f: 0, l: 0 });
});

Deno.test("j² = -1", () => {
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	qNear(multiply(j, j), { t: -1, p: 0, f: 0, l: 0 });
});

Deno.test("k² = -1", () => {
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	qNear(multiply(k, k), { t: -1, p: 0, f: 0, l: 0 });
});

Deno.test("ij = k", () => {
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	qNear(multiply(i, j), { t: 0, p: 0, f: 0, l: 1 });
});

Deno.test("ji = -k", () => {
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	qNear(multiply(j, i), { t: 0, p: 0, f: 0, l: -1 });
});

Deno.test("jk = i", () => {
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	qNear(multiply(j, k), { t: 0, p: 1, f: 0, l: 0 });
});

Deno.test("kj = -i", () => {
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	qNear(multiply(k, j), { t: 0, p: -1, f: 0, l: 0 });
});

Deno.test("ki = j", () => {
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	qNear(multiply(k, i), { t: 0, p: 0, f: 1, l: 0 });
});

Deno.test("ik = -j", () => {
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	qNear(multiply(i, k), { t: 0, p: 0, f: -1, l: 0 });
});

Deno.test("ijk = -1", () => {
	const i: Quaternion = { t: 0, p: 1, f: 0, l: 0 };
	const j: Quaternion = { t: 0, p: 0, f: 1, l: 0 };
	const k: Quaternion = { t: 0, p: 0, f: 0, l: 1 };
	qNear(multiply(i, j, k), { t: -1, p: 0, f: 0, l: 0 });
});

Deno.test("q * inverse(q) = 1", () => {
	const inv = inverse(q1);
	const result = multiply(q1, inv);
	near(result.t, 1);
	near(result.p, 0);
	near(result.f, 0);
	near(result.l, 0);
});

Deno.test("project returns real component", () => {
	assertEquals(project(q1), 1);
});

// ---------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------

Deno.test("exp of zero = 1", () => {
	qNear(exp({ t: 0, p: 0, f: 0, l: 0 }), { t: 1, p: 0, f: 0, l: 0 });
});

Deno.test("exp of real = scalar exponential", () => {
	qNear(exp({ t: 1, p: 0, f: 0, l: 0 }), { t: Math.E, p: 0, f: 0, l: 0 });
});

Deno.test("exp(πi) = -1  (Euler for i)", () => {
	// exp(0 + π·i) = cos π + i·sin π = -1
	qNear(exp({ t: 0, p: Math.PI, f: 0, l: 0 }), { t: -1, p: 0, f: 0, l: 0 });
});

Deno.test("exp((π/2)·k) = k  (unit rotation about k)", () => {
	qNear(exp({ t: 0, p: 0, f: 0, l: Math.PI / 2 }), { t: 0, p: 0, f: 0, l: 1 });
});

Deno.test("|exp(q)| = e^q.t", () => {
	near(norm(exp(q1)), Math.exp(q1.t));
	near(norm(exp(q2)), Math.exp(q2.t));
});

// ---------------------------------------------------------------------------
// log
// ---------------------------------------------------------------------------

Deno.test("log of 1 = 0", () => {
	qNear(log({ t: 1, p: 0, f: 0, l: 0 }), { t: 0, p: 0, f: 0, l: 0 });
});

Deno.test("log(e) = 1", () => {
	qNear(log({ t: Math.E, p: 0, f: 0, l: 0 }), { t: 1, p: 0, f: 0, l: 0 });
});

Deno.test("exp(log(q)) ≈ q  (round-trip, pure-imaginary norm < π)", () => {
	// Use a quaternion whose pure part has norm well inside (0, π)
	const q: Quaternion = { t: 2, p: 0.4, f: 0.3, l: 0.1 };
	qNear(exp(log(q)), q);
});

Deno.test("log(exp(q)) ≈ q  (round-trip, pure-imaginary norm < π)", () => {
	const q: Quaternion = { t: 0.5, p: 0.2, f: -0.3, l: 0.1 };
	qNear(log(exp(q)), q);
});
