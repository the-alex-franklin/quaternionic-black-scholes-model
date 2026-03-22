/**
 * Quaternion Math Layer
 *
 * Q = t + p·i + f·j + l·k
 *
 *   t — real (time)
 *   p — price (i)
 *   f — funding rate (j)
 *   l — liquidity pressure (k, emergent)
 *
 * Multiplication rules:
 *   i² = j² = k² = ijk = -1
 *   ij =  k,  ji = -k
 *   jk =  i,  kj = -i
 *   ki =  j,  ik = -j
 */

export type Quaternion = {
	t: number; // real
	p: number; // i
	f: number; // j
	l: number; // k
};

// ---------------------------------------------------------------------------
// Multiplication table
// Key: "xy" → { component: "t"|"p"|"f"|"l", sign: 1|-1 }
// Encodes what basis element xy produces and its sign
// ---------------------------------------------------------------------------
type BasisComponent = "t" | "p" | "f" | "l";

const MUL_TABLE: Record<string, { component: BasisComponent; sign: 1 | -1 }> = {
	tt: { component: "t", sign: 1 },
	tp: { component: "p", sign: 1 },
	tf: { component: "f", sign: 1 },
	tl: { component: "l", sign: 1 },

	pt: { component: "p", sign: 1 },
	pp: { component: "t", sign: -1 }, // i² = -1
	pf: { component: "l", sign: 1 }, // ij =  k
	pl: { component: "f", sign: -1 }, // ik = -j

	ft: { component: "f", sign: 1 },
	fp: { component: "l", sign: -1 }, // ji = -k
	ff: { component: "t", sign: -1 }, // j² = -1
	fl: { component: "p", sign: 1 }, // jk =  i

	lt: { component: "l", sign: 1 },
	lp: { component: "f", sign: 1 }, // ki =  j
	lf: { component: "p", sign: -1 }, // kj = -i
	ll: { component: "t", sign: -1 }, // k² = -1
};

const BASIS: BasisComponent[] = ["t", "p", "f", "l"];
const BASIS_KEYS: Record<BasisComponent, string> = { t: "t", p: "p", f: "f", l: "l" };

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

export const add = (a: Quaternion, b: Quaternion): Quaternion => ({
	t: a.t + b.t,
	p: a.p + b.p,
	f: a.f + b.f,
	l: a.l + b.l,
});

export const subtract = (a: Quaternion, b: Quaternion): Quaternion => ({
	t: a.t - b.t,
	p: a.p - b.p,
	f: a.f - b.f,
	l: a.l - b.l,
});

export const scale = (a: Quaternion, s: number): Quaternion => ({
	t: a.t * s,
	p: a.p * s,
	f: a.f * s,
	l: a.l * s,
});

/**
 * Multiply two quaternions using the precomputed table.
 * Each pair of basis components maps to a result component + sign.
 */
export const multiply = (...qs: Quaternion[]): Quaternion => {
	if (qs.length < 2) throw new Error("multiply requires at least 2 quaternions");
	return qs.reduce((acc, q) => {
		const result: Quaternion = { t: 0, p: 0, f: 0, l: 0 };
		for (const ai of BASIS) {
			for (const bi of BASIS) {
				const key = `${BASIS_KEYS[ai]}${BASIS_KEYS[bi]}`;
				const entry = MUL_TABLE[key];
				if (!entry) throw new Error("No entry found!");
				result[entry.component] += acc[ai] * q[bi] * entry.sign;
			}
		}
		return result;
	});
};
/** Conjugate: negate the imaginary components */
export const conjugate = (q: Quaternion): Quaternion => ({
	t: q.t,
	p: -q.p,
	f: -q.f,
	l: -q.l,
});

/** Squared norm: t² + p² + f² + l² */
export const normSquared = (q: Quaternion): number => q.t ** 2 + q.p ** 2 + q.f ** 2 + q.l ** 2;

/** Norm (magnitude) */
export const norm = (q: Quaternion): number => Math.sqrt(normSquared(q));

/** Inverse: conjugate / norm² */
export const inverse = (q: Quaternion): Quaternion => {
	const ns = normSquared(q);
	if (ns === 0) throw new Error("Cannot invert zero quaternion");
	return scale(conjugate(q), 1 / ns);
};

/** Project onto real (time) axis — the market-observable scalar */
export const project = (q: Quaternion): number => q.t;
