/**
 * Vol Surface Signal Processing
 *
 * Uses DFT (not FFT — N is small, O(N²) is fine for option chains) to:
 *   1. Smooth the IV-vs-log-moneyness curve by zeroing high-frequency modes
 *   2. Extract vol.t, vol.p (skew), and vol.l (smile curvature) from the
 *      smoothed curve via finite differences
 *
 * These replace the OLS fits in fitQuatVol for those three components.
 * vol.f (term structure) is kept on OLS since it operates across a different
 * axis (sqrt(T)) and requires multi-expiry data.
 *
 * Mapping:
 *   Mode 0 (DC)   → vol.t  (mean / ATM implied vol)
 *   Mode 1 (linear) → vol.p  (skew slope at ATM)
 *   Mode 2 (quadratic) → vol.l (smile curvature at ATM — direct calibration
 *                                instead of leaving it emergent from Σ²)
 */

type Complex = { re: number; im: number };

// ---------------------------------------------------------------------------
// Core DFT / IDFT
// ---------------------------------------------------------------------------

/** 1D DFT of a real signal. Returns N complex coefficients. */
const dft = (signal: number[]): Complex[] => {
	const N = signal.length;
	return Array.from({ length: N }, (_, k) => {
		let re = 0, im = 0;
		for (let n = 0; n < N; n++) {
			const phi = -2 * Math.PI * k * n / N;
			re += signal[n]! * Math.cos(phi);
			im += signal[n]! * Math.sin(phi);
		}
		return { re, im };
	});
};

/** 1D IDFT of a complex spectrum. Returns real-valued output. */
const idft = (F: Complex[]): number[] => {
	const N = F.length;
	return Array.from({ length: N }, (_, n) => {
		let re = 0;
		for (let k = 0; k < N; k++) {
			const phi = 2 * Math.PI * k * n / N;
			re += F[k]!.re * Math.cos(phi) - F[k]!.im * Math.sin(phi);
		}
		return re / N;
	});
};

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/**
 * Interpolate irregular (xs, ys) samples onto a uniform grid of N points
 * spanning [xMin, xMax]. Uses linear interpolation between neighbours;
 * flat-extrapolates outside the observed data range.
 */
export const interpolateUniform = (
	xs: number[],
	ys: number[],
	N: number,
	xMin: number,
	xMax: number,
): number[] => {
	const pts = xs
		.map((x, i) => [x, ys[i]!] as [number, number])
		.sort(([a], [b]) => a - b);

	return Array.from({ length: N }, (_, i) => {
		const xTarget = xMin + (xMax - xMin) * i / (N - 1);

		if (xTarget <= pts[0]![0]) return pts[0]![1];
		if (xTarget >= pts[pts.length - 1]![0]) return pts[pts.length - 1]![1];

		let lo = 0, hi = pts.length - 1;
		while (hi - lo > 1) {
			const mid = (lo + hi) >> 1;
			if (pts[mid]![0] <= xTarget) lo = mid;
			else hi = mid;
		}
		const [x0, y0] = pts[lo]!;
		const [x1, y1] = pts[hi]!;
		return y0 + (xTarget - x0) / (x1 - x0) * (y1 - y0);
	});
};

// ---------------------------------------------------------------------------
// Low-pass filter
// ---------------------------------------------------------------------------

/**
 * Low-pass filter on a uniformly-sampled signal.
 *
 * DFTs the signal, zeros all modes from index `keepModes` through
 * N − keepModes (the high-frequency half), then IDFTs back.
 * The result is a smooth curve with only the lowest keepModes harmonics.
 */
export const lowPass = (signal: number[], keepModes: number): number[] => {
	const N = signal.length;
	const F = dft(signal);
	for (let k = keepModes; k <= N - keepModes; k++) {
		F[k] = { re: 0, im: 0 };
	}
	return idft(F);
};

// ---------------------------------------------------------------------------
// Vol curve extraction
// ---------------------------------------------------------------------------

/**
 * Extract ATM vol, skew slope, and smile curvature from an IV-vs-moneyness
 * dataset using DFT-based smoothing.
 *
 * Steps:
 *   1. Interpolate the irregular (log-moneyness, IV) points onto a uniform
 *      grid spanning the observed moneyness range.
 *   2. Low-pass filter (keep modes 0..keepModes-1) to remove noise from
 *      illiquid or stale quotes.
 *   3. Evaluate the smooth curve at x = 0 (ATM) → vol.t
 *   4. First finite difference at x = 0 → vol.p (skew slope)
 *   5. Second finite difference at x = 0 → vol.l (curvature)
 *
 * Returns:
 *   volT  — smoothed ATM implied vol
 *   volP  — dIV/d(log-moneyness) at ATM  (skew, i-dimension of vol)
 *   volL  — d²IV/d(log-moneyness)² at ATM (curvature, k-dimension of vol)
 */
export const extractVolCurve = (
	moneyness: number[],
	ivs: number[],
	N = 32,
	keepModes = 3,
): { volT: number; volP: number; volL: number } => {
	const fallback = { volT: ivs.reduce((s, v) => s + v, 0) / (ivs.length || 1), volP: 0, volL: 0 };

	if (moneyness.length < 3) return fallback;

	const xMin = Math.min(...moneyness);
	const xMax = Math.max(...moneyness);
	if (xMax - xMin < 1e-6) return { ...fallback, volT: ivs[0] ?? fallback.volT };

	// Uniform grid
	const grid = interpolateUniform(moneyness, ivs, N, xMin, xMax);

	// Low-pass filter
	const smooth = lowPass(grid, Math.min(keepModes, Math.floor(N / 2)));

	const step = (xMax - xMin) / (N - 1);

	// Grid index closest to ATM (x = 0)
	const atmIdx = Math.max(0, Math.min(N - 1, Math.round(-xMin / step)));
	const volT = smooth[atmIdx] ?? fallback.volT;

	// Skew: central difference, or forward/backward at the edges
	const loIdx = Math.max(0, atmIdx - 1);
	const hiIdx = Math.min(N - 1, atmIdx + 1);
	const volP = hiIdx > loIdx
		? ((smooth[hiIdx] ?? 0) - (smooth[loIdx] ?? 0)) / ((hiIdx - loIdx) * step)
		: 0;

	// Curvature: second central difference (requires interior point)
	const volL = atmIdx > 0 && atmIdx < N - 1
		? ((smooth[atmIdx + 1] ?? 0) - 2 * volT + (smooth[atmIdx - 1] ?? 0)) / (step ** 2)
		: 0;

	return { volT, volP, volL };
};
