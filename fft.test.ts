import { assertEquals, assertAlmostEquals } from "@std/assert";
import { extractVolCurve, interpolateUniform, lowPass } from "./fft.ts";

// ---------------------------------------------------------------------------
// interpolateUniform
// ---------------------------------------------------------------------------

Deno.test("interpolateUniform: exact nodes are reproduced", () => {
	const xs = [0, 0.5, 1];
	const ys = [1, 2, 3];
	const grid = interpolateUniform(xs, ys, 3, 0, 1);
	assertAlmostEquals(grid[0]!, 1, 1e-10);
	assertAlmostEquals(grid[1]!, 2, 1e-10);
	assertAlmostEquals(grid[2]!, 3, 1e-10);
});

Deno.test("interpolateUniform: midpoints are linearly interpolated", () => {
	const xs = [0, 1];
	const ys = [0, 1];
	const grid = interpolateUniform(xs, ys, 5, 0, 1);
	// should give [0, 0.25, 0.5, 0.75, 1]
	assertAlmostEquals(grid[2]!, 0.5, 1e-10);
});

Deno.test("interpolateUniform: flat-extrapolates outside range", () => {
	const xs = [0.2, 0.8];
	const ys = [10, 20];
	const grid = interpolateUniform(xs, ys, 5, 0, 1);
	// Left edge extrapolation: should equal ys[0] = 10
	assertAlmostEquals(grid[0]!, 10, 1e-10);
	// Right edge extrapolation: should equal ys[1] = 20
	assertAlmostEquals(grid[4]!, 20, 1e-10);
});

Deno.test("interpolateUniform: N=1 edge case", () => {
	const grid = interpolateUniform([0, 1], [5, 10], 1, 0, 1);
	assertEquals(grid.length, 1);
});

// ---------------------------------------------------------------------------
// lowPass
// ---------------------------------------------------------------------------

Deno.test("lowPass: DC-only signal is unchanged", () => {
	const signal = [3, 3, 3, 3, 3, 3, 3, 3];
	const filtered = lowPass(signal, 1);
	for (const v of filtered) assertAlmostEquals(v, 3, 1e-9);
});

Deno.test("lowPass: single sine wave is preserved when within keepModes", () => {
	// signal = sin(2π n / N), fundamental frequency = mode 1
	const N = 16;
	const signal = Array.from({ length: N }, (_, n) => Math.sin(2 * Math.PI * n / N));
	const filtered = lowPass(signal, 2); // keep mode 0 and 1
	for (let n = 0; n < N; n++) {
		assertAlmostEquals(filtered[n]!, signal[n]!, 1e-9);
	}
});

Deno.test("lowPass: high-frequency component is zeroed", () => {
	// signal = cos(2π * 7 * n / 16) — mode 7, should be killed by keepModes=3
	const N = 16;
	const signal = Array.from({ length: N }, (_, n) => Math.cos(2 * Math.PI * 7 * n / N));
	const filtered = lowPass(signal, 3);
	for (const v of filtered) assertAlmostEquals(v, 0, 1e-9);
});

Deno.test("lowPass: output length equals input length", () => {
	const signal = [1, 2, 3, 4, 5, 6, 7, 8];
	const filtered = lowPass(signal, 2);
	assertEquals(filtered.length, signal.length);
});

// ---------------------------------------------------------------------------
// extractVolCurve
// ---------------------------------------------------------------------------

Deno.test("extractVolCurve: flat surface gives constant volT, zero skew/curv", () => {
	const xs = [-0.2, -0.1, 0, 0.1, 0.2];
	const ivs = [0.3, 0.3, 0.3, 0.3, 0.3];
	const { volT, volP, volL } = extractVolCurve(xs, ivs);
	assertAlmostEquals(volT, 0.3, 1e-6);
	assertAlmostEquals(volP, 0, 1e-6);
	assertAlmostEquals(volL, 0, 1e-4);
});

Deno.test("extractVolCurve: linear surface gives nonzero skew", () => {
	// IV = 0.3 + 0.1 * x  (positive slope)
	const N = 10;
	const xs = Array.from({ length: N }, (_, i) => -0.4 + i * 0.1);
	const ivs = xs.map((x) => 0.3 + 0.1 * x);
	const { volT, volP } = extractVolCurve(xs, ivs);
	assertAlmostEquals(volT, 0.3, 0.01); // IV at x=0, DFT reconstruction tolerance
	// Skew should be positive
	assertEquals(volP > 0, true);
});

Deno.test("extractVolCurve: symmetric smile gives near-zero skew", () => {
	// IV = 0.2 + 0.5 * x²  (symmetric U-shape, no skew)
	const xs = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3];
	const ivs = xs.map((x) => 0.2 + 0.5 * x * x);
	const { volT, volP, volL } = extractVolCurve(xs, ivs);
	assertAlmostEquals(volT, 0.2, 0.01);     // ATM IV, DFT reconstruction tolerance
	assertAlmostEquals(volP, 0, 0.02);        // near-zero skew (symmetric)
	assertEquals(volL > 0, true);             // positive curvature
});

Deno.test("extractVolCurve: fewer than 3 points returns fallback", () => {
	const { volT, volP, volL } = extractVolCurve([0.1], [0.25]);
	assertAlmostEquals(volT, 0.25, 1e-10);
	assertAlmostEquals(volP, 0, 1e-10);
	assertAlmostEquals(volL, 0, 1e-10);
});

Deno.test("extractVolCurve: noise is attenuated by low-pass filter", () => {
	// Smooth underlying: IV = 0.25 + 0.05 * x
	// Plus high-frequency noise
	const xs = Array.from({ length: 20 }, (_, i) => -0.4 + i * 0.04);
	const signal = xs.map((x) => 0.25 + 0.05 * x);
	const noise = xs.map((_, i) => (i % 2 === 0 ? 0.02 : -0.02));
	const ivs = xs.map((_, i) => signal[i]! + noise[i]!);

	const { volT: volTNoisy } = extractVolCurve(xs, ivs, 32, 3);
	const { volT: volTClean } = extractVolCurve(xs, signal, 32, 3);

	// Filtered result should be closer to the clean signal than the raw noisy input
	const rawAtmIv = ivs[Math.round(xs.length / 2)]!;
	assertAlmostEquals(volTNoisy, volTClean, Math.abs(rawAtmIv - volTClean));
});
