/**
 * evaluate.ts — Score prediction calibration against realized outcomes
 *
 * For each expired option in predictions.ndjson, fetches the Deribit settlement
 * price, determines ITM/OTM, and computes:
 *
 *   Brier score  = mean((prob − outcome)²)  [lower is better; perfect = 0]
 *
 * Also prints a reliability table (actual ITM rate per probability bucket) to
 * check whether the model is well-calibrated — the 60%-bucket should produce
 * ~60% ITM outcomes.
 *
 * Usage:
 *   deno run --allow-net --allow-env --allow-read --allow-write evaluate.ts
 */

import { fetchDeribitSettlement, type DeribitCurrency } from "../src/deribit/deribit.ts";
import { loadPredictions, savePredictions, type Prediction } from "../src/predictions.ts";

const predictions = await loadPredictions();
const now = Date.now() / 1000;

// Partition
const resolved   = predictions.filter((p) => p.itm !== undefined);
const toResolve  = predictions.filter((p) => p.itm === undefined && p.expiryTs < now);
const pending    = predictions.filter((p) => p.itm === undefined && p.expiryTs >= now);

console.log(`\nLoaded ${predictions.length} predictions`);
console.log(`  resolved : ${resolved.length}`);
console.log(`  to resolve: ${toResolve.length}`);
console.log(`  pending  : ${pending.length} (not yet expired)\n`);

// Resolve newly expired predictions
if (toResolve.length > 0) {
	// Batch settlement lookups: one API call per (ticker, expiryDate) pair
	const needed = new Map<string, Set<string>>();
	for (const p of toResolve) {
		if (!needed.has(p.ticker)) needed.set(p.ticker, new Set());
		needed.get(p.ticker)!.add(p.expiryDate);
	}

	const settlements = new Map<string, number>(); // "TICKER-YYYY-MM-DD" → price
	for (const [ticker, dates] of needed) {
		console.log(`Fetching settlement prices for ${ticker} (${[...dates].join(", ")})...`);
		try {
			for (const date of dates) {
				const price = await fetchDeribitSettlement(ticker as DeribitCurrency, date);
				if (price != null) {
					settlements.set(`${ticker}-${date}`, price);
					console.log(`  ${date}: $${price.toFixed(2)}`);
				} else {
					console.log(`  ${date}: not found (may not be settled yet)`);
				}
			}
		} catch (err) {
			console.error(`  ERROR: ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	let newlyResolved = 0;
	for (const p of toResolve) {
		const key = `${p.ticker}-${p.expiryDate}`;
		const settlement = settlements.get(key);
		if (settlement == null) continue;

		p.settlement = settlement;
		p.itm = p.optionType === "call"
			? settlement > p.strike
			: settlement < p.strike;
		newlyResolved++;
	}
	console.log(`\nResolved ${newlyResolved} new predictions.\n`);

	await savePredictions(predictions);
}

// Score all resolved predictions
const all = predictions.filter((p) => p.itm !== undefined);
if (all.length === 0) {
	console.log("No resolved predictions yet — collect data and wait for options to expire.");
	Deno.exit(0);
}

const brierScore = (probs: number[], outcomes: number[]): number =>
	probs.reduce((s, p, i) => s + (p - outcomes[i]!) ** 2, 0) / probs.length;

const outcomes = all.map((p) => p.itm ? 1 : 0);
const clasProbs = all.map((p) => p.classicalProb);
const quatProbs = all.map((p) => p.quatProb);

const classBrier = brierScore(clasProbs, outcomes);
const quatBrier  = brierScore(quatProbs, outcomes);
const brierAlpha = classBrier - quatBrier; // positive = quat is better

console.log("─".repeat(60));
console.log("  Brier Scores  (lower = better calibrated)");
console.log("─".repeat(60));
console.log(`  Classical  : ${classBrier.toFixed(5)}`);
console.log(`  Quaternion : ${quatBrier.toFixed(5)}`);
console.log(
	`  Alpha      : ${brierAlpha >= 0 ? "+" : ""}${brierAlpha.toFixed(5)} ` +
	`${brierAlpha > 0 ? "← quaternionic wins" : "← classical wins"}`,
);
console.log(`  N          : ${all.length} resolved predictions\n`);

// Reliability table
const BUCKETS = 10;
type Bucket = { n: number; sumProb: number; sumOutcome: number };
const mkBuckets = (): Bucket[] => Array.from({ length: BUCKETS }, () => ({ n: 0, sumProb: 0, sumOutcome: 0 }));

const classBuckets = mkBuckets();
const quatBuckets  = mkBuckets();

for (let i = 0; i < all.length; i++) {
	const outcome = outcomes[i]!;
	const ci = Math.min(Math.floor(clasProbs[i]! * BUCKETS), BUCKETS - 1);
	const qi = Math.min(Math.floor(quatProbs[i]! * BUCKETS), BUCKETS - 1);
	classBuckets[ci]!.n++;
	classBuckets[ci]!.sumProb += clasProbs[i]!;
	classBuckets[ci]!.sumOutcome += outcome;
	quatBuckets[qi]!.n++;
	quatBuckets[qi]!.sumProb += quatProbs[i]!;
	quatBuckets[qi]!.sumOutcome += outcome;
}

const pct = (x: number) => (x * 100).toFixed(1).padStart(5) + "%";

console.log("─".repeat(60));
console.log("  Reliability (predicted prob vs actual ITM rate)");
console.log("─".repeat(60));
console.log("  Bucket     Classical          Quaternionic");
console.log("  ─────────  ─────────────────  ─────────────────");

for (let b = 0; b < BUCKETS; b++) {
	const lo = b / BUCKETS;
	const hi = (b + 1) / BUCKETS;
	const label = `${pct(lo)}–${pct(hi)}`;

	const cb = classBuckets[b]!;
	const qb = quatBuckets[b]!;

	const classStr = cb.n > 0
		? `${pct(cb.sumOutcome / cb.n)} ITM (n=${cb.n}, mean pred ${pct(cb.sumProb / cb.n)})`
		: "(no data)";
	const quatStr = qb.n > 0
		? `${pct(qb.sumOutcome / qb.n)} ITM (n=${qb.n}, mean pred ${pct(qb.sumProb / qb.n)})`
		: "(no data)";

	console.log(`  ${label}  ${classStr.padEnd(25)}  ${quatStr}`);
}

console.log(`\n${"─".repeat(60)}\n`);
