/**
 * collect.ts — Snapshot predictions to disk
 *
 * For each ticker, fetches a live options snapshot, calibrates the quaternionic
 * model, then for every liquid option at the nearest expiry (>= 7 DTE) records:
 *
 *   classicalProb  — N(d₂) from scalar Black-Scholes using calibrated ATM vol
 *   quatProb       — Nd₂.t from quaternionic Black-Scholes
 *
 * Appends to predictions.ndjson. Run on a schedule (e.g. daily cron) to build
 * a time series. evaluate.ts resolves outcomes after settlement.
 *
 * Usage:
 *   deno run --allow-net --allow-env --allow-read --allow-write collect.ts BTC ETH
 */

import { fetchDeribitOptionsChain } from "./collect-market-data.ts";
import { fitQuatVol, extractQuatSpot, liquidQuotes } from "../src/backtest/backtest.ts";
import { price as bsPrice, normalCDF, quatN, type BSParams } from "../src/bs-model/bs-model.ts";
import type { Quaternion } from "../src/quaternion/quaternion.ts";
import { appendPredictions, type Prediction } from "./predictions.ts";

const RISK_FREE_RATE = 0.0525;
const MIN_DTE = 7;

const isoDate = (ts: number) => new Date(ts * 1000).toISOString().slice(0, 10);

const timeToExpiry = (asOf: Date, expiryTs: number): number =>
	Math.max((expiryTs - asOf.getTime() / 1000) / (60 * 60 * 24 * 365.25), 1 / 365);

const args = Deno.args.length > 0 ? Deno.args : ["BTC", "ETH"];
const recordedAt = new Date().toISOString();

for (const ticker of args) {
	const currency = ticker.toUpperCase() as "BTC" | "ETH";
	console.log(`\n${"─".repeat(60)}`);
	console.log(`  ${currency}`);
	console.log(`${"─".repeat(60)}`);

	try {
		const snapshot = await fetchDeribitOptionsChain(currency);
		const rate = RISK_FREE_RATE;
		const { vol } = fitQuatVol(snapshot, rate);

		// Nearest expiry with >= MIN_DTE days remaining
		const minTs = snapshot.asOf.getTime() / 1000 + MIN_DTE * 24 * 60 * 60;
		const firstExpiryTs = [...snapshot.calls, ...snapshot.puts]
			.map((q) => q.expiryTs)
			.filter((ts) => ts >= minTs)
			.sort((a, b) => a - b)[0];

		if (!firstExpiryTs) {
			console.log("  No eligible expiry found — skipping.");
			continue;
		}

		const expiryDate = isoDate(firstExpiryTs);
		const quatSpot = extractQuatSpot(snapshot, rate, firstExpiryTs);
		const realSpot: Quaternion = { t: snapshot.spot, p: 0, f: 0, l: 0 };
		const realVol: Quaternion = { t: vol.t, p: 0, f: 0, l: 0 };

		const quotes = liquidQuotes([
			...snapshot.calls.filter((q) => q.expiryTs === firstExpiryTs),
			...snapshot.puts.filter((q) => q.expiryTs === firstExpiryTs),
		]);

		const predictions: Prediction[] = [];

		for (const q of quotes) {
			if (q.mid < 0.01) continue;
			const tte = timeToExpiry(snapshot.asOf, firstExpiryTs);

			const baseParams: BSParams = {
				spot: realSpot,
				strike: q.strike,
				expiry: tte,
				rate,
				vol: realVol,
			};

			let classicalProb: number;
			let quatProb: number;

			try {
				const cf = bsPrice(baseParams);
				// N(d₂): risk-neutral probability of expiring ITM under classical BS
				classicalProb = q.type === "call"
					? normalCDF(cf.d2.t)
					: 1 - normalCDF(cf.d2.t);
			} catch {
				continue;
			}

			try {
				const qf = bsPrice({ ...baseParams, spot: quatSpot, vol });
				// Nd₂.t for calls; 1 − Nd₂.t for puts (put-call symmetry on the real part)
				const nd2t = quatN(qf.d2).t;
				quatProb = q.type === "call" ? nd2t : 1 - nd2t;
			} catch {
				continue;
			}

			predictions.push({
				recordedAt,
				ticker: currency,
				strike: q.strike,
				expiryDate,
				expiryTs: firstExpiryTs,
				optionType: q.type,
				spot: snapshot.spot,
				tte,
				classicalProb,
				quatProb,
			});
		}

		await appendPredictions(predictions);

		console.log(`  As of      : ${snapshot.asOf.toISOString()}`);
		console.log(`  Spot       : $${snapshot.spot.toFixed(2)}`);
		console.log(`  Expiry     : ${expiryDate}`);
		console.log(`  Recorded   : ${predictions.length} predictions`);

		const pctDiff = predictions.map((p) => Math.abs(p.quatProb - p.classicalProb));
		if (pctDiff.length > 0) {
			const meanDiff = pctDiff.reduce((s, x) => s + x, 0) / pctDiff.length;
			console.log(`  Mean |quat − classical| : ${(meanDiff * 100).toFixed(2)}pp`);
		}
	} catch (err) {
		console.error(`  ERROR: ${err instanceof Error ? err.message : String(err)}`);
	}
}

console.log(`\n${"─".repeat(60)}\n`);
