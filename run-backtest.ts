/**
 * run-backtest.ts
 *
 * Fetches live options chains for SPY, SPX, and AAPL via Polygon.io
 * and runs the quaternionic vs classical BS backtest on each.
 *
 * Usage:
 *   deno run --allow-net --allow-env --allow-read run-backtest.ts
 */

import { fetchOptionsChain } from "./market-data.ts";
import { runBacktest } from "./backtest.ts";

const RATE = 0.0525; // ~5.25% risk-free (3-month T-bill, March 2026)
const TICKERS = ["SPY", "SPX", "AAPL"];

for (const ticker of TICKERS) {
	console.log(`\n${"─".repeat(60)}`);
	console.log(`  ${ticker}`);
	console.log(`${"─".repeat(60)}`);

	try {
		const snapshot = await fetchOptionsChain(ticker);
		const result = runBacktest(snapshot, RATE);

		const pct = (x: number) => (x * 100).toFixed(4) + "%";
		const usd = (x: number) => "$" + x.toFixed(4);
		const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(4);

		console.log(`  As of          : ${result.asOf.toISOString()}`);
		console.log(`  Spot           : ${usd(result.spot)}`);
		console.log(`  Contracts used : ${result.comparisons.length}`);
		console.log(``);
		console.log(`  vol.t (ATM IV) : ${pct(result.volT)}`);
		console.log(`  vol.p (skew)   : ${sign(result.volP)}`);
		console.log(`  vol.f (TS)     : ${sign(result.volF)}`);
		console.log(`  spot.p (fund.) : ${usd(result.spotP)}`);
		console.log(`  spot.f (liq.)  : ${usd(result.spotF)}`);
		console.log(``);
		console.log(`  Classical RMSE : ${usd(result.classicalRMSE)}`);
		console.log(`  Quatern.  RMSE : ${usd(result.quaternionicRMSE)}`);
		console.log(
			`  Alpha          : ${sign(result.alpha)} ${
				result.alpha > 0 ? "← quaternionic wins" : "← classical wins"
			}`,
		);

		// Per-strike detail
		console.log(`\n  Strike   Type  Market     Classical  Quat       ΔClass   ΔQuat`);
		console.log(`  ${"─".repeat(70)}`);
		for (const c of result.comparisons) {
			const K = c.strike.toFixed(2).padStart(8);
			const t = c.type.padEnd(4);
			const mkt = c.marketMid.toFixed(3).padStart(9);
			const cls = c.classicalPrice.toFixed(3).padStart(9);
			const qua = c.quaternionicPrice.toFixed(3).padStart(9);
			const ec = c.classicalError.toFixed(3).padStart(7);
			const eq = c.quaternionicError.toFixed(3).padStart(7);
			console.log(`  ${K}   ${t}  ${mkt}  ${cls}  ${qua}  ${ec}  ${eq}`);
		}
	} catch (err) {
		console.error(`  ERROR: ${err instanceof Error ? err.message : String(err)}`);
	}
}

console.log(`\n${"─".repeat(60)}\n`);
