/**
 * historical-backtest.ts — Retroactive directional calibration test
 *
 * Uses Deribit public historical data to test whether the quaternionic model's
 * ITM probability (Nd₂.t) is better calibrated than classical N(d₂) across
 * past expiries without needing to wait for future settlements.
 *
 * For each past expiry, reconstructs the state 7 days prior using:
 *   - Deribit settlement price (realized outcome)
 *   - Daily spot OHLCV from BTC/ETH perpetual
 *   - Historical volatility (Deribit's volatility index)
 *   - Historical 8h funding rate (perpetual swap)
 *
 * Tests a grid of strikes around spot at T-7. Because historical options chains
 * are not available, vol.p/f/l = 0 — this specifically isolates the funding
 * rate (spot.p) hypothesis. The Brier score tells us whether it helps.
 *
 * Usage:
 *   deno run --allow-net --allow-env --allow-read historical-backtest.ts
 *   deno run --allow-net --allow-env --allow-read historical-backtest.ts ETH
 */

import axios from "axios";
import { z } from "zod";
import { price as bsPrice, normalCDF, quatN, type BSParams } from "../src/bs-model/bs-model.ts";
import type { Quaternion } from "../src/quaternion/quaternion.ts";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DERIBIT = "https://www.deribit.com/api/v2/public";
const RATE = 0.0525;
const HORIZON_DAYS = 7;
const DELIVERIES_TO_FETCH = 40;

// Strike moneyness offsets: test calls and puts at these levels relative to spot
const MONEYNESS = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20];

// ---------------------------------------------------------------------------
// Deribit schemas
// ---------------------------------------------------------------------------

const DeliveryZ = z.object({
	result: z.object({
		data: z.array(z.object({ date: z.string(), delivery_price: z.number() })),
	}),
});

// [[timestamp_ms, vol_pct], ...]
const HistVolZ = z.object({
	result: z.array(z.tuple([z.number(), z.number()])),
});

const OhlcvZ = z.object({
	result: z.object({
		ticks: z.array(z.number()),
		close: z.array(z.number()),
	}),
});

const FundingZ = z.object({
	result: z.object({
		data: z.array(z.object({
			timestamp: z.number(),
			interest_8h: z.number(),
		})),
	}),
});

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

const fetchDeliveries = async (indexName: string) => {
	const res = await axios.get(`${DERIBIT}/get_delivery_prices?index_name=${indexName}&count=${DELIVERIES_TO_FETCH}`);
	return DeliveryZ.parse(res.data).result.data;
};

const fetchHistVol = async (currency: string): Promise<Array<[number, number]>> => {
	const res = await axios.get(`${DERIBIT}/get_historical_volatility?currency=${currency}`);
	return HistVolZ.parse(res.data).result;
};

/** Daily OHLCV for instrument going back ~2 years */
const fetchDailyOhlcv = async (instrument: string): Promise<Array<{ ts: number; close: number }>> => {
	const end = Date.now();
	const start = end - 2 * 365 * 24 * 60 * 60 * 1000;
	const res = await axios.get(
		`${DERIBIT}/get_tradingview_chart_data?instrument_name=${instrument}&start_timestamp=${start}&end_timestamp=${end}&resolution=D`,
	);
	const { ticks, close } = OhlcvZ.parse(res.data).result;
	return ticks.map((ts, i) => ({ ts, close: close[i] ?? 0 }));
};

/** Historical funding rates — Deribit returns ~30 days of data with length=1m */
const fetchFundingHistory = async (perpName: string): Promise<Array<{ ts: number; rate: number }>> => {
	try {
		const res = await axios.get(`${DERIBIT}/get_funding_chart_data?instrument_name=${perpName}&length=1m`);
		return FundingZ.parse(res.data).result.data.map((d) => ({ ts: d.timestamp, rate: d.interest_8h }));
	} catch {
		return [];
	}
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Find the entry in a sorted-by-ts array closest to targetTs (ms) */
const findClosest = <T extends { ts: number }>(series: T[], targetTs: number): T | null => {
	if (series.length === 0) return null;
	return series.reduce((best, cur) =>
		Math.abs(cur.ts - targetTs) < Math.abs(best.ts - targetTs) ? cur : best
	);
};

const brierScore = (probs: number[], outcomes: number[]): number =>
	probs.reduce((s, p, i) => s + (p - outcomes[i]!) ** 2, 0) / probs.length;

// ---------------------------------------------------------------------------
// Per-expiry scoring
// ---------------------------------------------------------------------------

type PredRow = {
	date: string;
	strike: number;
	type: "call" | "put";
	spot: number;
	vol: number;
	funding8h: number;
	tte: number;
	classicalProb: number;
	quatProb: number;
	settlement: number;
	itm: boolean;
};

const scoreExpiry = (
	date: string,
	settlement: number,
	spot: number,
	vol: number,        // annualised decimal
	funding8h: number,  // 8h rate, signed
	tte: number,        // years to expiry (7/365)
): PredRow[] => {
	const rows: PredRow[] = [];

	const realSpot: Quaternion = { t: spot, p: 0, f: 0, l: 0 };
	const quatSpot: Quaternion = { t: spot, p: funding8h * spot, f: 0, l: 0 };
	const realVol: Quaternion = { t: vol, p: 0, f: 0, l: 0 };

	for (const offset of MONEYNESS) {
		const strike = Math.round(spot * (1 + offset));

		for (const optType of ["call", "put"] as const) {
			const params: BSParams = {
				spot: realSpot,
				strike,
				expiry: tte,
				rate: RATE,
				vol: realVol,
			};
			const quatParams: BSParams = { ...params, spot: quatSpot };

			let classicalProb: number;
			let quatProb: number;

			try {
				const cf = bsPrice(params);
				classicalProb = optType === "call" ? normalCDF(cf.d2.t) : 1 - normalCDF(cf.d2.t);
				const qf = bsPrice(quatParams);
				quatProb = optType === "call" ? quatN(qf.d2).t : 1 - quatN(qf.d2).t;
			} catch {
				continue;
			}

			const itm = optType === "call" ? settlement > strike : settlement < strike;

			rows.push({ date, strike, type: optType, spot, vol, funding8h, tte, classicalProb, quatProb, settlement, itm });
		}
	}

	return rows;
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const tickers = Deno.args.length > 0
	? Deno.args.map((t) => t.toUpperCase())
	: ["BTC", "ETH"];

type TickerResult = {
	ticker: string;
	n: number;
	classicalBrier: number;
	quatBrier: number;
	fundingCoverage: number; // fraction of rows with non-zero funding
};

const allResults: TickerResult[] = [];

for (const ticker of tickers) {
	console.log(`\n${"─".repeat(60)}`);
	console.log(`  ${ticker}  —  historical directional backtest`);
	console.log(`${"─".repeat(60)}`);

	try {
		const indexName = ticker === "BTC" ? "btc_usd" : "eth_usd";
		const perpName = `${ticker}-PERPETUAL`;

		console.log("  Fetching data...");
		const [deliveries, histVol, ohlcv, funding] = await Promise.all([
			fetchDeliveries(indexName),
			fetchHistVol(ticker),
			fetchDailyOhlcv(perpName),
			fetchFundingHistory(perpName),
		]);

		const fundingFrom = funding.length > 0 ? new Date(Math.min(...funding.map((f) => f.ts))).toISOString().slice(0, 10) : "n/a";
		const fundingTo   = funding.length > 0 ? new Date(Math.max(...funding.map((f) => f.ts))).toISOString().slice(0, 10) : "n/a";
		console.log(`  Deliveries: ${deliveries.length}, vol: ${histVol.length}, ohlcv: ${ohlcv.length}, funding: ${funding.length} (${fundingFrom} → ${fundingTo})`);

		// Convert historical vol series to { ts, vol } (vol as decimal, not percent)
		const volSeries = histVol.map(([ts, pct]) => ({ ts, vol: pct / 100 }));

		const allRows: PredRow[] = [];

		for (const { date, delivery_price: settlement } of deliveries) {
			// T-7: 7 days before expiry at 08:00 UTC
			const expiryMs = new Date(`${date}T08:00:00Z`).getTime();
			const targetMs = expiryMs - HORIZON_DAYS * 24 * 60 * 60 * 1000;

			const spotEntry = findClosest(ohlcv, targetMs);
			const volEntry = findClosest(volSeries, targetMs);
			if (!spotEntry || spotEntry.close <= 0) continue;
			if (!volEntry || volEntry.vol <= 0) continue;

			const spot = spotEntry.close;
			const vol = volEntry.vol;
			const tte = HORIZON_DAYS / 365.25;

			// Funding: match to T-7; if not available, use 0
			const fundingEntry = funding.length > 0 ? findClosest(funding, targetMs) : null;
			// Trust funding if it's within the range of the fetched series (use
			// the series min/max as bounds with a 2-day buffer either side)
			const fundingMin = funding.length > 0 ? Math.min(...funding.map((f) => f.ts)) : Infinity;
			const fundingMax = funding.length > 0 ? Math.max(...funding.map((f) => f.ts)) : -Infinity;
			const inFundingRange = targetMs >= fundingMin - 2 * 24 * 60 * 60 * 1000 &&
				targetMs <= fundingMax + 2 * 24 * 60 * 60 * 1000;
			const fundingRate = fundingEntry && inFundingRange ? fundingEntry.rate : 0;

			const rows = scoreExpiry(date, settlement, spot, vol, fundingRate, tte);
			allRows.push(...rows);
		}

		if (allRows.length === 0) {
			console.log("  No scored rows — check data availability.");
			continue;
		}

		const outcomes = allRows.map((r) => r.itm ? 1 : 0);
		const classBrier = brierScore(allRows.map((r) => r.classicalProb), outcomes);
		const quatBrier  = brierScore(allRows.map((r) => r.quatProb), outcomes);
		const alpha = classBrier - quatBrier;
		const fundingCoverage = allRows.filter((r) => r.funding8h !== 0).length / allRows.length;

		console.log(`\n  Predictions   : ${allRows.length} (${deliveries.length} expiries × ${MONEYNESS.length * 2} strikes/types)`);
		console.log(`  Funding cover : ${(fundingCoverage * 100).toFixed(0)}% of rows have non-zero funding`);
		console.log(`\n  Classical Brier : ${classBrier.toFixed(5)}`);
		console.log(`  Quatern. Brier  : ${quatBrier.toFixed(5)}`);
		console.log(
			`  Alpha           : ${alpha >= 0 ? "+" : ""}${alpha.toFixed(5)} ` +
			`${alpha > 0 ? "← quaternionic wins" : "← classical wins"}`,
		);

		// Reliability table
		const BUCKETS = 5;
		type B = { n: number; sumProb: number; sumOutcome: number };
		const mkB = (): B[] => Array.from({ length: BUCKETS }, () => ({ n: 0, sumProb: 0, sumOutcome: 0 }));
		const cb = mkB(), qb = mkB();
		for (let i = 0; i < allRows.length; i++) {
			const o = outcomes[i]!;
			const ci = Math.min(Math.floor(allRows[i]!.classicalProb * BUCKETS), BUCKETS - 1);
			const qi = Math.min(Math.floor(allRows[i]!.quatProb * BUCKETS), BUCKETS - 1);
			cb[ci]!.n++; cb[ci]!.sumProb += allRows[i]!.classicalProb; cb[ci]!.sumOutcome += o;
			qb[qi]!.n++; qb[qi]!.sumProb += allRows[i]!.quatProb;     qb[qi]!.sumOutcome += o;
		}

		console.log(`\n  Reliability (predicted → actual ITM rate):`);
		console.log(`  Bucket     Classical                 Quaternionic`);
		for (let b = 0; b < BUCKETS; b++) {
			const lo = (b / BUCKETS * 100).toFixed(0).padStart(3);
			const hi = ((b + 1) / BUCKETS * 100).toFixed(0).padStart(3);
			const fmt = (x: B) => x.n > 0
				? `${(x.sumOutcome / x.n * 100).toFixed(1).padStart(5)}% ITM  pred ${(x.sumProb / x.n * 100).toFixed(1).padStart(5)}%  (n=${x.n})`
				: "(no data)";
			console.log(`  ${lo}–${hi}%    ${fmt(cb[b]!).padEnd(38)} ${fmt(qb[b]!)}`);
		}

		allResults.push({ ticker, n: allRows.length, classicalBrier: classBrier, quatBrier, fundingCoverage });
	} catch (err) {
		console.error(`  ERROR: ${err instanceof Error ? err.message : String(err)}`);
	}
}

// Summary
if (allResults.length > 1) {
	console.log(`\n${"─".repeat(60)}`);
	console.log("  Summary");
	console.log(`${"─".repeat(60)}`);
	for (const r of allResults) {
		const alpha = r.classicalBrier - r.quatBrier;
		console.log(
			`  ${r.ticker.padEnd(4)}  Brier class=${r.classicalBrier.toFixed(5)}  quat=${r.quatBrier.toFixed(5)}  ` +
			`alpha=${alpha >= 0 ? "+" : ""}${alpha.toFixed(5)}  funding=${(r.fundingCoverage * 100).toFixed(0)}%  n=${r.n}`,
		);
	}
}

console.log(`\n${"─".repeat(60)}\n`);
