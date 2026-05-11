import { impliedVol, normalCDF, price, type BSParams } from "../src/bs-model/bs-model.ts";

type Row = {
	date: string;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	fundingRate: number | null;
};

type BacktestRow = {
	date: string;
	close: number;
	realizedVol: number;
	momentum: number;
	fundingRate: number | null;
	strike: number;
	classicalPrice: number;
	carryPrice: number | null;
	quatPrice: number | null;
	realizedPayoff: number;
	breakevenVol: number | null;
	classicalError: number;
	carryError: number | null;
	quatError: number | null;
};

/**
 * Merton (1973) call price with continuous dividend yield q.
 * Funding rate enters as a carry cost: longs pay q to hold the perpetual,
 * so the forward is F = S·exp((r−q)·T). When q > 0 (bull market funding),
 * the forward and thus the call are cheaper than vanilla BS.
 */
const carryCall = (S: number, K: number, T: number, r: number, sigma: number, q: number): number => {
	const sqrtT = Math.sqrt(T);
	const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
	const d2 = d1 - sigma * sqrtT;
	return S * Math.exp(-q * T) * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
};

const WINDOW = 30;
const HORIZON = 30;
const RATE = 0.05;

const annualizedVol = (closes: number[]): number => {
	const logReturns = closes.slice(1).map((c, i) => Math.log(c / closes[i]!));
	const mean = logReturns.reduce((a, b) => a + b, 0) / logReturns.length;
	const variance = logReturns.reduce((a, r) => a + (r - mean) ** 2, 0) / (logReturns.length - 1);
	return Math.sqrt(variance * 252);
};

const text = await Deno.readTextFile("data/BTC_daily.ndjson");
const rows: Row[] = text.trim().split("\n").map((l) => JSON.parse(l));
rows.sort((a, b) => a.date.localeCompare(b.date));

const results: BacktestRow[] = [];

for (let i = WINDOW; i < rows.length - HORIZON; i++) {
	const row = rows[i]!;
	const windowCloses = rows.slice(i - WINDOW, i + 1).map((r) => r.close);
	const sigma = annualizedVol(windowCloses);
	const momentum = row.close - rows[i - WINDOW]!.close;
	const strike = row.close;
	const T = HORIZON / 365;
	const futureClose = rows[i + HORIZON]!.close;
	const realizedPayoff = Math.max(futureClose - strike, 0);

	const classicalParams: BSParams = {
		spot: { t: row.close, p: 0, f: 0, l: 0 },
		strike,
		expiry: T,
		rate: RATE,
		vol: { t: sigma, p: 0, f: 0, l: 0 },
	};
	const classicalPrice = price(classicalParams).call.t;

	let carryPrice: number | null = null;
	let quatPrice: number | null = null;
	if (row.fundingRate !== null) {
		// OKX settles funding 3× per day; annualize to a continuous yield
		const q = row.fundingRate * 3 * 365;
		carryPrice = carryCall(row.close, strike, T, RATE, sigma, q);

		const annualizedFunding = row.fundingRate * 365;
		const quatParams: BSParams = {
			spot: { t: row.close, p: momentum, f: annualizedFunding * row.close, l: 0 },
			strike,
			expiry: T,
			rate: RATE,
			vol: { t: sigma, p: 0, f: 0, l: 0 },
		};
		quatPrice = price(quatParams).call.t;
	}

	let breakevenVol: number | null = null;
	if (realizedPayoff > 0) {
		try {
			breakevenVol = impliedVol(realizedPayoff, classicalParams);
		} catch { /* outside bounds or no convergence */ }
	}

	results.push({
		date: row.date,
		close: row.close,
		realizedVol: sigma,
		momentum,
		fundingRate: row.fundingRate,
		strike,
		classicalPrice,
		carryPrice,
		quatPrice,
		realizedPayoff,
		breakevenVol,
		classicalError: classicalPrice - realizedPayoff,
		carryError: carryPrice !== null ? carryPrice - realizedPayoff : null,
		quatError: quatPrice !== null ? quatPrice - realizedPayoff : null,
	});
}

await Deno.writeTextFile(
	"data/backtest.ndjson",
	results.map((r) => JSON.stringify(r)).join("\n") + "\n",
);

// --- display ---

const pad = (s: string, n: number) => s.padStart(n);
const fmt = (n: number | null, d = 2, w = 9) =>
	n === null ? pad("-", w) : pad(n.toFixed(d), w);

console.log(`\nBacktest: ${results.length} ATM ${HORIZON}-day calls on BTC/USDT\n`);
console.log(
	"Date        Close     RealVol   Classical  Carry     Quat      Realized   BEVol     CErr      CarryErr  QErr",
);
console.log("-".repeat(110));

for (const r of results) {
	console.log(
		`${r.date}  ` +
			`${pad(r.close.toFixed(0), 8)}  ` +
			`${pad((r.realizedVol * 100).toFixed(1) + "%", 7)}   ` +
			`${fmt(r.classicalPrice)}  ` +
			`${fmt(r.carryPrice)}  ` +
			`${fmt(r.quatPrice)}  ` +
			`${fmt(r.realizedPayoff)}  ` +
			`${r.breakevenVol !== null ? pad((r.breakevenVol * 100).toFixed(1) + "%", 7) : pad("-", 7)}  ` +
			`${fmt(r.classicalError)}  ` +
			`${fmt(r.carryError)}  ` +
			`${fmt(r.quatError)}`,
	);
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const mae = (xs: number[]) => mean(xs.map(Math.abs));
const rmse = (xs: number[]) => Math.sqrt(mean(xs.map((x) => x * x)));

const cErrs = results.map((r) => r.classicalError);
const carryErrs = results.flatMap((r) => r.carryError !== null ? [r.carryError] : []);
const qErrs = results.flatMap((r) => r.quatError !== null ? [r.quatError] : []);
// Classical errors restricted to the rows where funding data exists (fair comparison)
const cErrsOnFundingRows = results.flatMap((r) =>
	r.fundingRate !== null ? [r.classicalError] : []
);

console.log("\nSummary (all rows):");
console.log(
	`  Classical    n=${cErrs.length.toString().padEnd(3)}  ` +
		`MAE=${mae(cErrs).toFixed(2).padStart(8)}  ` +
		`RMSE=${rmse(cErrs).toFixed(2).padStart(8)}  ` +
		`Bias=${mean(cErrs).toFixed(2).padStart(8)}`,
);

console.log("\nSummary (funding-data rows only — apples-to-apples):");
console.log(
	`  Classical    n=${cErrsOnFundingRows.length.toString().padEnd(3)}  ` +
		`MAE=${mae(cErrsOnFundingRows).toFixed(2).padStart(8)}  ` +
		`RMSE=${rmse(cErrsOnFundingRows).toFixed(2).padStart(8)}  ` +
		`Bias=${mean(cErrsOnFundingRows).toFixed(2).padStart(8)}`,
);
if (carryErrs.length > 0) {
	console.log(
		`  Carry-adj    n=${carryErrs.length.toString().padEnd(3)}  ` +
			`MAE=${mae(carryErrs).toFixed(2).padStart(8)}  ` +
			`RMSE=${rmse(carryErrs).toFixed(2).padStart(8)}  ` +
			`Bias=${mean(carryErrs).toFixed(2).padStart(8)}`,
	);
}
if (qErrs.length > 0) {
	console.log(
		`  Quaternionic n=${qErrs.length.toString().padEnd(3)}  ` +
			`MAE=${mae(qErrs).toFixed(2).padStart(8)}  ` +
			`RMSE=${rmse(qErrs).toFixed(2).padStart(8)}  ` +
			`Bias=${mean(qErrs).toFixed(2).padStart(8)}`,
	);
}
