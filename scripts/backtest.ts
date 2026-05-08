import { impliedVol, price, type BSParams } from "../src/bs-model/bs-model.ts";

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
	quatPrice: number | null;
	realizedPayoff: number;
	breakevenVol: number | null;
	classicalError: number;
	quatError: number | null;
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

	let quatPrice: number | null = null;
	if (row.fundingRate !== null) {
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
		quatPrice,
		realizedPayoff,
		breakevenVol,
		classicalError: classicalPrice - realizedPayoff,
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
	"Date        Close     RealVol   Classical  Quat      Realized   BEVol     CErr      QErr",
);
console.log("-".repeat(95));

for (const r of results) {
	console.log(
		`${r.date}  ` +
			`${pad(r.close.toFixed(0), 8)}  ` +
			`${pad((r.realizedVol * 100).toFixed(1) + "%", 7)}   ` +
			`${fmt(r.classicalPrice)}  ` +
			`${fmt(r.quatPrice)}  ` +
			`${fmt(r.realizedPayoff)}  ` +
			`${r.breakevenVol !== null ? pad((r.breakevenVol * 100).toFixed(1) + "%", 7) : pad("-", 7)}  ` +
			`${fmt(r.classicalError)}  ` +
			`${fmt(r.quatError)}`,
	);
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const mae = (xs: number[]) => mean(xs.map(Math.abs));
const rmse = (xs: number[]) => Math.sqrt(mean(xs.map((x) => x * x)));

const cErrs = results.map((r) => r.classicalError);
const qErrs = results.flatMap((r) => r.quatError !== null ? [r.quatError] : []);

console.log("\nSummary:");
console.log(
	`  Classical    n=${cErrs.length.toString().padEnd(3)}  ` +
		`MAE=${mae(cErrs).toFixed(2).padStart(8)}  ` +
		`RMSE=${rmse(cErrs).toFixed(2).padStart(8)}  ` +
		`Bias=${mean(cErrs).toFixed(2).padStart(8)}`,
);
if (qErrs.length > 0) {
	console.log(
		`  Quaternionic n=${qErrs.length.toString().padEnd(3)}  ` +
			`MAE=${mae(qErrs).toFixed(2).padStart(8)}  ` +
			`RMSE=${rmse(qErrs).toFixed(2).padStart(8)}  ` +
			`Bias=${mean(qErrs).toFixed(2).padStart(8)}`,
	);
}
