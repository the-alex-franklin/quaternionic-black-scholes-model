import { candles, fundingRates } from "../src/okx/okx.ts";

const days = Number(Deno.args[0] ?? "300");

console.log(`Fetching ${days} days of BTC data from OKX...`);

const [candleData, rateData] = await Promise.all([
	candles("BTC-USDT", days),
	fundingRates("BTC-USDT-SWAP", days),
]);

// Aggregate 8h funding rates to daily mean
const rateByDate = new Map<string, number[]>();
for (const r of rateData) {
	const date = new Date(r.timestamp).toISOString().slice(0, 10);
	const bucket = rateByDate.get(date) ?? [];
	bucket.push(r.fundingRate);
	rateByDate.set(date, bucket);
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export type DailyRow = {
	date: string;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	fundingRate: number | null;
};

const rows: DailyRow[] = candleData.map((c) => {
	const rates = rateByDate.get(c.date);
	return {
		date: c.date,
		open: c.open,
		high: c.high,
		low: c.low,
		close: c.close,
		volume: c.volume,
		fundingRate: rates !== undefined ? mean(rates) : null,
	};
});

await Deno.mkdir("data", { recursive: true });
const outPath = "data/BTC_daily.ndjson";
await Deno.writeTextFile(outPath, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");

const withRates = rows.filter((r) => r.fundingRate !== null).length;
console.log(`Wrote ${rows.length} rows → ${outPath} (${withRates} with funding rate)`);
