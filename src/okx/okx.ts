const BASE_URL = "https://www.okx.com";

export type Candle = {
	date: string;     // YYYY-MM-DD
	timestamp: number;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
};

export type FundingRate = {
	timestamp: number;
	fundingRate: number;
};

// deno-lint-ignore no-explicit-any
const okxGet = async (path: string, params: Record<string, string> = {}): Promise<any[]> => {
	const url = new URL(`${BASE_URL}${path}`);
	for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
	const res = await fetch(url.toString());
	if (!res.ok) throw new Error(`OKX HTTP ${res.status}`);
	const json = await res.json();
	if (json.code !== "0") throw new Error(`OKX: ${json.msg}`);
	return json.data;
};

/** Daily spot candles for instId (e.g. "BTC-USDT"). Max 300 per request; paginates automatically. */
export const candles = async (instId: string, days: number): Promise<Candle[]> => {
	const results: Candle[] = [];
	let after: string | undefined;

	while (results.length < days) {
		const limit = Math.min(300, days - results.length);
		const params: Record<string, string> = { instId, bar: "1D", limit: String(limit) };
		if (after !== undefined) params.after = after;

		const batch = await okxGet("/api/v5/market/candles", params);
		if (batch.length === 0) break;

		for (const row of batch) {
			const ts = Number(row[0]);
			results.push({
				date: new Date(ts).toISOString().slice(0, 10),
				timestamp: ts,
				open: Number(row[1]),
				high: Number(row[2]),
				low: Number(row[3]),
				close: Number(row[4]),
				volume: Number(row[5]),
			});
		}

		after = String(batch[batch.length - 1][0]);
		if (batch.length < limit) break;
	}

	return results;
};

/**
 * Historical funding rates for a swap instId (e.g. "BTC-USDT-SWAP").
 * Returns one entry per settlement (every 8h). Paginates to cover `days` of history.
 */
export const fundingRates = async (instId: string, days: number): Promise<FundingRate[]> => {
	const results: FundingRate[] = [];
	let after: string | undefined;
	const target = days * 3; // 3 settlements per day

	while (results.length < target) {
		const params: Record<string, string> = { instId, limit: "100" };
		if (after !== undefined) params.after = after;

		const batch = await okxGet("/api/v5/public/funding-rate-history", params);
		if (batch.length === 0) break;

		for (const row of batch) {
			results.push({
				timestamp: Number(row.fundingTime),
				fundingRate: Number(row.realizedRate),
			});
		}

		after = String(batch[batch.length - 1].fundingTime);
		if (batch.length < 100) break;
	}

	return results;
};
