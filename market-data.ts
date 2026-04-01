/**
 * Market Data Layer — Polygon.io + Deribit
 *
 * Polygon: historical OHLCV and options chains for equities/indices.
 *   Requires POLYGON_API_KEY env var, --allow-net --allow-env.
 *
 * Deribit: live BTC/ETH options chains. No API key needed — all public endpoints.
 *   Use fetchDeribitOptionsChain() to get a MarketSnapshot ready for runBacktest().
 *
 * Tests use fixture snapshots and do not call these functions directly.
 */

import axios from "axios";
import { z } from "zod";
import { env } from "./env.ts";

const apiKey = () => env.POLYGON_API_KEY;
const BASE = "https://api.polygon.io";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export type OHLCV = {
	date: Date;
	open: number;
	high: number;
	low: number;
	close: number;
	adjClose: number;
	volume: number;
};

export type OptionQuote = {
	strike: number;
	expiry: Date;
	expiryTs: number;           // Unix seconds
	bid: number;
	ask: number;
	mid: number;
	iv: number;                 // Implied vol (annualised)
	volume: number;
	openInterest: number;
	type: "call" | "put";
};

export type MarketSnapshot = {
	ticker: string;
	asOf: Date;
	spot: number;
	calls: OptionQuote[];
	puts: OptionQuote[];
	/** 8-hour perpetual funding rate (signed, e.g. 0.0001 = 0.01%). Undefined for equities. */
	funding8h?: number;
};

// ---------------------------------------------------------------------------
// Polygon — historical prices (v2/aggs)
// ---------------------------------------------------------------------------

const PolygonAggsZ = z.object({
	results: z.array(z.object({
		t: z.number(),
		o: z.number(),
		h: z.number(),
		l: z.number(),
		c: z.number(),
		v: z.number(),
	})).optional().default([]),
	status: z.string(),
});

/**
 * Fetch daily OHLCV history for `ticker` between `from` and `to`.
 */
export const fetchHistoricalPrices = async (
	ticker: string,
	from: Date,
	to: Date,
): Promise<OHLCV[]> => {
	const fmt = (d: Date) => d.toISOString().slice(0, 10);
	const url = `${BASE}/v2/aggs/ticker/${ticker}/range/1/day/${fmt(from)}/${fmt(to)}?adjusted=true&sort=asc&limit=5000&apiKey=${apiKey()}`;
	const res = await axios.get(url);
	const parsed = PolygonAggsZ.parse(res.data);
	return parsed.results.map((r) => ({
		date: new Date(r.t),
		open: r.o,
		high: r.h,
		low: r.l,
		close: r.c,
		adjClose: r.c,
		volume: r.v,
	}));
};

// ---------------------------------------------------------------------------
// Polygon — spot price
// ---------------------------------------------------------------------------

const fetchSpot = async (ticker: string): Promise<number> => {
	// Use latest daily close bar — works on free tier for equities and indices
	const polygonTicker = ticker === "SPX" ? "I:SPX" : ticker;
	const to = new Date();
	const from = new Date(to.getTime() - 7 * 24 * 60 * 60 * 1000);
	const bars = await fetchHistoricalPrices(polygonTicker, from, to);
	const last = bars.at(-1);
	if (!last) throw new Error(`No price data for ${ticker}`);
	return last.close;
};

// ---------------------------------------------------------------------------
// Polygon — options snapshot (v3/snapshot/options)
// ---------------------------------------------------------------------------
//
// The options snapshot endpoint requires a paid plan. On the free tier we
// enumerate contracts via /v3/reference/options/contracts (free) and then
// fetch the previous day's OHLCV close per contract via /v2/aggs (free).
// The close price is used as the market mid; we derive IV ourselves via
// the impliedVol() inverter in bs-model.ts.
// ---------------------------------------------------------------------------

import { impliedVol } from "./bs-model.ts";

const PolygonContractZ = z.object({
	ticker: z.string(),
	contract_type: z.enum(["call", "put"]),
	strike_price: z.number(),
	expiration_date: z.string(),
});

const PolygonContractsPageZ = z.object({
	results: z.array(PolygonContractZ).optional().default([]),
	next_url: z.string().optional(),
	status: z.string(),
});

/** Enumerate all option contract tickers for an underlying within a date range */
const listContracts = async (
	underlying: string,
	fromDate: string,
	toDate: string,
): Promise<z.infer<typeof PolygonContractZ>[]> => {
	const acc: z.infer<typeof PolygonContractZ>[] = [];
	let next: string | undefined =
		`${BASE}/v3/reference/options/contracts` +
		`?underlying_ticker=${underlying}` +
		`&expiration_date.gte=${fromDate}` +
		`&expiration_date.lte=${toDate}` +
		`&limit=250`;

	while (next) {
		const sep = next.includes("?") ? "&" : "?";
		const res = await axios.get(`${next}${sep}apiKey=${apiKey()}`);
		const parsed = PolygonContractsPageZ.parse(res.data);
		acc.push(...parsed.results);
		next = parsed.next_url;
		if (acc.length >= 1000) break;
	}
	return acc;
};

/** Fetch the most recent daily close for a single option contract ticker */
const fetchContractClose = async (
	ticker: string,
	asOf: string,
): Promise<number | null> => {
	// Look back 5 calendar days to catch the last trading day
	const from = new Date(asOf);
	from.setDate(from.getDate() - 5);
	const url = `${BASE}/v2/aggs/ticker/${ticker}/range/1/day/${from.toISOString().slice(0, 10)}/${asOf}` +
		`?adjusted=true&sort=desc&limit=1&apiKey=${apiKey()}`;
	const res = await axios.get(url);
	const parsed = PolygonAggsZ.parse(res.data);
	return parsed.results[0]?.c ?? null;
};

/**
 * Fetch an options chain snapshot for `ticker` using prior-day closing prices.
 * Covers options expiring within the next 45 days by default.
 * Pass `expiryDate` as "YYYY-MM-DD" to pin a specific expiry.
 */
export const fetchOptionsChain = async (
	ticker: string,
	expiryDate?: string,
): Promise<MarketSnapshot> => {
	const spot = await fetchSpot(ticker);
	const underlying = ticker === "SPX" ? "SPXW" : ticker;
	const today = new Date().toISOString().slice(0, 10);
	const maxExpiry = expiryDate ?? (() => {
		const d = new Date();
		d.setDate(d.getDate() + 45);
		return d.toISOString().slice(0, 10);
	})();

	const contracts = await listContracts(underlying, today, maxExpiry);

	// Focus on near-ATM strikes (±20%) to limit API calls
	const atmFilter = (k: number) => Math.abs(k / spot - 1) < 0.20;
	const liquid = contracts.filter((c) => atmFilter(c.strike_price));

	// Fetch closes in small concurrent batches of 10
	const calls: OptionQuote[] = [];
	const puts: OptionQuote[] = [];

	const BATCH = 10;
	for (let i = 0; i < liquid.length; i += BATCH) {
		const batch = liquid.slice(i, i + BATCH);
		const closes = await Promise.all(
			batch.map((c) => fetchContractClose(c.ticker, today)),
		);

		for (let j = 0; j < batch.length; j++) {
			const c = batch[j]!;
			const close = closes[j];
			if (close == null || close <= 0) continue;

			const expiry = new Date(c.expiration_date + "T21:00:00Z");
			const T = Math.max(
				(expiry.getTime() - Date.now()) / (1000 * 60 * 60 * 24 * 365.25),
				1 / 365,
			);
			const expiryTs = Math.floor(expiry.getTime() / 1000);

			// Derive IV from the close price using our BS inverter
			let iv = 0;
			try {
				iv = impliedVol(close, {
					spot: { t: spot, p: 0, f: 0, l: 0 },
					strike: c.strike_price,
					expiry: T,
					rate: 0.0525,
					vol: { t: 0.20, p: 0, f: 0, l: 0 },
				});
			} catch {
				iv = 0;
			}

			const quote: OptionQuote = {
				strike: c.strike_price,
				expiry,
				expiryTs,
				bid: close * 0.99,   // synthetic spread: ±1%
				ask: close * 1.01,
				mid: close,
				iv,
				volume: 1,
				openInterest: 1,
				type: c.contract_type,
			};

			if (c.contract_type === "call") calls.push(quote);
			else puts.push(quote);
		}
	}

	return { ticker, asOf: new Date(), spot, calls, puts };
};

// ---------------------------------------------------------------------------
// Historical volatility estimate
// ---------------------------------------------------------------------------

/**
 * Annualised close-to-close historical volatility over the last `windowDays`
 * bars of `prices`.
 *
 * Uses log returns: σ = std(log(S_t / S_{t-1})) × √252
 */
export const historicalVol = (prices: OHLCV[], windowDays = 30): number => {
	const slice = prices.slice(-windowDays - 1);
	if (slice.length < 2) throw new Error("historicalVol: insufficient price history");

	const logReturns: number[] = [];
	for (let i = 1; i < slice.length; i++) {
		const prev = slice[i - 1];
		const curr = slice[i];
		if (!prev || !curr) continue;
		logReturns.push(Math.log(curr.adjClose / prev.adjClose));
	}

	const n = logReturns.length;
	const mean = logReturns.reduce((s, r) => s + r, 0) / n;
	const variance = logReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / (n - 1);
	return Math.sqrt(variance * 252);
};

// ---------------------------------------------------------------------------
// Deribit — BTC / ETH options (public API, no key required)
// ---------------------------------------------------------------------------
//
// All Deribit prices are quoted in the underlying coin (BTC or ETH).
// Multiply by spot USD to get dollar-denominated values.
//
// Instrument name format: {CCY}-{DDMMMYY}-{STRIKE}-{C|P}
//   e.g.  BTC-28MAR25-95000-C
// ---------------------------------------------------------------------------

export type DeribitCurrency = "BTC" | "ETH";

const DERIBIT = "https://www.deribit.com/api/v2/public";

const DeribitIndexZ = z.object({
	result: z.object({ index_price: z.number() }),
});

const DeribitDeliveryZ = z.object({
	result: z.object({
		data: z.array(z.object({ date: z.string(), delivery_price: z.number() })),
	}),
});

const DeribitTickerZ = z.object({
	result: z.object({
		funding_8h: z.number(),
	}),
});

const DeribitBookSummaryZ = z.object({
	result: z.array(z.object({
		instrument_name: z.string(),
		bid_price: z.number().nullable(),
		ask_price: z.number().nullable(),
		mark_price: z.number().nullable(),
		mark_iv: z.number().nullable(),
		volume: z.number(),
		open_interest: z.number(),
		underlying_price: z.number().optional(),
	})),
});

/** Parse a Deribit instrument name into its components, or null if unrecognised */
const parseDeribitInstrument = (
	name: string,
): { strike: number; expiry: Date; expiryTs: number; type: "call" | "put" } | null => {
	// BTC-28MAR25-95000-C
	const parts = name.split("-");
	if (parts.length !== 4) return null;
	const [, dateStr, strikeStr, typeChar] = parts;
	if (!dateStr || !strikeStr || !typeChar) return null;

	const strike = Number(strikeStr);
	if (!isFinite(strike) || strike <= 0) return null;

	const type = typeChar === "C" ? "call" : typeChar === "P" ? "put" : null;
	if (!type) return null;

	// "28MAR25" → Date (treated as 08:00 UTC, standard Deribit expiry time)
	const expiry = new Date(`${dateStr.slice(0, 2)} ${dateStr.slice(2, 5)} 20${dateStr.slice(5)} 08:00:00 UTC`);
	if (isNaN(expiry.getTime())) return null;

	return { strike, expiry, expiryTs: Math.floor(expiry.getTime() / 1000), type };
};

/**
 * Fetch a live BTC or ETH options snapshot from Deribit.
 *
 * Returns a MarketSnapshot compatible with runBacktest().
 * Prices are converted from coin-denominated to USD using the Deribit index price.
 *
 * @param currency  "BTC" or "ETH"
 * @param expiryTs  Optional Unix-seconds timestamp to filter to a single expiry.
 *                  If omitted, includes all expiries (backtest will use the nearest one).
 */
export const fetchDeribitOptionsChain = async (
	currency: DeribitCurrency,
	expiryTs?: number,
): Promise<MarketSnapshot> => {
	const indexName = currency === "BTC" ? "btc_usd" : "eth_usd";
	const perpName = `${currency}-PERPETUAL`;

	const [indexRes, bookRes, tickerRes] = await Promise.all([
		axios.get(`${DERIBIT}/get_index_price?index_name=${indexName}`),
		axios.get(`${DERIBIT}/get_book_summary_by_currency?currency=${currency}&kind=option`),
		axios.get(`${DERIBIT}/ticker?instrument_name=${perpName}`),
	]);

	const spot = DeribitIndexZ.parse(indexRes.data).result.index_price;
	const summaries = DeribitBookSummaryZ.parse(bookRes.data).result;
	const funding8h = DeribitTickerZ.parse(tickerRes.data).result.funding_8h;

	const calls: OptionQuote[] = [];
	const puts: OptionQuote[] = [];

	for (const s of summaries) {
		const parsed = parseDeribitInstrument(s.instrument_name);
		if (!parsed) continue;
		if (expiryTs !== undefined && parsed.expiryTs !== expiryTs) continue;

		// Skip if no usable price
		const markCoin = s.mark_price;
		if (markCoin == null || markCoin <= 0) continue;

		// Convert coin → USD.
		// Deribit order books can contain stale resting orders far from the mark
		// (e.g. a 0.005 BTC bid on a 0.0001 BTC option). Clamp bid/ask to within
		// [mark/3, mark*3] so the spread doesn't blow up the liquidity calibration.
		const markUsd = markCoin * spot;
		const rawBid = s.bid_price != null && s.bid_price > 0 ? s.bid_price * spot : null;
		const rawAsk = s.ask_price != null && s.ask_price > 0 ? s.ask_price * spot : null;
		const bidUsd = rawBid != null && rawBid >= markUsd / 3 ? rawBid : markUsd * 0.99;
		const askUsd = rawAsk != null && rawAsk <= markUsd * 3 ? rawAsk : markUsd * 1.01;

		// Deribit provides mark IV directly (as a percentage, e.g. 65.3 = 65.3%)
		const iv = s.mark_iv != null && s.mark_iv > 0 ? s.mark_iv / 100 : 0;

		const quote: OptionQuote = {
			strike: parsed.strike,
			expiry: parsed.expiry,
			expiryTs: parsed.expiryTs,
			bid: bidUsd,
			ask: askUsd,
			mid: markUsd,
			iv,
			volume: s.volume,
			openInterest: s.open_interest,
			type: parsed.type,
		};

		if (parsed.type === "call") calls.push(quote);
		else puts.push(quote);
	}

	return { ticker: currency, asOf: new Date(), spot, calls, puts, funding8h };
};

/**
 * Fetch Deribit's historical settlement (delivery) price for a given currency
 * and date string ("YYYY-MM-DD"). Returns null if the date is not yet settled
 * or not found in the last 100 weekly expiries.
 */
export const fetchDeribitSettlement = async (
	currency: DeribitCurrency,
	date: string,
): Promise<number | null> => {
	const indexName = currency === "BTC" ? "btc_usd" : "eth_usd";
	const res = await axios.get(
		`${DERIBIT}/get_delivery_prices?index_name=${indexName}&count=100`,
	);
	const { data } = DeribitDeliveryZ.parse(res.data).result;
	const record = data.find((d) => d.date === date);
	return record?.delivery_price ?? null;
};
