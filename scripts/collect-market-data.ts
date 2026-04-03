/**
 * Market Data Layer — Deribit
 *
 * Deribit: live BTC/ETH options chains. No API key needed — all public endpoints.
 *   Use fetchDeribitOptionsChain() to get a MarketSnapshot ready for runBacktest().
 *
 * Tests use fixture snapshots and do not call these functions directly.
 */

import axios from "axios";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

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
	/** 8-hour perpetual funding rate (signed, e.g. 0.0001 = 0.01%). */
	funding8h?: number;
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
