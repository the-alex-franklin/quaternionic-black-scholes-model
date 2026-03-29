/**
 * Market Data Layer — Yahoo Finance
 *
 * Fetches historical OHLCV prices and live options chains for SPY, SPX, AAPL
 * (or any ticker). All network I/O is isolated here; backtest.ts is pure.
 *
 * Require: --allow-net when running scripts that call these functions.
 * Tests use fixture snapshots and do not call these functions directly.
 */

import axios from "axios";
import { z } from "zod";

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
	iv: number;                 // Yahoo's implied vol estimate (annualised)
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
};

// ---------------------------------------------------------------------------
// Yahoo Finance — chart (historical prices)
// ---------------------------------------------------------------------------

const YahooQuoteZ = z.object({
	open: z.array(z.number().nullable()),
	high: z.array(z.number().nullable()),
	low: z.array(z.number().nullable()),
	close: z.array(z.number().nullable()),
	volume: z.array(z.number().nullable()),
});

const YahooAdjCloseZ = z.object({
	adjclose: z.array(z.number().nullable()),
});

const YahooChartResultZ = z.object({
	timestamp: z.array(z.number()),
	indicators: z.object({
		quote: z.array(YahooQuoteZ),
		adjclose: z.array(YahooAdjCloseZ).optional(),
	}),
});

const YahooChartZ = z.object({
	chart: z.object({
		result: z.array(YahooChartResultZ),
	}),
});

/**
 * Fetch daily OHLCV history for `ticker` between `from` and `to`.
 *
 * Requires Deno --allow-net flag.
 */
export const fetchHistoricalPrices = async (
	ticker: string,
	from: Date,
	to: Date,
): Promise<OHLCV[]> => {
	const p1 = Math.floor(from.getTime() / 1000);
	const p2 = Math.floor(to.getTime() / 1000);
	const url =
		`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${p1}&period2=${p2}&interval=1d&events=history`;

	const res = await axios.get(url, {
		headers: { "User-Agent": "Mozilla/5.0" },
	});
	const parsed = YahooChartZ.parse(res.data);
	const result = parsed.chart.result[0];
	if (!result) throw new Error(`No chart data for ${ticker}`);

	const { timestamp, indicators } = result;
	const quote = indicators.quote[0];
	if (!quote) throw new Error(`No quote data for ${ticker}`);
	const adjClose = indicators.adjclose?.[0]?.adjclose ?? null;

	const bars: OHLCV[] = [];
	for (let i = 0; i < timestamp.length; i++) {
		const o = quote.open[i];
		const h = quote.high[i];
		const l = quote.low[i];
		const c = quote.close[i];
		const v = quote.volume[i];
		const ac = adjClose ? (adjClose[i] ?? c) : c;
		if (o == null || h == null || l == null || c == null || v == null) continue;
		bars.push({
			date: new Date((timestamp[i] as number) * 1000),
			open: o,
			high: h,
			low: l,
			close: c,
			adjClose: ac ?? c,
			volume: v,
		});
	}
	return bars;
};

// ---------------------------------------------------------------------------
// Yahoo Finance — options chain
// ---------------------------------------------------------------------------

const YahooOptionItemZ = z.object({
	strike: z.number(),
	bid: z.number(),
	ask: z.number(),
	lastPrice: z.number(),
	impliedVolatility: z.number(),
	volume: z.number().optional().default(0),
	openInterest: z.number().optional().default(0),
	expiration: z.number(),
});

const YahooOptionsZ = z.object({
	optionChain: z.object({
		result: z.array(z.object({
			quote: z.object({ regularMarketPrice: z.number() }),
			options: z.array(z.object({
				expirationDate: z.number(),
				calls: z.array(YahooOptionItemZ),
				puts: z.array(YahooOptionItemZ),
			})),
		})),
	}),
});

const parseOptionItem = (
	item: z.infer<typeof YahooOptionItemZ>,
	type: "call" | "put",
): OptionQuote => ({
	strike: item.strike,
	expiry: new Date(item.expiration * 1000),
	expiryTs: item.expiration,
	bid: item.bid,
	ask: item.ask,
	mid: (item.bid + item.ask) / 2,
	iv: item.impliedVolatility,
	volume: item.volume,
	openInterest: item.openInterest,
	type,
});

/**
 * Fetch a live options chain snapshot for `ticker`.
 *
 * Pass `expiryTs` (Unix seconds) to pin a specific expiry;
 * omit to use the nearest available expiry.
 *
 * Requires Deno --allow-net flag.
 */
export const fetchOptionsChain = async (
	ticker: string,
	expiryTs?: number,
): Promise<MarketSnapshot> => {
	const url = expiryTs
		? `https://query2.finance.yahoo.com/v7/finance/options/${ticker}?date=${expiryTs}`
		: `https://query2.finance.yahoo.com/v7/finance/options/${ticker}`;

	const res = await axios.get(url, {
		headers: { "User-Agent": "Mozilla/5.0" },
	});
	const parsed = YahooOptionsZ.parse(res.data);
	const result = parsed.optionChain.result[0];
	if (!result) throw new Error(`No options data for ${ticker}`);

	const spot = result.quote.regularMarketPrice;
	const chain = result.options[0];
	if (!chain) throw new Error(`No option expiry data for ${ticker}`);

	return {
		ticker,
		asOf: new Date(),
		spot,
		calls: chain.calls.map((c) => parseOptionItem(c, "call")),
		puts: chain.puts.map((p) => parseOptionItem(p, "put")),
	};
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
