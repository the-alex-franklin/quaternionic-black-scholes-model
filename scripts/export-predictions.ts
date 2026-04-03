/**
 * Prediction Store
 *
 * Persistent NDJSON log of directional probability predictions.
 * Each record captures the classical N(d₂) vs quaternionic Nd₂.t for
 * a live option, keyed by expiry date so evaluate.ts can resolve ITM/OTM
 * after settlement.
 *
 * File: predictions.ndjson in the project root (one JSON object per line).
 */

export type Prediction = {
	/** ISO timestamp when the snapshot was taken */
	recordedAt: string;
	/** "BTC" | "ETH" */
	ticker: string;
	strike: number;
	/** "YYYY-MM-DD" — matches Deribit delivery_prices date format */
	expiryDate: string;
	/** Unix seconds */
	expiryTs: number;
	optionType: "call" | "put";
	/** Spot price at time of recording */
	spot: number;
	/** Years to expiry at time of recording */
	tte: number;
	/** N(d₂) from scalar Black-Scholes using calibrated ATM vol */
	classicalProb: number;
	/** Nd₂.t from quaternionic Black-Scholes */
	quatProb: number;
	/** Deribit settlement (index) price at expiry — filled by evaluate.ts */
	settlement?: number;
	/** True if option expired in-the-money — filled by evaluate.ts */
	itm?: boolean;
};

const STORE = "predictions.ndjson";

export const loadPredictions = async (): Promise<Prediction[]> => {
	try {
		const text = await Deno.readTextFile(STORE);
		return text
			.trim()
			.split("\n")
			.filter(Boolean)
			.map((line) => JSON.parse(line) as Prediction);
	} catch (e) {
		if (e instanceof Deno.errors.NotFound) return [];
		throw e;
	}
};

export const savePredictions = async (predictions: Prediction[]): Promise<void> => {
	const text = predictions.map((p) => JSON.stringify(p)).join("\n") + "\n";
	await Deno.writeTextFile(STORE, text);
};

export const appendPredictions = async (predictions: Prediction[]): Promise<void> => {
	if (predictions.length === 0) return;
	const text = predictions.map((p) => JSON.stringify(p)).join("\n") + "\n";
	await Deno.writeTextFile(STORE, text, { append: true });
};
