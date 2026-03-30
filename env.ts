import { z } from "zod";
import { loadSync } from "dotenv";

loadSync({ export: true });

export const env = z.object({
	POLYGON_API_KEY: z.string(),
}).parse(Deno.env.toObject());
