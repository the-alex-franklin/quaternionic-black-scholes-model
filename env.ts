import { load } from "dotenv";
import z from "zod";

await load({ export: true });

export const env = z.object({

}).parse(Deno.env.toObject());