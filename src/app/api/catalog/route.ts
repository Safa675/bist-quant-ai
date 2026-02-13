import { NextResponse } from "next/server";
import { getUnifiedCatalog } from "@/lib/server/catalogService";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
    try {
        const url = new URL(request.url);
        const refresh = url.searchParams.get("refresh") === "1";
        const catalog = await getUnifiedCatalog({ forceRefresh: refresh });
        return NextResponse.json(catalog, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to load unified catalog.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
