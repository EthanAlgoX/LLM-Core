import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    ok: true,
    service: "llm-core-paid-site",
    timestamp: new Date().toISOString(),
  });
}
