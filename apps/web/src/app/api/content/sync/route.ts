import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

import { clearDocManifestCache } from "@/lib/docs";
import { env } from "@/lib/env";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  if (authHeader !== `Bearer ${env.CONTENT_SYNC_TOKEN}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  clearDocManifestCache();
  revalidatePath("/", "layout");
  revalidatePath("/pricing");
  revalidatePath("/account");
  revalidatePath("/docs/[...slug]", "page");

  return NextResponse.json({ ok: true, message: "Content cache cleared and routes revalidated." });
}
