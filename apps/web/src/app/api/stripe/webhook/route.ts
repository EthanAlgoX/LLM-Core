import { NextResponse } from "next/server";

import { handleStripeEvent } from "@/lib/billing";

export async function POST(request: Request) {
  const signature = request.headers.get("stripe-signature");
  if (!signature) {
    return NextResponse.json({ error: "Missing stripe-signature header." }, { status: 400 });
  }

  const rawBody = await request.text();
  await handleStripeEvent(rawBody, signature);
  return NextResponse.json({ received: true });
}
