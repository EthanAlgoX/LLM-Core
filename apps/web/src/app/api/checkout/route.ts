import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { createCheckoutSession } from "@/lib/billing";

export async function POST() {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.redirect(new URL("/sign-in", process.env.NEXTAUTH_URL ?? "http://localhost:3000"), 303);
  }

  const checkout = await createCheckoutSession({
    userId: session.user.id,
    email: session.user.email,
  });

  if (!checkout.url) {
    return NextResponse.json({ error: "Stripe checkout session has no redirect URL." }, { status: 500 });
  }

  return NextResponse.redirect(checkout.url, 303);
}
