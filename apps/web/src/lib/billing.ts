import type Stripe from "stripe";

import { db } from "@/lib/db";
import { env } from "@/lib/env";

let stripeClient: Stripe | null = null;

async function getStripeClient() {
  if (!env.STRIPE_SECRET_KEY) {
    throw new Error("STRIPE_SECRET_KEY is not configured.");
  }

  if (stripeClient) {
    return stripeClient;
  }

  const { default: StripeClient } = await import("stripe");
  stripeClient = new StripeClient(env.STRIPE_SECRET_KEY, {
    apiVersion: "2025-02-24.acacia",
  });
  return stripeClient;
}

export async function createCheckoutSession(input: {
  userId: string;
  email?: string | null;
}) {
  if (!env.STRIPE_PRO_PRICE_ID) {
    throw new Error("STRIPE_PRO_PRICE_ID is not configured.");
  }

  const stripe = await getStripeClient();
  const session = await stripe.checkout.sessions.create({
    mode: "subscription",
    payment_method_types: ["card"],
    line_items: [
      {
        price: env.STRIPE_PRO_PRICE_ID,
        quantity: 1,
      },
    ],
    success_url: `${env.NEXT_PUBLIC_APP_URL}/account?checkout=success`,
    cancel_url: `${env.NEXT_PUBLIC_APP_URL}/pricing?checkout=canceled`,
    customer_email: input.email ?? undefined,
    metadata: {
      userId: input.userId,
    },
    allow_promotion_codes: true,
  });

  await db.purchase.create({
    data: {
      userId: input.userId,
      status: "PENDING",
      stripeCheckoutId: session.id,
      stripePriceId: env.STRIPE_PRO_PRICE_ID,
      currency: session.currency ?? undefined,
      amountSubtotal: session.amount_subtotal ?? undefined,
      amountTotal: session.amount_total ?? undefined,
    },
  });

  return session;
}

export async function handleStripeEvent(rawBody: string, signature: string) {
  if (!env.STRIPE_WEBHOOK_SECRET) {
    throw new Error("STRIPE_WEBHOOK_SECRET is not configured.");
  }

  const stripe = await getStripeClient();
  const event = stripe.webhooks.constructEvent(rawBody, signature, env.STRIPE_WEBHOOK_SECRET);

  const existing = await db.webhookEvent.findUnique({
    where: { stripeEventId: event.id },
  });
  if (existing) {
    return event;
  }

  await db.webhookEvent.create({
    data: {
      stripeEventId: event.id,
      type: event.type,
      payload: event as unknown as object,
    },
  });

  if (event.type === "checkout.session.completed") {
    const session = event.data.object;
    const userId = session.metadata?.userId;
    if (userId) {
      await db.purchase.updateMany({
        where: { stripeCheckoutId: session.id },
        data: {
          status: "PAID",
          paidAt: new Date(),
          stripePaymentIntentId:
            typeof session.payment_intent === "string" ? session.payment_intent : undefined,
          currency: session.currency ?? undefined,
          amountSubtotal: session.amount_subtotal ?? undefined,
          amountTotal: session.amount_total ?? undefined,
        },
      });

      await db.user.update({
        where: { id: userId },
        data: {
          membershipPlan: "PRO",
          membershipStatus: "ACTIVE",
          stripeCustomerId:
            typeof session.customer === "string" ? session.customer : undefined,
        },
      });
    }
  }

  if (event.type === "customer.subscription.updated" || event.type === "customer.subscription.created") {
    const subscription = event.data.object;
    if (typeof subscription.customer === "string") {
      await db.user.updateMany({
        where: { stripeCustomerId: subscription.customer },
        data: {
          membershipPlan: "PRO",
          membershipStatus:
            subscription.status === "trialing"
              ? "TRIALING"
              : subscription.status === "active"
                ? "ACTIVE"
                : subscription.status === "past_due"
                  ? "PAST_DUE"
                  : "CANCELED",
          membershipExpiresAt:
            subscription.current_period_end != null
              ? new Date(subscription.current_period_end * 1000)
              : null,
          stripeSubscriptionId: subscription.id,
        },
      });
    }
  }

  if (event.type === "customer.subscription.deleted") {
    const subscription = event.data.object;
    if (typeof subscription.customer === "string") {
      await db.user.updateMany({
        where: { stripeCustomerId: subscription.customer },
        data: {
          membershipStatus: "CANCELED",
          membershipExpiresAt:
            subscription.current_period_end != null
              ? new Date(subscription.current_period_end * 1000)
              : null,
        },
      });
    }
  }

  return event;
}
