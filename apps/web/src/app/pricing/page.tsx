import { CheckoutForm } from "@/components/checkout-form";
import { env } from "@/lib/env";
import { siteConfig } from "@/lib/site-config";

export default function PricingPage() {
  const stripeReady = Boolean(env.STRIPE_SECRET_KEY && env.STRIPE_PRO_PRICE_ID);
  const plan = siteConfig.plans.pro;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-10 px-6 py-14">
      <section className="max-w-2xl">
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-700 dark:text-teal-300">Pricing</p>
        <h1 className="mt-3 text-5xl font-semibold tracking-tight text-slate-950 dark:text-white">为真实收费而设计，而不是前端假付费墙。</h1>
        <p className="mt-5 text-lg leading-8 text-slate-600 dark:text-slate-300">
          付费后服务端才返回全文。未购买用户即使打开页面，也只能拿到试读内容，不会拿到完整 HTML 源文。
        </p>
      </section>

      <section className="rounded-[2rem] border border-slate-200 bg-white/80 p-8 shadow-lg dark:border-slate-800 dark:bg-slate-950/70">
        <div className="grid gap-8 lg:grid-cols-[1fr_0.8fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-amber-700 dark:text-amber-300">{plan.name}</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950 dark:text-white">{plan.priceLabel}</h2>
            <p className="mt-4 max-w-xl text-base leading-8 text-slate-600 dark:text-slate-300">{plan.description}</p>
            <ul className="mt-6 space-y-3">
              {plan.features.map((feature) => (
                <li key={feature} className="rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-700 dark:bg-slate-900 dark:text-slate-300">
                  {feature}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 dark:border-slate-800 dark:bg-slate-900">
            <h3 className="text-xl font-semibold text-slate-950 dark:text-white">Checkout wiring</h3>
            <p className="mt-3 text-sm leading-7 text-slate-600 dark:text-slate-300">
              页面提交到 <code>/api/checkout</code>，服务端创建 Stripe Checkout Session，支付完成后由 webhook 更新用户权限。
            </p>
            <div className="mt-6">
              <CheckoutForm disabled={!stripeReady} />
            </div>
            {!stripeReady ? (
              <p className="mt-4 text-sm text-amber-700 dark:text-amber-300">
                当前还没有配置 Stripe 环境变量，所以这里只是骨架，不会真的发起支付。
              </p>
            ) : null}
          </div>
        </div>
      </section>
    </main>
  );
}
