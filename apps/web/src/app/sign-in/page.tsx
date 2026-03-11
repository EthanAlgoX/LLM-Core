import { redirect } from "next/navigation";

import { SignInProviderList } from "@/components/sign-in-provider-list";
import { auth, getEnabledProviders } from "@/lib/auth";

export default async function SignInPage() {
  const session = await auth();
  if (session?.user) {
    redirect("/account");
  }

  const providers = getEnabledProviders();

  return (
    <main className="mx-auto flex min-h-[calc(100vh-96px)] w-full max-w-xl items-center px-6 py-14">
      <section className="w-full rounded-[2rem] border border-slate-200 bg-white/80 p-8 shadow-lg dark:border-slate-800 dark:bg-slate-950/70">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700 dark:text-teal-300">Sign in</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950 dark:text-white">登录后继续解锁付费内容</h1>
        <p className="mt-4 text-sm leading-7 text-slate-600 dark:text-slate-300">
          生产环境建议至少配置一个 OAuth Provider，并在 Stripe 支付完成后把购买结果绑定到当前用户。
        </p>
        <div className="mt-8">
          <SignInProviderList providers={providers} />
        </div>
      </section>
    </main>
  );
}
