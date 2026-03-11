import Link from "next/link";
import { redirect } from "next/navigation";

import { auth } from "@/lib/auth";
import { hasPaidAccess } from "@/lib/access";

export default async function AccountPage() {
  const session = await auth();
  if (!session?.user) {
    redirect("/sign-in");
  }

  const paid = hasPaidAccess(session);

  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-6 py-14">
      <section className="rounded-[2rem] border border-slate-200 bg-white/80 p-8 shadow-lg dark:border-slate-800 dark:bg-slate-950/70">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-700 dark:text-teal-300">Account</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950 dark:text-white">{session.user.email ?? session.user.name ?? "Signed-in user"}</h1>
        <dl className="mt-8 grid gap-4 sm:grid-cols-2">
          <div className="rounded-2xl bg-slate-50 p-5 dark:bg-slate-900">
            <dt className="text-sm text-slate-500 dark:text-slate-400">Membership plan</dt>
            <dd className="mt-2 text-lg font-semibold text-slate-950 dark:text-white">{session.user.membershipPlan}</dd>
          </div>
          <div className="rounded-2xl bg-slate-50 p-5 dark:bg-slate-900">
            <dt className="text-sm text-slate-500 dark:text-slate-400">Membership status</dt>
            <dd className="mt-2 text-lg font-semibold text-slate-950 dark:text-white">{session.user.membershipStatus}</dd>
          </div>
          <div className="rounded-2xl bg-slate-50 p-5 dark:bg-slate-900 sm:col-span-2">
            <dt className="text-sm text-slate-500 dark:text-slate-400">Effective access</dt>
            <dd className="mt-2 text-lg font-semibold text-slate-950 dark:text-white">{paid ? "Full paid access" : "Preview only"}</dd>
          </div>
        </dl>
        <div className="mt-8 flex gap-3">
          <Link
            href="/pricing"
            className="rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400"
          >
            Manage billing
          </Link>
          <Link
            href="/docs/modules/01_foundation_rl/01_foundation_rl"
            className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
          >
            Continue reading
          </Link>
        </div>
      </section>
    </main>
  );
}
