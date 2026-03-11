import Link from "next/link";

import type { Session } from "next-auth";

export function SiteHeader({ session }: { session: Session | null }) {
  return (
    <header className="border-b border-slate-200 bg-white/85 backdrop-blur dark:border-slate-800 dark:bg-slate-950/85">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-6 py-4">
        <div>
          <Link href="/" className="text-lg font-semibold tracking-tight text-slate-950 dark:text-slate-50">
            LLM-Core Academy
          </Link>
          <p className="text-sm text-slate-500 dark:text-slate-400">Preview-first paid documentation site</p>
        </div>
        <nav className="flex items-center gap-3 text-sm font-medium">
          <Link href="/" className="rounded-full px-4 py-2 text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-900">
            Home
          </Link>
          <Link href="/pricing" className="rounded-full px-4 py-2 text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-900">
            Pricing
          </Link>
          <Link href="/docs/modules/01_foundation_rl/01_foundation_rl" className="rounded-full px-4 py-2 text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-900">
            Docs
          </Link>
          <Link
            href={session ? "/account" : "/sign-in"}
            className="rounded-full bg-slate-950 px-4 py-2 text-white hover:bg-slate-800 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400"
          >
            {session ? "Account" : "Sign in"}
          </Link>
        </nav>
      </div>
    </header>
  );
}
