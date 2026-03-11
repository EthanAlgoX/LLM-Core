import Link from "next/link";

export function DocPaywall({ title }: { title: string }) {
  return (
    <section className="rounded-3xl border border-amber-300 bg-amber-50 p-8 shadow-sm dark:border-amber-500/30 dark:bg-amber-500/10">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-amber-700 dark:text-amber-300">Premium Content</p>
      <h3 className="mt-3 text-2xl font-semibold text-slate-950 dark:text-slate-50">Unlock the rest of “{title}”</h3>
      <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-700 dark:text-slate-300">
        你当前看到的是公开试读部分。订阅后可解锁完整章节、后续新增内容，以及全部工程细节说明。
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link
          href="/pricing"
          className="rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400"
        >
          View pricing
        </Link>
        <Link
          href="/sign-in"
          className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-white dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
        >
          Sign in
        </Link>
      </div>
    </section>
  );
}
