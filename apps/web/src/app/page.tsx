import Link from "next/link";

import { getDocManifest, getGroupedDocs } from "@/lib/docs";

export default async function HomePage() {
  const docs = await getDocManifest();
  const groups = await getGroupedDocs();
  const featured = docs.filter((doc) => doc.publicAccess).slice(0, 6);

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-12 px-6 py-14">
      <section className="grid gap-8 rounded-[2rem] border border-slate-200/80 bg-white/80 p-10 shadow-xl shadow-slate-200/40 backdrop-blur dark:border-slate-800 dark:bg-slate-950/70 dark:shadow-none lg:grid-cols-[1.2fr_0.8fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-teal-700 dark:text-teal-300">Paid Knowledge Base</p>
          <h1 className="mt-4 max-w-3xl text-5xl font-semibold tracking-tight text-slate-950 dark:text-white">
            把现有文档站升级成可试读、可付费、可持续运营的产品。
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600 dark:text-slate-300">
            首页和大纲公开，模块总览公开，每个主模块开放前几章试读。剩余内容只在服务端按付费权限返回全文。
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/pricing"
              className="rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400"
            >
              查看定价
            </Link>
            <Link
              href="/docs/modules/01_foundation_rl/01_foundation_rl"
              className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-white dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
            >
              进入试读
            </Link>
          </div>
        </div>
        <div className="grid gap-4">
          <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 dark:border-slate-800 dark:bg-slate-900">
            <p className="text-sm uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Current Inventory</p>
            <p className="mt-3 text-4xl font-semibold text-slate-950 dark:text-white">{docs.length}</p>
            <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">篇文章已接入权限控制站点。</p>
          </div>
          <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 dark:border-slate-800 dark:bg-slate-900">
            <p className="text-sm uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Access Strategy</p>
            <p className="mt-3 text-base leading-7 text-slate-700 dark:text-slate-300">
              Guides、首页和模块总览公开；每个模块额外开放前两篇章节；剩余内容通过订阅解锁。
            </p>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-end justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-amber-700 dark:text-amber-300">Featured Free Reads</p>
            <h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">公开可读内容</h2>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {featured.map((doc) => (
            <Link
              key={doc.href}
              href={doc.href}
              className="rounded-3xl border border-slate-200 bg-white/80 p-6 shadow-sm transition hover:-translate-y-1 hover:shadow-lg dark:border-slate-800 dark:bg-slate-950/70"
            >
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-teal-700 dark:text-teal-300">{doc.groupLabel}</p>
              <h3 className="mt-3 text-xl font-semibold text-slate-950 dark:text-white">{doc.title}</h3>
              <p className="mt-3 text-sm leading-7 text-slate-600 dark:text-slate-300">{doc.summary}</p>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-5">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">Module Index</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950 dark:text-white">按模块浏览</h2>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {Array.from(groups.entries()).map(([groupKey, groupDocs]) => {
            if (groupKey === "home") {
              return null;
            }

            return (
              <section key={groupKey} className="rounded-3xl border border-slate-200 bg-white/80 p-6 dark:border-slate-800 dark:bg-slate-950/70">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-xl font-semibold text-slate-950 dark:text-white">{groupDocs[0]?.groupLabel ?? groupKey}</h3>
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{groupDocs.length} pages</p>
                  </div>
                  <Link
                    href={groupDocs[0]?.href ?? "/"}
                    className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
                  >
                    查看
                  </Link>
                </div>
                <ul className="space-y-3">
                  {groupDocs.slice(0, 5).map((doc) => (
                    <li key={doc.href} className="rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-900">
                      <Link href={doc.href} className="font-medium text-slate-900 dark:text-slate-100">
                        {doc.title}
                      </Link>
                      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{doc.publicAccess ? "公开试读" : "付费解锁"}</p>
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      </section>
    </main>
  );
}
