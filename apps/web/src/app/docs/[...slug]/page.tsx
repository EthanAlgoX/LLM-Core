import Link from "next/link";
import { notFound } from "next/navigation";

import { auth } from "@/lib/auth";
import { canReadFullDoc } from "@/lib/access";
import { getDocBySlug } from "@/lib/docs";
import { DocPaywall } from "@/components/doc-paywall";

type PageProps = {
  params: Promise<{ slug: string[] }>;
};

export default async function DocPage({ params }: PageProps) {
  const { slug } = await params;
  const doc = await getDocBySlug(slug);
  if (!doc) {
    notFound();
  }

  const session = await auth();
  const fullAccess = canReadFullDoc(session, doc);
  const html = fullAccess ? doc.bodyHtml : doc.previewHtml;

  return (
    <main className="mx-auto flex w-full max-w-7xl gap-10 px-6 py-12">
      <article className="min-w-0 flex-1">
        <div className="rounded-[2rem] border border-slate-200 bg-white/85 p-8 shadow-lg dark:border-slate-800 dark:bg-slate-950/70">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full bg-teal-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-teal-800 dark:bg-teal-500/15 dark:text-teal-300">
              {doc.groupLabel}
            </span>
            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] ${
                doc.publicAccess
                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300"
                  : "bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-300"
              }`}
            >
              {fullAccess ? "Unlocked" : doc.publicAccess ? "Free Preview" : "Locked after preview"}
            </span>
          </div>

          <h1 className="mt-5 text-4xl font-semibold tracking-tight text-slate-950 dark:text-white">{doc.title}</h1>
          <p className="mt-4 max-w-3xl text-base leading-8 text-slate-600 dark:text-slate-300">{doc.summary}</p>

          <div className={`prose-doc mt-10 ${fullAccess ? "" : "preview-mask"}`} dangerouslySetInnerHTML={{ __html: html }} />

          {!fullAccess ? <div className="mt-10"><DocPaywall title={doc.title} /></div> : null}

          <div className="mt-10 flex gap-3">
            <Link
              href="/pricing"
              className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
            >
              查看付费方案
            </Link>
            <Link
              href="/"
              className="rounded-full border border-slate-300 px-5 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
            >
              返回首页
            </Link>
          </div>
        </div>
      </article>

      <aside className="hidden w-80 shrink-0 lg:block">
        <div className="sticky top-24 rounded-[2rem] border border-slate-200 bg-white/85 p-6 shadow-lg dark:border-slate-800 dark:bg-slate-950/70">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">On this page</p>
          <ol className="mt-4 space-y-3 text-sm">
            {doc.outline.map((item) => (
              <li key={item.id} className={item.level === 3 ? "pl-4 text-slate-500 dark:text-slate-400" : ""}>
                <a href={`#${item.id}`} className="hover:text-teal-700 dark:hover:text-teal-300">
                  {item.text}
                </a>
              </li>
            ))}
          </ol>
        </div>
      </aside>
    </main>
  );
}
