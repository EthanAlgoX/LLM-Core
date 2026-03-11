"use client";

import { signIn } from "next-auth/react";

type ProviderItem = {
  id: string;
  name: string;
};

export function SignInProviderList({ providers }: { providers: ProviderItem[] }) {
  if (!providers.length) {
    return (
      <div className="rounded-2xl border border-amber-300 bg-amber-50 p-5 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
        当前还没有配置 OAuth Provider。先在 <code>.env.local</code> 中补上 GitHub 或 Google 凭据。
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {providers.map((provider) => (
        <button
          key={provider.id}
          type="button"
          onClick={() => void signIn(provider.id, { callbackUrl: "/account" })}
          className="flex min-h-12 w-full items-center justify-center rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-900 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-50 dark:hover:bg-slate-800"
        >
          Continue with {provider.name}
        </button>
      ))}
    </div>
  );
}
