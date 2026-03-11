import type { Metadata } from "next";
import "./globals.css";

import { auth } from "@/lib/auth";
import { siteConfig } from "@/lib/site-config";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  title: `${siteConfig.name} | Paid Documentation`,
  description: siteConfig.description,
};

export const dynamic = "force-dynamic";

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();

  return (
    <html lang="zh-CN">
      <body>
        <SiteHeader session={session} />
        {children}
      </body>
    </html>
  );
}
