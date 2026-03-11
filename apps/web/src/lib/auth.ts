import { PrismaAdapter } from "@next-auth/prisma-adapter";
import type { NextAuthOptions } from "next-auth";
import { getServerSession } from "next-auth";
import GitHubProvider from "next-auth/providers/github";
import GoogleProvider from "next-auth/providers/google";

import { db } from "@/lib/db";
import { env } from "@/lib/env";

const providers = [];

if (env.GITHUB_ID && env.GITHUB_SECRET) {
  providers.push(
    GitHubProvider({
      clientId: env.GITHUB_ID,
      clientSecret: env.GITHUB_SECRET,
    }),
  );
}

if (env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET) {
  providers.push(
    GoogleProvider({
      clientId: env.GOOGLE_CLIENT_ID,
      clientSecret: env.GOOGLE_CLIENT_SECRET,
    }),
  );
}

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(db),
  session: {
    strategy: "database",
  },
  pages: {
    signIn: "/sign-in",
  },
  providers,
  callbacks: {
    async session({ session, user }) {
      if (!session.user) {
        return session;
      }

      session.user.id = user.id;
      session.user.membershipPlan = user.membershipPlan;
      session.user.membershipStatus = user.membershipStatus;
      session.user.membershipExpiresAt = user.membershipExpiresAt?.toISOString() ?? null;
      return session;
    },
  },
};

export function auth() {
  return getServerSession(authOptions);
}

export function getEnabledProviders() {
  return authOptions.providers.map((provider) => ({
    id: provider.id,
    name: provider.name,
  }));
}
