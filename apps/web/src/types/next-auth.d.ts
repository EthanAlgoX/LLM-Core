import type { DefaultSession } from "next-auth";
import type { MembershipPlan, MembershipStatus } from "@prisma/client";

declare module "next-auth" {
  interface Session {
    user: DefaultSession["user"] & {
      id: string;
      membershipPlan: MembershipPlan;
      membershipStatus: MembershipStatus;
      membershipExpiresAt: string | null;
    };
  }

  interface User {
    membershipPlan: MembershipPlan;
    membershipStatus: MembershipStatus;
    membershipExpiresAt: Date | null;
  }
}
