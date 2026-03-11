import type { MembershipStatus } from "@prisma/client";
import type { Session } from "next-auth";

import type { DocRecord } from "@/lib/docs";

const ACTIVE_STATUSES: MembershipStatus[] = ["ACTIVE", "TRIALING"];

export function hasPaidAccess(session: Session | null) {
  const user = session?.user;
  if (!user) {
    return false;
  }

  if (!ACTIVE_STATUSES.includes(user.membershipStatus)) {
    return false;
  }

  if (!user.membershipExpiresAt) {
    return true;
  }

  return new Date(user.membershipExpiresAt).getTime() > Date.now();
}

export function canReadFullDoc(session: Session | null, doc: DocRecord) {
  return doc.publicAccess || hasPaidAccess(session);
}
