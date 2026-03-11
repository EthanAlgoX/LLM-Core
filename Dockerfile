FROM node:22-bookworm-slim AS builder

RUN apt-get update \
  && apt-get install -y --no-install-recommends openssl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@10.28.2

WORKDIR /app

COPY apps/web/package.json apps/web/pnpm-lock.yaml ./apps/web/
RUN pnpm --dir apps/web install --frozen-lockfile

COPY apps/web ./apps/web
COPY html ./html

RUN pnpm --dir apps/web prisma generate
RUN pnpm --dir apps/web build

FROM node:22-bookworm-slim AS runner

RUN apt-get update \
  && apt-get install -y --no-install-recommends openssl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm@10.28.2

ENV NODE_ENV=production
ENV PORT=3000

WORKDIR /app

COPY --from=builder /app/apps/web ./apps/web
COPY --from=builder /app/html ./html

WORKDIR /app/apps/web

CMD ["pnpm", "start"]
