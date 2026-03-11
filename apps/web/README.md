# LLM-Core Paid Site

这是面向真实收费文档站的应用骨架。它不再直接暴露 `html/` 静态目录，而是由服务端按权限读取内容并决定返回预览版还是全文。

## 技术方案

- `Next.js App Router`
- `NextAuth + Prisma + PostgreSQL`
- `Stripe Checkout + Webhook`
- 服务端按权限读取仓库根目录下的 `html/` 文档

## 访问模型

- 首页、导航页、模块总览页默认公开
- 每个主模块额外开放前 `2` 篇章节作为试读
- 其余章节仅对已付费用户返回全文

## 本地启动

```bash
cd apps/web
cp .env.example .env.local
pnpm install
pnpm prisma:generate
pnpm db:push
pnpm dev
```

生产构建校验：

```bash
pnpm build
```

首次初始化数据库：

```bash
pnpm prisma migrate deploy
```

## Railway 部署

仓库根目录已经提供：

- `Dockerfile`
- `railway.json`
- `/api/health` 健康检查路由

推荐部署方式：

1. 在 Railway 新建 Project
2. 从 GitHub 导入当前仓库
3. 给项目添加 PostgreSQL 服务
4. 给 Web 服务生成公网域名
5. 在 Web 服务变量里设置：

```bash
NEXTAUTH_SECRET=replace-with-a-long-random-secret
NEXTAUTH_URL=https://your-service.up.railway.app
NEXT_PUBLIC_APP_URL=https://your-service.up.railway.app
CONTENT_SYNC_TOKEN=replace-with-an-internal-token
STRIPE_SECRET_KEY=sk_live_or_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRO_PRICE_ID=price_xxx
GITHUB_ID=
GITHUB_SECRET=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

说明：

- `DATABASE_URL` 通常会由 Railway PostgreSQL 自动注入
- Railway 会先执行 `preDeployCommand` 跑 `pnpm prisma migrate deploy`，再启动 Next.js
- 文档内容直接从镜像内的 `/app/html` 读取，不依赖外部卷
- Prisma 初始迁移已经提交到 `prisma/migrations`

首次上线后：

1. 在 Railway Web 服务里确认 `healthcheck` 通过
2. 在 Stripe Dashboard 把 webhook 指到 `https://your-service.up.railway.app/api/stripe/webhook`
3. 重新部署一次，确认支付回调和登录都正常

## 内容同步

- 文档内容来源：仓库根目录 `html/`
- 若上游重新执行了 `python3 scripts/convert_md_to_html.py`，可调用：

```bash
curl -X POST http://localhost:3000/api/content/sync \
  -H "Authorization: Bearer $CONTENT_SYNC_TOKEN"
```
