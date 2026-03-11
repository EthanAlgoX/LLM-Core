# Railway 部署说明

## 目标

将当前付费文档站以单个 Web 服务的形式部署到 Railway，并接入 Railway PostgreSQL。

## 当前实现

- Web 应用目录：[apps/web](/Users/yunxuanhan/Documents/workspace/ai/LLM-Core/apps/web)
- Railway 配置：[railway.json](/Users/yunxuanhan/Documents/workspace/ai/LLM-Core/railway.json)
- Docker 构建文件：[Dockerfile](/Users/yunxuanhan/Documents/workspace/ai/LLM-Core/Dockerfile)

## 部署方式

采用 `Dockerfile` 方案，而不是直接依赖 Railpack 自动推断。

原因：

- 当前仓库是多目录结构，付费站在 `apps/web`
- 文档内容来自仓库根目录 `html/`
- 用根目录 `Dockerfile` 可以在同一个镜像里同时打包 `apps/web` 和 `html/`
- 避免 Railway monorepo root directory 配置后拿不到 `html/` 内容

## 部署流程

Railway 预部署命令：

```bash
cd /app/apps/web && pnpm prisma migrate deploy
```

也就是说：

1. Railway 先在部署阶段执行 Prisma 正式迁移
2. 迁移成功后，再启动 Next.js 生产服务

容器启动命令：

```bash
pnpm start
```

## Railway 变量

Web 服务至少需要：

```bash
NEXTAUTH_SECRET=...
NEXTAUTH_URL=https://your-service.up.railway.app
NEXT_PUBLIC_APP_URL=https://your-service.up.railway.app
CONTENT_SYNC_TOKEN=...
STRIPE_SECRET_KEY=...
STRIPE_WEBHOOK_SECRET=...
STRIPE_PRO_PRICE_ID=...
```

数据库变量：

- `DATABASE_URL` 由 Railway PostgreSQL 服务提供

可选登录变量：

- `GITHUB_ID`
- `GITHUB_SECRET`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`

## 上线步骤

1. 在 Railway 创建新项目并导入当前 GitHub 仓库
2. 添加 PostgreSQL 服务
3. 给 Web 服务生成域名
4. 填写环境变量
5. 等待部署完成并访问 `/api/health`
6. 在 Stripe 中配置 webhook 到 `/api/stripe/webhook`

## 运维要点

- Railway 健康检查路径已设为 `/api/health`
- `railway.json` 已启用失败重启策略
- `watchPatterns` 只关注 `apps/web`、`html` 和 Railway 相关文件
- 如果上游文档重新生成，需要重新部署 Web 服务，让新 `html/` 进入镜像

## Railway CLI 实操顺序

如果你想完全用 CLI，而不是在网页里点：

1. 登录 Railway
2. 创建项目
3. 添加 PostgreSQL
4. 添加 Web 服务
5. 把当前仓库链接到该服务
6. 配置环境变量
7. 生成 Railway 域名
8. 执行首次部署
9. 配 Stripe webhook
10. 再次部署或直接验证支付链路

仓库里已经提供脚本：

- [railway_cli_deploy.sh](/Users/yunxuanhan/Documents/workspace/ai/LLM-Core/scripts/railway_cli_deploy.sh)

从仓库根目录执行：

```bash
bash scripts/railway_cli_deploy.sh
```

它会打印实际命令顺序，不会直接修改你的 Railway 资源。

### 可直接执行的命令

在仓库根目录：

```bash
pnpm dlx @railway/cli login
pnpm dlx @railway/cli init --name llm-core-paid-site
pnpm dlx @railway/cli add --database postgres
pnpm dlx @railway/cli add --service llm-core-web
pnpm dlx @railway/cli link --service llm-core-web
pnpm dlx @railway/cli variable set -s llm-core-web NEXTAUTH_SECRET="..." NEXTAUTH_URL="https://your-service.up.railway.app" NEXT_PUBLIC_APP_URL="https://your-service.up.railway.app" CONTENT_SYNC_TOKEN="..." STRIPE_SECRET_KEY="..." STRIPE_WEBHOOK_SECRET="..." STRIPE_PRO_PRICE_ID="..."
pnpm dlx @railway/cli domain -s llm-core-web
pnpm dlx @railway/cli up -s llm-core-web -c
pnpm dlx @railway/cli status
pnpm dlx @railway/cli open --print
```

### 上线后补的最后一步

把 Stripe webhook 指到：

```text
https://your-service.up.railway.app/api/stripe/webhook
```

然后再验证：

```bash
curl https://your-service.up.railway.app/api/health
```
