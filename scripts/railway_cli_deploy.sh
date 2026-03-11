#!/usr/bin/env bash

set -euo pipefail

CLI="pnpm dlx @railway/cli"
PROJECT_NAME="${PROJECT_NAME:-llm-core-paid-site}"
SERVICE_NAME="${SERVICE_NAME:-llm-core-web}"

cat <<EOF
Railway CLI deploy helper

Project: ${PROJECT_NAME}
Service: ${SERVICE_NAME}

Before running the full flow, export these variables:
  export NEXTAUTH_SECRET='...'
  export NEXTAUTH_URL='https://your-domain'
  export NEXT_PUBLIC_APP_URL='https://your-domain'
  export CONTENT_SYNC_TOKEN='...'
  export STRIPE_SECRET_KEY='...'
  export STRIPE_WEBHOOK_SECRET='...'
  export STRIPE_PRO_PRICE_ID='...'

Optional OAuth:
  export GITHUB_ID='...'
  export GITHUB_SECRET='...'
  export GOOGLE_CLIENT_ID='...'
  export GOOGLE_CLIENT_SECRET='...'
EOF

echo
echo "1) Login"
echo "   ${CLI} login"
echo
echo "2) Create project"
echo "   ${CLI} init --name '${PROJECT_NAME}'"
echo
echo "3) Add PostgreSQL"
echo "   ${CLI} add --database postgres"
echo
echo "4) Add web service"
echo "   ${CLI} add --service '${SERVICE_NAME}'"
echo
echo "5) Link current directory to the web service"
echo "   ${CLI} link --service '${SERVICE_NAME}'"
echo
echo "6) Set required variables"
echo "   ${CLI} variable set -s '${SERVICE_NAME}' NEXTAUTH_SECRET=\"\$NEXTAUTH_SECRET\" NEXTAUTH_URL=\"\$NEXTAUTH_URL\" NEXT_PUBLIC_APP_URL=\"\$NEXT_PUBLIC_APP_URL\" CONTENT_SYNC_TOKEN=\"\$CONTENT_SYNC_TOKEN\" STRIPE_SECRET_KEY=\"\$STRIPE_SECRET_KEY\" STRIPE_WEBHOOK_SECRET=\"\$STRIPE_WEBHOOK_SECRET\" STRIPE_PRO_PRICE_ID=\"\$STRIPE_PRO_PRICE_ID\""
echo
echo "7) Set optional OAuth variables"
echo "   ${CLI} variable set -s '${SERVICE_NAME}' GITHUB_ID=\"\${GITHUB_ID:-}\" GITHUB_SECRET=\"\${GITHUB_SECRET:-}\" GOOGLE_CLIENT_ID=\"\${GOOGLE_CLIENT_ID:-}\" GOOGLE_CLIENT_SECRET=\"\${GOOGLE_CLIENT_SECRET:-}\""
echo
echo "8) Create Railway domain"
echo "   ${CLI} domain -s '${SERVICE_NAME}'"
echo
echo "9) Deploy current repository root"
echo "   ${CLI} up -s '${SERVICE_NAME}' -c"
echo
echo "10) Check status"
echo "    ${CLI} status"
echo "    ${CLI} open --print"
