# Railway deployment

Leo runs as three services in one Railway project:

- `leo-slack` runs `python -m leo slack` as a long-running Slack Socket Mode process.
- `leo-dashboard-api` runs the read-only FastAPI dashboard API and applies Alembic migrations
  before each deployment.
- `leo-dashboard` runs the public Next.js monitoring dashboard.

The Slack service uses outbound Socket Mode WebSockets, so it does not need a public callback
domain. The API and dashboard services each receive a Railway `*.up.railway.app` domain. The
dashboard's `NEXT_PUBLIC_DASHBOARD_API_URL` must point at the API domain.

## Required Railway variables

Set these on `leo-slack`:

```text
LEO_ENV=development
LEO_MODEL=...
SLACK_BOT_TOKEN=...
SLACK_APP_TOKEN=...
LEO_SLACK_TEAM_ID=...
OPENROUTER_API_KEY=...
DATABASE_URL=...
FINNHUB_API_KEY=...
```

Set `DATABASE_URL` on `leo-dashboard-api`, plus:

```text
LEO_DASHBOARD_CORS_ORIGINS=https://<dashboard-domain>
```

Set this build/runtime variable on `leo-dashboard`:

```text
NEXT_PUBLIC_DASHBOARD_API_URL=https://<api-domain>
```

The dashboard is intentionally read-only, but its current application has no user
authentication. Treat the generated URL as a public monitoring surface and add an auth layer
before exposing sensitive operational data to untrusted users.

## Automatic deploys

Each Railway service is connected to `Aleksander2a/leo`, branch `main`. Railway automatically
builds and deploys the affected service after a push or merge into `main`; the GitHub Actions
workflow in `.github/workflows/ci.yml` runs the Python and dashboard quality gates for pushes and
pull requests.
