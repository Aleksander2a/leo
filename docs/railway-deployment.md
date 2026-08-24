# Railway deployment

Leo runs as three services in one Railway project:

- `leo-slack` runs `python -m leo slack` as a long-running Slack Socket Mode process.
- `leo-dashboard-api` runs the read-only FastAPI dashboard API and applies Alembic migrations
  before each deployment.
- `leo-dashboard` runs the public Next.js monitoring dashboard.

The Slack service uses outbound Socket Mode WebSockets, so it does not need a public callback
domain. The API and dashboard services each receive a Railway `*.up.railway.app` domain. The
dashboard's `NEXT_PUBLIC_DASHBOARD_API_URL` must point at the API domain.

## Start commands

| Service | Start command |
| --- | --- |
| `leo-slack` | `python -m leo slack` |
| `leo-dashboard-api` | `alembic upgrade head && python scripts/run_dashboard_api.py` |
| `leo-dashboard` | `npm run start` |

`leo-slack` gets its command from the `CMD` in [`Dockerfile`](../Dockerfile), and
`tests/test_entrypoints.py` asserts that command still resolves against the CLI. **Prefer
leaving the Railway *Start Command* field empty for that service** so the Dockerfile stays the
single source of truth.

A value typed into Railway's Start Command field overrides the Dockerfile and is invisible to
this repository — CI cannot check it, and a CLI rename will not update it. The symptom is a
green build with `Your service's container is not running (status exited)`, and the deploy log
shows `No such command`. Fix it by clearing the field, or by setting it to the command in the
table above.

> `leo slack-live` is kept as a hidden alias of `leo slack` for deployments still pinned to the
> older name. It boots normally and logs a warning naming the command to switch to.

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
