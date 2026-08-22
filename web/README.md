# Leo admin dashboard

A read-only, admin-only Next.js app for inspecting how the Leo harness is behaving:
aggregate stats (tool call success rate, per-integration call volume, memory writes, context
stats, cost/token usage, failure modes) and per-run deep dives (full event timeline, context
manifests, tool calls and observations, plan/delegation trees, verification, Slack delivery).

It reads exclusively from the FastAPI dashboard API in `../src/leo/api/dashboard/` (see
`src/lib/api.ts`). It has no privileged database credential of its own, no write endpoints, and
no authentication -- it is meant to run locally next to the harness.

## Running locally

Two processes, from the repository root:

```powershell
# 1. Dashboard API (Windows needs the selector-loop-safe runner, not plain uvicorn -- see
#    scripts/run_dashboard_api.py for why)
uv run python scripts/run_dashboard_api.py
```

```bash
# 2. This app, in a second terminal
cd web
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). `NEXT_PUBLIC_DASHBOARD_API_URL` in
`.env.local` points the app at the API (defaults to `http://127.0.0.1:8000`).

## Structure

- `src/app/` -- one route per dashboard section (overview, runs, memory, integrations,
  failures, conversations), App Router, server-rendered against live data (`cache: "no-store"`).
- `src/components/dashboard/` -- page-specific views: the event timeline, the recursive
  plan/delegation tree, run-detail tabs, filter bars.
- `src/components/ui/` -- shared primitives (status pill, data table, JSON tree viewer, KPI
  card, pager).
- `src/lib/api.ts` / `src/lib/types.ts` -- the typed client for the dashboard API. Kept in sync
  by hand with `src/leo/api/dashboard/routers/*.py` -- there is no generated client.

## Stack

Next.js (App Router, TypeScript) · Tailwind CSS v4 (class-based dark mode via `next-themes`) ·
Recharts · TanStack Table/Query · `react-json-view-lite`.
