# Running Leo on Slack

Leo connects over Socket Mode, so it runs from a laptop or a container with only outbound
WebSocket traffic. There is no callback URL to expose.

## 1. Install the app

Create the app from [`slack/manifest.yml`](../slack/manifest.yml). It requests three bot
scopes and subscribes to two events:

| Scope | Why |
| --- | --- |
| `app_mentions:read` | Receive `app_mention` events in channels. |
| `chat:write` | Post the reply and update the progress placeholder. |
| `im:history` | Receive `message.im` events in direct messages. |

Leo needs no bulk history scopes. It answers from what it has recorded itself in
`agent_messages` for the conversation it is speaking in, so it cannot read a channel it was
not addressed in, and cannot read a thread it did not participate in.

> If you installed an earlier version of this app, its grant is a superset of these scopes
> and keeps working. Reinstall only if you want the narrower permissions.

Then:

1. Enable **Socket Mode** and create an app-level token (`xapp-…`) with `connections:write`.
2. Install to the workspace and copy the bot token (`xoxb-…`).
3. Put both in `.env` as `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`.
4. Invite Leo to any channel where it should answer. DMs need no invitation.

Never paste a token into a message or a log. `.env` is gitignored, and the process redacts
configured secrets from its own log output.

## 2. Check the wiring

```bash
uv run leo health
```

This reports which credentials are present, whether the database is reachable and migrated,
and how many tools the deployment can offer.

## 3. Run it

```bash
uv run leo slack
```

On startup Leo calls `auth.test` to learn its own user and team id, then listens. Stop with
Ctrl-C; in-flight turns are allowed to finish and reply before the process exits.

## How Leo behaves in Slack

**Channels.** Leo answers when mentioned. It replies in a thread on the message, so a long
answer does not fill the channel. If you mention it inside an existing thread, it stays
there.

**Direct messages.** Every message is a question; no mention needed.

**While it works.** Leo posts `Working on it…` immediately, updates that message with the
tools it is calling, and finally replaces it with the answer. Answers longer than one Slack
block are split across follow-up messages in the same thread.

**Isolation.** Each channel and each DM is a separate scope
(`slack:<team>:<channel>`). Conversation history and memory are read with that scope in the
WHERE clause, so what Leo learns in a DM is not merely deprioritised elsewhere — it is not in
the result set.

## When something goes wrong

Leo always replies. If a turn fails, the reply names the actual failure (a provider error
code, a timeout) rather than a generic apology, so the next step is obvious.

| Symptom | What to check |
| --- | --- |
| No reply at all | Is `leo slack` still running? Check the process log for a socket disconnect. |
| "missing Slack configuration" on start | `SLACK_BOT_TOKEN` or `SLACK_APP_TOKEN` is unset. |
| Replies stop after a restart | Slack redelivers events; duplicates are dropped by `client_msg_id`. A genuinely missed message needs to be re-sent. |
| Answers lack live data | Run `leo health` — provider credentials are optional, and Leo works without them but cannot look things up. |

### Token and access operations

**Rotate the app-level token.** Create a new `xapp-` token in the app's Basic Information
page, replace `SLACK_APP_TOKEN` in `.env`, restart. The old token stops working immediately.

**Rotate the bot token.** Reinstall the app to the workspace, replace `SLACK_BOT_TOKEN`,
restart.

**Suspected compromise.** Revoke the tokens in Slack first, then rotate both as above. Leo
holds no other Slack credential and cannot act in Slack without them.

**Remove Leo from one conversation.** `/remove @Leo` in the channel. Its stored history and
memories for that scope remain in the database; delete the rows for that `scope_key` if you
want them gone:

```sql
DELETE FROM agent_conversations WHERE scope_key = 'slack:T123:C456';
```

Messages, runs, and steps cascade from that row. Memories are keyed by the same
`scope_key` and are deleted separately.

**Uninstall.** Removing the app from the workspace revokes both tokens. Nothing in the
database is affected; reinstalling restores service with the same history.
