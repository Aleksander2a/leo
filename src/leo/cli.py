"""Leo's operator CLI."""

from __future__ import annotations

import logging
import sys

import typer
from sqlalchemy import text

from leo.agent.contracts import Scope
from leo.agent.db import create_engine, create_sessions, run
from leo.agent.memory import MemoryService
from leo.agent.runtime import TurnRequest, runtime
from leo.config import Settings, has_value
from leo.safe_logging import configure_safe_logging

app = typer.Typer(help="Leo — an AI portfolio research agent that lives in Slack.")
memory_app = typer.Typer(help="Inspect and prune what Leo remembers.")
app.add_typer(memory_app, name="memory")


def _configure_logging(settings: Settings) -> None:
    configure_safe_logging(
        level_name=settings.leo_log_level,
        sensitive_values=settings.sensitive_values_for_logging(),
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "mcp.client.streamable_http", "slack_bolt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _scope(key: str, actor: str) -> Scope:
    return Scope(key=key, actor_id=actor)


@app.command()
def ask(
    question: str = typer.Argument(..., help="What to ask Leo."),
    scope: str = typer.Option("cli:local:default", help="Conversation isolation key."),
    actor: str = typer.Option("cli-user", help="Who is asking."),
    trace: bool = typer.Option(False, "--trace", help="Print each tool call as it runs."),
) -> None:
    """Ask Leo one question from the terminal."""

    settings = Settings()
    _configure_logging(settings)

    async def main() -> int:
        async with runtime(settings) as agent:

            async def progress(names: str) -> None:
                typer.secho(f"  … {names}", fg=typer.colors.BRIGHT_BLACK, err=True)

            result = await agent.handle(
                TurnRequest(
                    question=question,
                    scope=_scope(scope, actor),
                    scope_description="a terminal session",
                    conversation_kind="cli",
                    on_step=progress if trace else None,
                )
            )
        if result.answered:
            typer.echo(result.answer)
        else:
            typer.secho(f"No answer: {result.error}", fg=typer.colors.RED, err=True)
        typer.secho(
            f"[{result.status} · {result.turns} turns · {result.tool_calls} tool calls · "
            f"{result.usage.total_tokens} tokens · ${result.usage.cost:.4f}]",
            fg=typer.colors.BRIGHT_BLACK,
            err=True,
        )
        return 0 if result.answered else 1

    raise typer.Exit(run(main()))


@app.command()
def chat(
    scope: str = typer.Option("cli:local:chat", help="Conversation isolation key."),
    actor: str = typer.Option("cli-user", help="Who is asking."),
) -> None:
    """Hold a multi-turn conversation with Leo in the terminal."""

    settings = Settings()
    _configure_logging(settings)

    async def main() -> None:
        async with runtime(settings) as agent:
            typer.secho(
                f"Leo ready ({len(agent.tool_names)} tools). Ctrl-D or 'exit' to quit.",
                fg=typer.colors.GREEN,
            )
            while True:
                try:
                    question = typer.prompt("\nyou", prompt_suffix="> ").strip()
                except (EOFError, typer.Abort):
                    return
                if question.lower() in {"exit", "quit"}:
                    return
                if not question:
                    continue

                async def progress(names: str) -> None:
                    typer.secho(f"  … {names}", fg=typer.colors.BRIGHT_BLACK, err=True)

                result = await agent.handle(
                    TurnRequest(
                        question=question,
                        scope=_scope(scope, actor),
                        scope_description="a terminal session",
                        conversation_kind="cli",
                        on_step=progress,
                    )
                )
                typer.echo("")
                if result.answered:
                    typer.echo(result.answer)
                else:
                    typer.secho(f"No answer: {result.error}", fg=typer.colors.RED)

    run(main())


@app.command()
def slack() -> None:
    """Run Leo on Slack over Socket Mode until stopped."""

    settings = Settings()
    _configure_logging(settings)
    from leo.slack.app import serve

    missing = settings.missing_for_live_slack()
    if missing:
        typer.secho(f"missing Slack configuration: {', '.join(missing)}", fg=typer.colors.RED)
        raise typer.Exit(2)
    try:
        run(serve(settings))
    except KeyboardInterrupt:
        typer.secho("stopped", fg=typer.colors.YELLOW)
    except RuntimeError as exc:
        # A misconfigured deployment is an operator problem, not a bug, and its
        # log line is the only thing an operator sees before the container exits.
        # A one-line cause beats a rich traceback wrapped in a Typer panel.
        typer.secho(f"cannot start: {exc}", fg=typer.colors.RED)
        raise typer.Exit(2) from None


@app.command("slack-live", hidden=True)
def slack_live() -> None:
    """Deprecated alias for `leo slack`."""

    # A host like Railway can pin a start command in its own settings, where this
    # repo cannot see it and CI cannot check it. `slack-live` was this command's
    # name before the CLI was reorganised, so a deployment still holding that name
    # would exit-loop on an unknown command with an otherwise green build. Keeping
    # the alias means it boots and logs what to change instead. Safe to delete once
    # no deployment references it.
    logging.getLogger(__name__).warning(
        "`leo slack-live` is a deprecated alias; update this deployment's start "
        "command to `python -m leo slack`."
    )
    slack()


@app.command()
def health() -> None:
    """Check configuration, database, model, and tool availability."""

    settings = Settings()
    _configure_logging(settings)
    ok = True

    async def main() -> bool:
        healthy = True
        for name, value in (
            ("OPENROUTER_API_KEY", settings.openrouter_api_key),
            ("LEO_MODEL", settings.leo_model),
            ("DATABASE_URL", settings.database_url),
            ("SLACK_BOT_TOKEN", settings.slack_bot_token),
            ("SLACK_APP_TOKEN", settings.slack_app_token),
        ):
            present = has_value(value)
            typer.secho(
                f"  {'✓' if present else '✗'} {name}",
                fg=typer.colors.GREEN if present else typer.colors.RED,
            )
            if not present and name != "SLACK_BOT_TOKEN" and name != "SLACK_APP_TOKEN":
                healthy = False
        if settings.database_url is not None:
            engine = create_engine(settings.database_url.get_secret_value())
            try:
                async with engine.connect() as connection:
                    count = (
                        await connection.execute(text("select count(*) from agent_runs"))
                    ).scalar()
                typer.secho(f"  ✓ database reachable ({count} runs recorded)", fg="green")
            except Exception as exc:
                typer.secho(f"  ✗ database: {type(exc).__name__}: {exc}", fg="red")
                healthy = False
            finally:
                await engine.dispose()
        if healthy:
            async with runtime(settings) as agent:
                typer.secho(f"  ✓ {len(agent.tool_names)} tools available", fg="green")
                for name in agent.tool_names:
                    typer.secho(f"      {name}", fg=typer.colors.BRIGHT_BLACK)
        return healthy

    ok = run(main())
    raise typer.Exit(0 if ok else 1)


@memory_app.command("list")
def memory_list(
    scope: str = typer.Option(..., help="Conversation isolation key to inspect."),
    limit: int = typer.Option(50),
) -> None:
    """Show what Leo remembers for one conversation."""

    settings = Settings()
    _configure_logging(settings)
    assert settings.database_url is not None
    url = settings.database_url.get_secret_value()

    async def main() -> None:
        engine = create_engine(url)
        try:
            service = MemoryService(sessions=create_sessions(engine))
            found = await service.list_all(_scope(scope, "cli"), limit=limit)
            if not found:
                typer.secho("no memories in this scope", fg=typer.colors.YELLOW)
            for item in found:
                typer.echo(f"{item.id}  [{item.kind}] {item.subject or '-'}: {item.content}")
        finally:
            await engine.dispose()

    run(main())


@memory_app.command("forget")
def memory_forget(
    scope: str = typer.Option(..., help="Conversation isolation key."),
    memory_id: str = typer.Argument(..., help="Memory id from `leo memory list`."),
) -> None:
    """Retire one memory."""

    settings = Settings()
    _configure_logging(settings)
    assert settings.database_url is not None
    url = settings.database_url.get_secret_value()

    async def main() -> None:
        engine = create_engine(url)
        try:
            service = MemoryService(sessions=create_sessions(engine))
            removed = await service.forget(_scope(scope, "cli"), memory_id)
            typer.secho(
                "forgotten" if removed else "no such active memory in this scope",
                fg=typer.colors.GREEN if removed else typer.colors.YELLOW,
            )
        finally:
            await engine.dispose()

    run(main())


if __name__ == "__main__":
    app()
