from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from leo.harness.models import ScopeKey
from leo.integrations.slack.cancellation import (
    SlackCancellationOutcome,
    SlackCancellationResult,
    is_slack_cancellation_request,
)
from leo.integrations.slack.events import AdmittedSlackMention, SlackLaunchRef
from leo.integrations.slack.socket_mode import (
    InMemorySlackIngressAdmission,
    SlackJobProcessor,
    _handle_app_mention,
    _handle_message_im,
)


@pytest.mark.parametrize(
    "prompt",
    [
        "cancel",
        "Cancel this task!",
        " please   stop Leo. ",
        "PLEASE CANCEL THIS RUN???",
    ],
)
def test_exact_slack_cancellation_vocabulary_is_normalized(prompt: str) -> None:
    assert is_slack_cancellation_request(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Can you explain task cancellation?",
        "Do not stop analyzing this",
        "What happens if I say cancel?",
        "cancel task-foreign",
        "stop after you summarize",
    ],
)
def test_conversational_stop_language_is_not_a_control_request(prompt: str) -> None:
    assert not is_slack_cancellation_request(prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("direct_message", [False, True], ids=["app-mention", "message-im"])
async def test_exact_cancel_is_intercepted_before_runtime_and_uses_terminal_delivery(
    direct_message: bool,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.posts: list[dict[str, str]] = []

        async def conversations_info(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            return {
                "ok": True,
                "channel": {"id": "C1", "is_channel": True, "is_member": True},
            }

        async def users_conversations(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"ok": True, "channels": [], "response_metadata": {"next_cursor": ""}}

        async def chat_postMessage(self, **kwargs: str) -> dict[str, str]:
            self.posts.append(kwargs)
            return {"ts": "2.0"}

    class Runtime:
        async def handle(self, admitted: AdmittedSlackMention) -> str:
            del admitted
            raise AssertionError("exact cancellation must not reach the model runtime")

    class Preparer:
        async def prepare(self, admitted: AdmittedSlackMention) -> AdmittedSlackMention:
            return replace(
                admitted,
                launch=SlackLaunchRef(
                    thread_id="thread-control",
                    task_id="task-control",
                    run_id="run-control",
                ),
            )

        async def recover(self) -> tuple[AdmittedSlackMention, ...]:
            return ()

    class Cancellation:
        def accepts(self, prompt: str) -> bool:
            return is_slack_cancellation_request(prompt)

        async def handle(
            self,
            admitted: AdmittedSlackMention,
            launch_preparer: Preparer,
        ) -> SlackCancellationResult:
            prepared = await launch_preparer.prepare(admitted)
            return SlackCancellationResult(
                admitted=prepared,
                outcome=SlackCancellationOutcome.APPLIED,
                message="Leo cancelled the active task and its child work.",
            )

        async def recover(
            self,
            launch_preparer: Preparer,
            *,
            limit: int = 100,
        ) -> tuple[SlackCancellationResult, ...]:
            del launch_preparer, limit
            return ()

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=Runtime())  # type: ignore[arg-type]
    channel_id = "D1" if direct_message else "C1"
    event: dict[str, object] = {
        "type": "message" if direct_message else "app_mention",
        "user": "U1",
        "channel": channel_id,
        "channel_type": "im" if direct_message else "channel",
        "text": "cancel this task" if direct_message else "<@UBOT> cancel this task",
        "ts": "2.0",
        "thread_ts": "1.0",
    }
    body: dict[str, object] = {
        "type": "event_callback",
        "event_id": "Ev-cancel-control",
        "team_id": "T1",
        "event": event,
    }

    handler = _handle_message_im if direct_message else _handle_app_mention
    await handler(
        body,
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=asyncio.Queue(maxsize=1),
        admission_timeout_seconds=1,
        launch_preparer=Preparer(),
        cancellation_handler=Cancellation(),  # type: ignore[arg-type]
    )

    assert processor.queue.empty()
    assert client.posts == [
        {
            "channel": channel_id,
            "thread_ts": "1.0",
            "text": "Leo cancelled the active task and its child work.",
        }
    ]
