"""Bounded, projection-safe, thread-isolated Slack context loading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

from pydantic import Field
from slack_sdk.web.async_client import AsyncWebClient

from leo.harness.context_budget import ContextBudgetError
from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    ContractModel,
    NonEmptyStr,
)
from leo.harness.thread_context import (
    ThreadContextRange,
    ThreadTurnRetentionInput,
    classify_thread_transcript,
    select_context_with_thread_compaction,
)
from leo.integrations.slack.events import SlackConversationKind, SlackMentionJob
from leo.persistence.slack_messages import (
    PersistedSlackThreadSnapshot,
    SlackThreadContextFallback,
    SlackThreadCoverageReason,
    SlackThreadCoverageSource,
)

HARD_MAX_MESSAGES_PER_CONVERSATION = 200
HARD_MAX_MESSAGES_GLOBAL = 500
HARD_MAX_PAGES_PER_CONVERSATION = 10
HARD_MAX_PAGES_GLOBAL = 50
HARD_MAX_CONCURRENCY = 8
HARD_MAX_CONTEXT_TOKENS = 16_000
HARD_MAX_PAGE_SIZE = 200
HARD_MAX_THREAD_MESSAGES = 200
HARD_MAX_THREAD_PAGES = 20
THREAD_RECENT_TURNS = 12
USER_HISTORY_AUTH_MAX_ATTEMPTS = 2
USER_HISTORY_AUTH_TIMEOUT_SECONDS = 2.0
USER_HISTORY_AUTH_BACKOFF_SECONDS = 0.05

_NON_CONVERSATIONAL_SUBTYPES = frozenset(
    {
        "channel_archive",
        "channel_join",
        "channel_leave",
        "channel_name",
        "channel_purpose",
        "channel_topic",
        "ekm_access_denied",
        "group_archive",
        "group_join",
        "group_leave",
        "group_name",
        "group_purpose",
        "group_topic",
        "message_deleted",
        "message_replied",
        "message_changed",
        "pinned_item",
        "unpinned_item",
    }
)


class SlackThreadContextError(RuntimeError):
    """A thread transcript could not be loaded completely and safely."""


class SlackHistoryClient(Protocol):
    async def auth_test(self, **kwargs: object) -> object: ...

    async def conversations_history(self, **kwargs: object) -> object: ...

    async def conversations_replies(self, **kwargs: object) -> object: ...


class SlackHistoryContextManifest(ContractModel):
    """Inspectable record of the authority, fetch, and budget decisions."""

    context_access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    requested_conversation_ids: tuple[NonEmptyStr, ...]
    loaded_conversation_ids: tuple[NonEmptyStr, ...] = ()
    failed_conversation_ids: tuple[NonEmptyStr, ...] = ()
    cap_skipped_conversation_ids: tuple[NonEmptyStr, ...] = ()
    history_requests: int = Field(ge=0)
    raw_messages_scanned: int = Field(ge=0)
    eligible_messages_ranked: int = Field(ge=0)
    selected_messages: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    selection_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    truncated: bool
    thread_triggered: bool = False
    thread_root_ts: str | None = None
    thread_requests: int = Field(default=0, ge=0)
    thread_raw_messages_scanned: int = Field(default=0, ge=0)
    thread_messages_loaded: int = Field(default=0, ge=0)
    thread_messages_compacted: int = Field(default=0, ge=0)
    thread_compaction_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    protected_thread_item_ids: tuple[NonEmptyStr, ...] = ()
    thread_reopen_handles: tuple[NonEmptyStr, ...] = ()
    thread_complete: bool = True
    thread_source: Literal[
        "not_applicable",
        "slack_replies_bot",
        "slack_replies_user",
        "persisted_complete",
    ] = "not_applicable"
    thread_coverage_reason: NonEmptyStr | None = None
    thread_coverage_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class SlackHistoryContextResult(ContractModel):
    items: tuple[ContextItem, ...]
    manifest: SlackHistoryContextManifest
    reopen_ranges: tuple[ThreadContextRange, ...] = Field(default=(), exclude=True)


def slack_history_authority_ids(
    manifest: SlackHistoryContextManifest,
) -> tuple[str, ...]:
    """Project full-thread loading into fixed-size, content-free authority evidence.

    ``SlackHistoryContextManifest`` intentionally retains runtime-only item IDs and
    opaque reopen handles.  Durable CONTEXT_BUILT events need proof of completeness
    and compaction without persisting either collection, so the authority snapshot
    receives only bounded counts and canonical SHA-256 digests.
    """

    reopen_digest = hashlib.sha256(
        json.dumps(
            list(manifest.thread_reopen_handles),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    coverage_digest = manifest.thread_coverage_digest or "none"
    compaction_digest = manifest.thread_compaction_digest or "none"
    return (
        "slack-thread-proof-version:v1",
        f"slack-history:{manifest.selection_digest}",
        f"slack-history-selection-digest:{manifest.selection_digest}",
        f"slack-thread-complete:{str(manifest.thread_complete).lower()}",
        f"slack-thread-source:{manifest.thread_source}",
        f"slack-thread-coverage-digest:{coverage_digest}",
        f"slack-thread-compaction-digest:{compaction_digest}",
        f"slack-thread-protected-count:{len(manifest.protected_thread_item_ids)}",
        f"slack-thread-compacted-count:{manifest.thread_messages_compacted}",
        f"slack-thread-reopen-handle-count:{len(manifest.thread_reopen_handles)}",
        f"slack-thread-reopen-handle-digest:{reopen_digest}",
    )


@dataclass(frozen=True, slots=True)
class _HistoryMessage:
    conversation_id: str
    message_ts: str
    actor_id: str
    actor_kind: str
    text: str


@dataclass(frozen=True, slots=True)
class _ConversationFetch:
    conversation_id: str
    messages: tuple[_HistoryMessage, ...]
    requests: int
    raw_messages_scanned: int
    loaded: bool
    failed: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class _ThreadFetch:
    messages: tuple[_HistoryMessage, ...]
    requests: int
    raw_messages_scanned: int
    failed: bool
    truncated: bool
    source: Literal["slack_replies_bot", "slack_replies_user", "persisted_complete"]
    coverage_reason: str | None = None
    coverage_digest: str | None = None


@dataclass(frozen=True, slots=True)
class _ThreadReadIdentity:
    client: AsyncWebClient | SlackHistoryClient | None
    source: Literal["slack_replies_bot", "slack_replies_user"]


class _GlobalPageBudget:
    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self._lock = asyncio.Lock()

    async def claim(self) -> bool:
        async with self._lock:
            if self._remaining == 0:
                return False
            self._remaining -= 1
            return True


class SlackHistoryContextLoader:
    """Fetch recent Slack messages inside a server-derived admission projection.

    The loader accepts no conversation IDs or query from a model.  Both the authorization
    set and relevance objective come from the trusted ``SlackMentionJob``.
    """

    def __init__(
        self,
        client: AsyncWebClient | SlackHistoryClient,
        *,
        user_history_client: AsyncWebClient | SlackHistoryClient | None = None,
        thread_fallback: SlackThreadContextFallback | None = None,
        max_messages_per_conversation: int = 40,
        max_messages_global: int = 120,
        max_pages_per_conversation: int = 3,
        max_pages_global: int = 20,
        max_concurrency: int = 4,
        page_size: int = 40,
        max_context_tokens: int = 4_000,
        max_thread_messages: int = 200,
        max_thread_pages: int = 10,
    ) -> None:
        _bounded(
            max_messages_per_conversation,
            "max_messages_per_conversation",
            HARD_MAX_MESSAGES_PER_CONVERSATION,
        )
        _bounded(max_messages_global, "max_messages_global", HARD_MAX_MESSAGES_GLOBAL)
        _bounded(
            max_pages_per_conversation,
            "max_pages_per_conversation",
            HARD_MAX_PAGES_PER_CONVERSATION,
        )
        _bounded(max_pages_global, "max_pages_global", HARD_MAX_PAGES_GLOBAL)
        _bounded(max_concurrency, "max_concurrency", HARD_MAX_CONCURRENCY)
        _bounded(page_size, "page_size", HARD_MAX_PAGE_SIZE)
        _bounded(max_context_tokens, "max_context_tokens", HARD_MAX_CONTEXT_TOKENS)
        _bounded(max_thread_messages, "max_thread_messages", HARD_MAX_THREAD_MESSAGES)
        _bounded(max_thread_pages, "max_thread_pages", HARD_MAX_THREAD_PAGES)
        self._client = client
        self._user_history_client = user_history_client
        self._thread_fallback = thread_fallback
        self._user_history_team_id: str | None = None
        self._user_history_attestation_complete = False
        self._user_history_attestation_lock = asyncio.Lock()
        self._max_messages_per_conversation = max_messages_per_conversation
        self._max_messages_global = max_messages_global
        self._max_pages_per_conversation = max_pages_per_conversation
        self._max_pages_global = max_pages_global
        self._max_concurrency = max_concurrency
        self._page_size = page_size
        self._max_context_tokens = max_context_tokens
        self._max_thread_messages = max_thread_messages
        self._max_thread_pages = max_thread_pages

    async def load(self, job: SlackMentionJob) -> SlackHistoryContextResult:
        thread_triggered = job.message_ts != job.thread_root_ts
        thread_fetch = (
            await self._load_complete_thread(job)
            if thread_triggered
            else _ThreadFetch(
                messages=(),
                requests=0,
                raw_messages_scanned=0,
                failed=False,
                truncated=False,
                source="slack_replies_bot",
            )
        )
        if thread_triggered:
            if thread_fetch.failed:
                raise SlackThreadContextError("slack_thread_history_unavailable")
            if thread_fetch.truncated:
                raise SlackThreadContextError("slack_thread_history_incomplete")
            if not any(
                message.message_ts == job.thread_root_ts for message in thread_fetch.messages
            ):
                raise SlackThreadContextError("slack_thread_root_missing")

        # Slack destination history is not ordinary conversational context.  In a
        # channel or group it contains unrelated roots and replies, so ranking it by
        # lexical overlap can silently turn an old thread into preferences or
        # constraints for a new request.  A threaded turn therefore receives only
        # its complete exact thread, while a fresh non-DM root starts isolated.
        #
        # Unthreaded 1:1 DMs are the sole continuity exception: their admission
        # projection is an explicit, current membership intersection and remains
        # bounded by the quotas below.  Deliberate cross-thread recall outside that
        # exception belongs to memory/search tools, not ambient Slack history.
        authorized_conversation_ids = self._trusted_projection(job)
        conversation_ids = (
            authorized_conversation_ids
            if not thread_triggered and job.conversation_kind is SlackConversationKind.DM
            else ()
        )
        quotas = _allocate_message_quotas(
            conversation_ids,
            per_conversation=self._max_messages_per_conversation,
            global_limit=self._max_messages_global,
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)
        page_budget = _GlobalPageBudget(self._max_pages_global)
        fetches = await asyncio.gather(
            *(
                self._fetch_conversation(
                    job,
                    conversation_id,
                    message_limit=quotas[conversation_id],
                    semaphore=semaphore,
                    page_budget=page_budget,
                )
                for conversation_id in conversation_ids
                if quotas[conversation_id] > 0
            )
        )
        fetched_by_id = {fetch.conversation_id: fetch for fetch in fetches}
        cap_skipped = tuple(
            conversation_id for conversation_id in conversation_ids if quotas[conversation_id] == 0
        )
        thread_messages = tuple(
            sorted(
                thread_fetch.messages,
                key=lambda item: (_slack_timestamp(item.message_ts), item.message_ts),
            )
        )
        thread_keys = {(message.conversation_id, message.message_ts) for message in thread_messages}
        background_messages = tuple(
            message
            for fetch in fetches
            for message in fetch.messages
            if (message.conversation_id, message.message_ts) not in thread_keys
        )
        ranked_background = tuple(
            sorted(background_messages, key=lambda item: _rank_key(item, job.prompt))
        )
        recent_thread_timestamps = frozenset(
            message.message_ts for message in thread_messages[-THREAD_RECENT_TURNS:]
        )
        thread_retentions = classify_thread_transcript(
            tuple(
                ThreadTurnRetentionInput(
                    content=message.text,
                    actor_id=message.actor_id,
                    speaker_role=(
                        "assistant" if message.actor_kind in {"assistant", "bot", "app"} else "user"
                    ),
                    is_root=message.message_ts == job.thread_root_ts,
                    is_recent=message.message_ts in recent_thread_timestamps,
                )
                for message in thread_messages
            )
        )
        thread_items = tuple(
            _context_item(
                job,
                message,
                thread=True,
                thread_retention=retention,
            )
            for message, retention in zip(thread_messages, thread_retentions, strict=True)
        )
        background_items = tuple(
            _context_item(job, message, objective=job.prompt) for message in ranked_background
        )
        items = (*thread_items, *background_items)
        try:
            selection = select_context_with_thread_compaction(
                items,
                thread_item_ids=frozenset(item.id for item in thread_items),
                conversation_id=job.channel_id,
                summary_id_namespace=(
                    f"slack-thread-compaction:{job.team_id}:{job.channel_id}:"
                    f"{job.thread_root_ts}:{job.event_id}"
                ),
                max_tokens=self._max_context_tokens,
            )
        except ContextBudgetError as exc:
            raise SlackThreadContextError(exc.safe_code) from exc
        selected = selection.items
        estimated_tokens = selection.budgeted.estimated_tokens
        selection_digest = _selection_digest(selected)
        budget_truncated = bool(selection.budgeted.evicted_names)
        manifest = SlackHistoryContextManifest(
            context_access_hash=job.context_access_hash,
            requested_conversation_ids=conversation_ids,
            loaded_conversation_ids=tuple(
                conversation_id
                for conversation_id in conversation_ids
                if ((fetch := fetched_by_id.get(conversation_id)) is not None and fetch.loaded)
            ),
            failed_conversation_ids=tuple(
                conversation_id
                for conversation_id in conversation_ids
                if ((fetch := fetched_by_id.get(conversation_id)) is not None and fetch.failed)
            ),
            cap_skipped_conversation_ids=cap_skipped,
            history_requests=sum(fetch.requests for fetch in fetches),
            raw_messages_scanned=sum(fetch.raw_messages_scanned for fetch in fetches),
            eligible_messages_ranked=len(thread_messages) + len(ranked_background),
            selected_messages=len(selected),
            estimated_tokens=estimated_tokens,
            selection_digest=selection_digest,
            truncated=(
                bool(cap_skipped)
                or budget_truncated
                or any(fetch.failed or fetch.truncated for fetch in fetches)
            ),
            thread_triggered=thread_triggered,
            thread_root_ts=job.thread_root_ts if thread_triggered else None,
            thread_requests=thread_fetch.requests,
            thread_raw_messages_scanned=thread_fetch.raw_messages_scanned,
            thread_messages_loaded=len(thread_messages),
            thread_messages_compacted=len(selection.compacted_item_ids),
            thread_compaction_digest=selection.compaction_digest,
            protected_thread_item_ids=tuple(
                item.id for item in thread_items if item.retention.pinned
            ),
            thread_reopen_handles=tuple(item.handle for item in selection.reopen_ranges),
            thread_complete=not thread_fetch.failed and not thread_fetch.truncated,
            thread_source=thread_fetch.source if thread_triggered else "not_applicable",
            thread_coverage_reason=thread_fetch.coverage_reason,
            thread_coverage_digest=thread_fetch.coverage_digest,
        )
        return SlackHistoryContextResult(
            items=selected,
            manifest=manifest,
            reopen_ranges=selection.reopen_ranges,
        )

    @staticmethod
    def _trusted_projection(job: SlackMentionJob) -> tuple[str, ...]:
        # SlackMentionJob validates this invariant at the authority boundary. Reassert it
        # here so future contract changes cannot accidentally broaden a non-DM fetch.
        if job.conversation_kind is SlackConversationKind.DM:
            return job.context_conversation_ids
        return (job.channel_id,)

    async def _load_complete_thread(self, job: SlackMentionJob) -> _ThreadFetch:
        """Prefer Slack replies, then accept only a proved-complete persisted snapshot."""

        identity = await self._thread_read_identity(job)
        direct = (
            await self._fetch_thread(
                job,
                client=identity.client,
                source=identity.source,
            )
            if identity.client is not None
            else _ThreadFetch(
                messages=(),
                requests=0,
                raw_messages_scanned=0,
                failed=True,
                truncated=False,
                source=identity.source,
            )
        )
        direct_has_root = any(
            message.message_ts == job.thread_root_ts for message in direct.messages
        )
        if not direct.failed and not direct.truncated and direct_has_root:
            return direct
        if self._thread_fallback is None:
            return direct

        coverage_recorded = False
        coverage_requests = 0
        coverage_scanned = 0
        if identity.source == "slack_replies_user" and identity.client is not None:
            (
                coverage_recorded,
                user_coverage_requests,
                user_coverage_scanned,
            ) = await self._refresh_persisted_root_coverage(
                job,
                client=identity.client,
                source=SlackThreadCoverageSource.USER_HISTORY,
            )
            coverage_requests += user_coverage_requests
            coverage_scanned += user_coverage_scanned

        # A channel user may be able to read replies but not be a member of this
        # destination.  The bot admission identity can still attest the exact root
        # through conversations.history.  This is coverage only: channel transcripts
        # are never broadened to bot conversations.replies after the user read fails.
        if not coverage_recorded:
            (
                _bot_recorded,
                bot_coverage_requests,
                bot_coverage_scanned,
            ) = await self._refresh_persisted_root_coverage(
                job,
                client=self._client,
                source=SlackThreadCoverageSource.BOT_HISTORY,
            )
            coverage_requests += bot_coverage_requests
            coverage_scanned += bot_coverage_scanned
        snapshot = await self._thread_fallback.load_complete_thread(
            team_id=job.team_id,
            channel_id=job.channel_id,
            thread_root_ts=job.thread_root_ts,
            current_message_ts=job.message_ts,
            current_actor_id=job.user_id,
            current_event_id=job.event_id,
            max_messages=self._max_thread_messages,
        )
        persisted = _persisted_thread_fetch(
            job,
            snapshot,
            requests=direct.requests + coverage_requests,
            raw_messages_scanned=direct.raw_messages_scanned + coverage_scanned,
        )
        if persisted is not None:
            return persisted
        return _ThreadFetch(
            messages=direct.messages,
            requests=direct.requests + coverage_requests,
            raw_messages_scanned=direct.raw_messages_scanned + coverage_scanned,
            failed=True,
            truncated=direct.truncated,
            source=direct.source,
            coverage_reason=snapshot.coverage_reason.value,
            coverage_digest=snapshot.coverage_digest,
        )

    async def _thread_read_identity(
        self,
        job: SlackMentionJob,
    ) -> _ThreadReadIdentity:
        if job.conversation_kind in {SlackConversationKind.DM, SlackConversationKind.MPIM}:
            return _ThreadReadIdentity(client=self._client, source="slack_replies_bot")
        if self._user_history_client is not None and await self._user_history_is_attested(
            job.team_id
        ):
            return _ThreadReadIdentity(
                client=self._user_history_client,
                source="slack_replies_user",
            )
        # Slack channel replies require a user token.  An absent, transiently
        # unavailable, malformed, or wrong-workspace user identity must not silently
        # broaden into a bot conversations.replies read.
        return _ThreadReadIdentity(client=None, source="slack_replies_user")

    async def _user_history_is_attested(self, team_id: str) -> bool:
        """Cache successful exact-workspace attestation, retrying transient failures."""

        if self._user_history_client is None:
            return False
        async with self._user_history_attestation_lock:
            if self._user_history_attestation_complete:
                return self._user_history_team_id == team_id
            for attempt in range(USER_HISTORY_AUTH_MAX_ATTEMPTS):
                try:
                    response = await asyncio.wait_for(
                        self._user_history_client.auth_test(),
                        timeout=USER_HISTORY_AUTH_TIMEOUT_SECONDS,
                    )
                    payload = getattr(response, "data", response)
                    attested_team = (
                        payload.get("team_id")
                        if isinstance(payload, Mapping) and payload.get("ok", True) is not False
                        else None
                    )
                except Exception:
                    attested_team = None
                if isinstance(attested_team, str) and attested_team:
                    # A successful response, including a wrong-team response, is stable
                    # identity evidence.  Only transport/API/malformed failures remain
                    # retryable on a later load.
                    self._user_history_team_id = attested_team
                    self._user_history_attestation_complete = True
                    return attested_team == team_id
                if attempt + 1 < USER_HISTORY_AUTH_MAX_ATTEMPTS:
                    await asyncio.sleep(USER_HISTORY_AUTH_BACKOFF_SECONDS * (2**attempt))
            self._user_history_team_id = None
            return False

    async def _fetch_thread(
        self,
        job: SlackMentionJob,
        *,
        client: AsyncWebClient | SlackHistoryClient,
        source: Literal["slack_replies_bot", "slack_replies_user"],
    ) -> _ThreadFetch:
        """Fetch the exact destination thread before any background history."""

        messages: list[_HistoryMessage] = []
        seen_message_ts: set[str] = set()
        seen_cursors: set[str] = set()
        cursor = ""
        requests = 0
        scanned = 0
        failed = False
        truncated = False

        while scanned < self._max_thread_messages and requests < self._max_thread_pages:
            request_limit = min(self._page_size, self._max_thread_messages - scanned)
            requests += 1
            try:
                if cursor:
                    response = await client.conversations_replies(
                        channel=job.channel_id,
                        cursor=cursor,
                        inclusive=False,
                        latest=job.message_ts,
                        limit=request_limit,
                        ts=job.thread_root_ts,
                    )
                else:
                    response = await client.conversations_replies(
                        channel=job.channel_id,
                        inclusive=False,
                        latest=job.message_ts,
                        limit=request_limit,
                        ts=job.thread_root_ts,
                    )
                payload = getattr(response, "data", response)
                page_messages, next_cursor, has_more = _history_page(payload)
            except Exception:
                failed = True
                break

            remaining = self._max_thread_messages - scanned
            bounded_page = page_messages[:remaining]
            if len(page_messages) > remaining:
                truncated = True
            scanned += len(bounded_page)
            for raw in bounded_page:
                message = _eligible_thread_message(job, raw)
                if message is None or message.message_ts in seen_message_ts:
                    continue
                seen_message_ts.add(message.message_ts)
                messages.append(message)

            if not next_cursor:
                if has_more:
                    truncated = True
                break
            if next_cursor in seen_cursors:
                truncated = True
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if scanned >= self._max_thread_messages and cursor:
            truncated = True
        elif requests >= self._max_thread_pages and cursor:
            truncated = True
        return _ThreadFetch(
            messages=tuple(messages),
            requests=requests,
            raw_messages_scanned=scanned,
            failed=failed,
            truncated=truncated,
            source=source,
        )

    async def _refresh_persisted_root_coverage(
        self,
        job: SlackMentionJob,
        *,
        client: AsyncWebClient | SlackHistoryClient,
        source: SlackThreadCoverageSource,
    ) -> tuple[bool, int, int]:
        """Attest coverage only from an exact root returned by conversations.history."""

        if self._thread_fallback is None:
            return False, 0, 0
        try:
            response = await client.conversations_history(
                channel=job.channel_id,
                oldest=job.thread_root_ts,
                latest=job.thread_root_ts,
                inclusive=True,
                limit=1,
            )
            payload = getattr(response, "data", response)
            page_messages, next_cursor, has_more = _history_page(payload)
        except Exception:
            return False, 1, 0
        if has_more or next_cursor or len(page_messages) != 1:
            return False, 1, len(page_messages)
        root = page_messages[0]
        if root.get("type", "message") != "message" or root.get("ts") != job.thread_root_ts:
            return False, 1, 1
        try:
            recorded = await self._thread_fallback.record_root_coverage(
                team_id=job.team_id,
                channel_id=job.channel_id,
                thread_root_ts=job.thread_root_ts,
                current_message_ts=job.message_ts,
                raw_root=root,
                source=source,
            )
        except Exception:
            return False, 1, 1
        return recorded, 1, 1

    async def _fetch_conversation(
        self,
        job: SlackMentionJob,
        conversation_id: str,
        *,
        message_limit: int,
        semaphore: asyncio.Semaphore,
        page_budget: _GlobalPageBudget,
    ) -> _ConversationFetch:
        messages: list[_HistoryMessage] = []
        seen_message_ts: set[str] = set()
        seen_cursors: set[str] = set()
        cursor = ""
        requests = 0
        scanned = 0
        loaded = False
        failed = False
        truncated = False

        while scanned < message_limit and requests < self._max_pages_per_conversation:
            if not await page_budget.claim():
                truncated = True
                break
            request_limit = min(self._page_size, message_limit - scanned)
            requests += 1
            try:
                async with semaphore:
                    if cursor:
                        response = await self._client.conversations_history(
                            channel=conversation_id,
                            cursor=cursor,
                            inclusive=False,
                            latest=job.message_ts,
                            limit=request_limit,
                        )
                    else:
                        response = await self._client.conversations_history(
                            channel=conversation_id,
                            inclusive=False,
                            latest=job.message_ts,
                            limit=request_limit,
                        )
                payload = getattr(response, "data", response)
                page_messages, next_cursor, has_more = _history_page(payload)
                loaded = True
            except Exception:
                failed = True
                break

            remaining = message_limit - scanned
            bounded_page = page_messages[:remaining]
            if len(page_messages) > remaining:
                truncated = True
            scanned += len(bounded_page)
            for raw in bounded_page:
                message = _eligible_message(job, conversation_id, raw)
                if message is None or message.message_ts in seen_message_ts:
                    continue
                seen_message_ts.add(message.message_ts)
                messages.append(message)

            if not next_cursor:
                if has_more:
                    truncated = True
                break
            if next_cursor in seen_cursors:
                truncated = True
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if scanned >= message_limit:
            truncated = truncated or bool(cursor)
        elif requests >= self._max_pages_per_conversation and cursor:
            truncated = True
        return _ConversationFetch(
            conversation_id=conversation_id,
            messages=tuple(messages),
            requests=requests,
            raw_messages_scanned=scanned,
            loaded=loaded,
            failed=failed,
            truncated=truncated,
        )


def _history_page(
    payload: object,
) -> tuple[list[Mapping[str, object]], str, bool]:
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise RuntimeError("Slack history API did not return a successful response")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise RuntimeError("Slack history API returned malformed messages")
    messages = [item for item in raw_messages if isinstance(item, Mapping)]
    metadata = payload.get("response_metadata")
    cursor = metadata.get("next_cursor") if isinstance(metadata, Mapping) else ""
    if not isinstance(cursor, str):
        cursor = ""
    return messages, cursor, payload.get("has_more") is True


def _persisted_thread_fetch(
    job: SlackMentionJob,
    snapshot: PersistedSlackThreadSnapshot,
    *,
    requests: int,
    raw_messages_scanned: int,
) -> _ThreadFetch | None:
    """Validate the exact fallback envelope again at the consumer boundary."""

    if (
        not snapshot.complete
        or snapshot.coverage_reason is not SlackThreadCoverageReason.COMPLETE
        or snapshot.team_id != job.team_id
        or snapshot.channel_id != job.channel_id
        or snapshot.thread_root_ts != job.thread_root_ts
        or snapshot.current_message_ts != job.message_ts
        or snapshot.complete_through_ts != job.message_ts
        or snapshot.authoritative_reply_count is None
        or snapshot.persisted_message_count != snapshot.authoritative_reply_count + 1
        or len(snapshot.messages) > snapshot.authoritative_reply_count
        or not snapshot.boundary_attested
        or snapshot.boundary_actor_id != job.user_id
        or snapshot.boundary_event_id != job.event_id
        or snapshot.coverage_source is None
        or snapshot.coverage_snapshot_hash is None
        or re.fullmatch(r"[0-9a-f]{64}", snapshot.coverage_snapshot_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", snapshot.coverage_digest) is None
    ):
        return None
    current_ts = _valid_slack_timestamp(job.message_ts)
    latest_ts = _valid_slack_timestamp(snapshot.authoritative_latest_reply_ts)
    if (
        current_ts is None
        or snapshot.authoritative_reply_count < 1
        or latest_ts is None
        or latest_ts < current_ts
    ):
        return None
    seen_ids: set[str] = set()
    seen_timestamps: set[str] = set()
    messages: list[_HistoryMessage] = []
    prior_key: Decimal | None = None
    for row in snapshot.messages:
        message_ts = _valid_slack_timestamp(row.message_ts)
        if (
            not row.id
            or row.id in seen_ids
            or not row.actor_id
            or row.role not in {"user", "assistant"}
            or not row.text.strip()
            or message_ts is None
            or message_ts >= current_ts
            or row.message_ts in seen_timestamps
            or (prior_key is not None and message_ts <= prior_key)
        ):
            return None
        seen_ids.add(row.id)
        seen_timestamps.add(row.message_ts)
        prior_key = message_ts
        messages.append(
            _HistoryMessage(
                conversation_id=job.channel_id,
                message_ts=row.message_ts,
                actor_id=row.actor_id,
                actor_kind=row.role,
                text=row.text.strip()[:12_000],
            )
        )
    if not any(message.message_ts == job.thread_root_ts for message in messages):
        return None
    return _ThreadFetch(
        messages=tuple(messages),
        requests=requests,
        raw_messages_scanned=raw_messages_scanned,
        failed=False,
        truncated=False,
        source="persisted_complete",
        coverage_reason=snapshot.coverage_reason.value,
        coverage_digest=snapshot.coverage_digest,
    )


def _eligible_message(
    job: SlackMentionJob,
    conversation_id: str,
    raw: Mapping[str, object],
) -> _HistoryMessage | None:
    message_ts = raw.get("ts")
    text = raw.get("text")
    subtype = raw.get("subtype")
    actor = _message_actor(raw)
    parsed_ts = _valid_slack_timestamp(message_ts)
    current_ts = _valid_slack_timestamp(job.message_ts)
    if (
        raw.get("type", "message") != "message"
        or (isinstance(subtype, str) and subtype in _NON_CONVERSATIONAL_SUBTYPES)
        or raw.get("hidden") is True
        or not isinstance(message_ts, str)
        or not message_ts
        or parsed_ts is None
        or current_ts is None
        or parsed_ts >= current_ts
        or message_ts == job.message_ts
        or raw.get("event_id") == job.event_id
        or raw.get("client_msg_id") == job.event_id
        or actor is None
        or actor[0] == "USLACKBOT"
        or not isinstance(text, str)
        or not text.strip()
    ):
        return None
    return _HistoryMessage(
        conversation_id=conversation_id,
        message_ts=message_ts,
        actor_id=actor[0],
        actor_kind=actor[1],
        text=text.strip()[:12_000],
    )


def _eligible_thread_message(
    job: SlackMentionJob,
    raw: Mapping[str, object],
) -> _HistoryMessage | None:
    message_ts = raw.get("ts")
    if not isinstance(message_ts, str):
        return None
    if message_ts != job.thread_root_ts and raw.get("thread_ts") != job.thread_root_ts:
        return None
    return _eligible_message(job, job.channel_id, raw)


def _message_actor(raw: Mapping[str, object]) -> tuple[str, str] | None:
    bot_id = raw.get("bot_id")
    if isinstance(bot_id, str) and bot_id:
        return f"bot:{bot_id}", "bot"
    bot_profile = raw.get("bot_profile")
    profile_id = bot_profile.get("id") if isinstance(bot_profile, Mapping) else None
    if isinstance(profile_id, str) and profile_id:
        return f"bot:{profile_id}", "bot"
    app_id = raw.get("app_id")
    if isinstance(app_id, str) and app_id:
        return f"app:{app_id}", "app"
    user_id = raw.get("user")
    if isinstance(user_id, str) and user_id:
        return user_id, "user"
    return None


def _context_item(
    job: SlackMentionJob,
    message: _HistoryMessage,
    *,
    thread: bool = False,
    thread_retention: tuple[ContextItemRetention, int] | None = None,
    objective: str = "",
) -> ContextItem:
    context_role = (
        "exact thread"
        if thread
        else "DM continuity (background only; never overrides the current request)"
    )
    label = (
        f"[Slack {context_role}; "
        f"team={job.team_id}; conversation={message.conversation_id}; "
        f"message_ts={message.message_ts}; author={message.actor_id}; "
        f"author_kind={message.actor_kind}]"
    )
    if thread:
        if thread_retention is None:
            raise ValueError("thread context requires a chronological retention class")
        retention, priority = thread_retention
    else:
        retention = None
        priority = min(89, 60 + round(_relevance(message.text, objective) * 25))
    return ContextItem(
        id=(
            f"slack-{'thread' if thread else 'history'}:{job.team_id}:"
            f"{message.conversation_id}:{message.message_ts}"
        ),
        kind=ContextItemKind.CONVERSATION_TURN,
        content=f"{label}\n{message.text}",
        conversation_id=message.conversation_id,
        source_actor_id=message.actor_id,
        **(
            {"retention": retention, "budget_priority": priority}
            if retention is not None
            else {"budget_priority": priority}
        ),
    )


def _rank_key(message: _HistoryMessage, objective: str) -> tuple[float, Decimal, str, str]:
    return (
        -_relevance(message.text, objective),
        -_slack_timestamp(message.message_ts),
        message.conversation_id,
        message.message_ts,
    )


def _relevance(content: str, objective: str) -> float:
    query = _tokens(objective)
    if not query:
        return 0.0
    return len(query & _tokens(content)) / len(query)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z0-9_.-]{2,64}", value.lower()))


def _slack_timestamp(value: str) -> Decimal:
    return _valid_slack_timestamp(value) or Decimal(0)


def _valid_slack_timestamp(value: object) -> Decimal | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed > 0 else None


def _selection_digest(items: tuple[ContextItem, ...]) -> str:
    encoded = json.dumps(
        [item.model_dump(mode="json") for item in items],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _allocate_message_quotas(
    conversation_ids: tuple[str, ...],
    *,
    per_conversation: int,
    global_limit: int,
) -> dict[str, int]:
    quotas = dict.fromkeys(conversation_ids, 0)
    remaining = global_limit
    for _ in range(per_conversation):
        for conversation_id in conversation_ids:
            if remaining == 0:
                return quotas
            quotas[conversation_id] += 1
            remaining -= 1
    return quotas


def _bounded(value: int, name: str, hard_maximum: int) -> None:
    if value < 1 or value > hard_maximum:
        raise ValueError(f"{name} must be between 1 and {hard_maximum}")
