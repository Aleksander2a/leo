"""In-memory conversation/thread pinning port used by deterministic policy tests."""

from __future__ import annotations

from typing import Protocol

from leo.domain.conversation import ConversationRef, ThreadRef
from leo.harness.models import ScopeKey


class ConversationStore(Protocol):
    def pin_thread(
        self,
        scope: ScopeKey,
        destination: ConversationRef,
        *,
        root_ts: str,
        mapping_version: int | None = None,
    ) -> ThreadRef: ...

    def load_thread(
        self, scope: ScopeKey, destination: ConversationRef, *, root_ts: str
    ) -> ThreadRef | None: ...


class ConversationStoreError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._threads: dict[tuple[ConversationRef, str], ThreadRef] = {}

    def pin_thread(
        self,
        scope: ScopeKey,
        destination: ConversationRef,
        *,
        root_ts: str,
        mapping_version: int | None = None,
    ) -> ThreadRef:
        if not root_ts or (mapping_version is not None and mapping_version < 1):
            raise ValueError("root_ts is required and mapping_version must be positive")
        key = (destination, root_ts)
        existing = self._threads.get(key)
        if existing is not None:
            if existing.scope.organization_id != scope.organization_id:
                raise ConversationStoreError("thread_organization_changed")
            return existing
        pinned = ThreadRef(
            conversation=destination,
            root_ts=root_ts,
            scope=scope,
            mapping_version=mapping_version,
        )
        self._threads[key] = pinned
        return pinned

    def load_thread(
        self, scope: ScopeKey, destination: ConversationRef, *, root_ts: str
    ) -> ThreadRef | None:
        thread = self._threads.get((destination, root_ts))
        if thread is None or thread.scope.organization_id != scope.organization_id:
            return None
        return thread
