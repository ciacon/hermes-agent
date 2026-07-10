"""Tests for the platform-neutral observe-only gateway path."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import MessageDisposition, MessageEvent
from gateway.session import SessionSource, build_session_key


class _RecordingSessionStore:
    """Minimal store that exercises real room session-key construction."""

    def __init__(self) -> None:
        self.sources = []
        self.session_ids = {}
        self.rows = []

    def _generate_session_key(self, source: SessionSource) -> str:
        return build_session_key(source, group_sessions_per_user=True)

    def get_or_create_session(self, source: SessionSource):
        self.sources.append(source)
        key = self._generate_session_key(source)
        session_id = self.session_ids.setdefault(key, f"session-{len(self.session_ids) + 1}")
        return SimpleNamespace(session_id=session_id, session_key=key)

    def append_to_transcript(self, session_id: str, message: dict) -> None:
        self.rows.append((session_id, message))


def _make_source(
    *,
    user_id: str = "user-1",
    user_name: str = "Alice",
    chat_type: str = "channel",
    thread_id: str | None = "thread-1",
) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        scope_id="guild-1",
        parent_chat_id="parent-1",
        chat_id="room-1",
        chat_name="bots",
        chat_type=chat_type,
        thread_id=thread_id,
        user_id=user_id,
        user_id_alt=f"stable-{user_id}",
        user_name=user_name,
    )


def _make_event(
    *,
    text: str = "ambient message",
    user_id: str = "user-1",
    user_name: str = "Alice",
    chat_type: str = "channel",
    thread_id: str | None = "thread-1",
    disposition: MessageDisposition = MessageDisposition.OBSERVE,
    message_id: str = "message-1",
    timestamp: datetime | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(
            user_id=user_id,
            user_name=user_name,
            chat_type=chat_type,
            thread_id=thread_id,
        ),
        message_id=message_id,
        timestamp=timestamp or datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
        disposition=disposition,
    )


def _make_runner(*, authorized: bool = True):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True)},
    )
    runner: Any = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.DISCORD: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.pairing_store.generate_code.return_value = "12345"
    runner.session_store = _RecordingSessionStore()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    runner._is_user_authorized = MagicMock(return_value=authorized)
    runner._get_unauthorized_dm_behavior = MagicMock(return_value="pair")
    runner._handle_message_with_agent = AsyncMock(return_value="agent-result")
    return runner, adapter


def _set_hook(monkeypatch, results) -> None:
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: results if name == "pre_gateway_dispatch" else [],
    )


@pytest.mark.asyncio
async def test_authorized_observe_event_appends_exactly_one_observed_row(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=True)
    event = _make_event()

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == [
        (
            "session-1",
            {
                "role": "user",
                "content": "[Alice|stable-user-1]\nambient message",
                "timestamp": "2026-07-10T12:00:00+00:00",
                "observed": True,
                "message_id": "message-1",
            },
        )
    ]
    observed_source = runner.session_store.sources[0]
    assert observed_source.user_id is None
    assert observed_source.user_id_alt is None
    assert observed_source.user_name is None
    assert observed_source.scope_id == "guild-1"
    assert observed_source.parent_chat_id == "parent-1"
    assert observed_source.chat_id == "room-1"
    assert observed_source.thread_id == "thread-1"
    runner._handle_message_with_agent.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_observed_users_share_room_session_but_keep_content_attribution(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, _adapter = _make_runner(authorized=True)

    await runner._handle_message(
        _make_event(
            text="first",
            user_id="user-1",
            user_name="Alice",
            message_id="m1",
            thread_id=None,
        )
    )
    await runner._handle_message(
        _make_event(
            text="second",
            user_id="user-2",
            user_name="Bob",
            message_id="m2",
            thread_id=None,
        )
    )

    assert len(runner.session_store.session_ids) == 1
    assert [session_id for session_id, _row in runner.session_store.rows] == [
        "session-1",
        "session-1",
    ]
    assert [row["content"] for _session_id, row in runner.session_store.rows] == [
        "[Alice|stable-user-1]\nfirst",
        "[Bob|stable-user-2]\nsecond",
    ]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_unauthorized_observe_event_is_silently_dropped_before_pairing(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=False)
    event = _make_event(chat_type="dm")

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == []
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_observe_event_is_dropped_without_persistence_or_dispatch(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, adapter = _make_runner(authorized=True)
    event = _make_event()
    event.internal = True

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.rows == []
    runner._is_user_authorized.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_skip_prevents_observation_storage(monkeypatch):
    _set_hook(monkeypatch, [{"action": "skip", "reason": "plugin-handled"}])
    runner, adapter = _make_runner(authorized=True)

    result = await runner._handle_message(_make_event())

    assert result is None
    assert runner.session_store.rows == []
    runner._is_user_authorized.assert_not_called()
    runner._handle_message_with_agent.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_event_still_uses_ordinary_agent_path(monkeypatch):
    _set_hook(monkeypatch, [])
    runner, _adapter = _make_runner(authorized=True)
    event = _make_event(disposition=MessageDisposition.DISPATCH)

    result = await runner._handle_message(event)

    assert result == "agent-result"
    assert runner.session_store.rows == []
    runner._handle_message_with_agent.assert_awaited_once()
