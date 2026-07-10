"""Platform-neutral persistence helpers for ambient room observations."""

import dataclasses
from typing import Any

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def room_scoped_source(source: SessionSource) -> SessionSource:
    """Return a source keyed to the room rather than an individual speaker."""

    return dataclasses.replace(
        source,
        user_id=None,
        user_id_alt=None,
        user_name=None,
    )


def observed_message_row(event: MessageEvent) -> dict[str, Any]:
    """Build the attributed transcript row for one observed message."""

    source = event.source
    stable_user_id = source.user_id_alt or source.user_id or "unknown"
    display_name = source.user_name or stable_user_id
    return {
        "role": "user",
        "content": f"[{display_name}|{stable_user_id}]\n{event.text}",
        "timestamp": event.timestamp.isoformat(),
        "observed": True,
        "message_id": event.message_id,
    }


def persist_room_observation(session_store: Any, event: MessageEvent) -> str:
    """Append one observed row and return its room-scoped session id."""

    source = room_scoped_source(event.source)
    session = session_store.get_or_create_session(source)
    session_store.append_to_transcript(
        session.session_id,
        observed_message_row(event),
    )
    return session.session_id
