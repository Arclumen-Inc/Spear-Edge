"""
Event definitions for Spear-Edge.

These are lightweight semantic events published on the internal event bus.
They are NOT heavy objects and are safe to serialize to UI / WS clients.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional


# ----------------------------------------------------------------------
# Base event
# ----------------------------------------------------------------------

@dataclass
class Event:
    """
    Generic event wrapper used by the event bus.
    """
    type: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=lambda: time())


# ----------------------------------------------------------------------
# Tripwire-related events
# ----------------------------------------------------------------------

def tripwire_cue(payload: Dict[str, Any]) -> Event:
    """
    RF cue received from a Tripwire node.
    """
    return Event(
        type="tripwire_cue",
        payload=payload,
    )


def tripwire_nodes_snapshot(nodes: list[dict]) -> Event:
    """
    Snapshot of all known Tripwire nodes (for UI display).
    """
    return Event(
        type="tripwire_nodes",
        payload={"nodes": nodes},
    )


def tripwire_auto_reject(payload: Dict[str, Any], reason: str) -> Event:
    """
    Armed-mode auto-capture rejected by policy.
    """
    return Event(
        type="tripwire_auto_reject",
        payload={
            "cue": payload,
            "reason": reason,
        },
    )


# ----------------------------------------------------------------------
# Edge / system state events
# ----------------------------------------------------------------------

def edge_mode_changed(mode: str) -> Event:
    """
    Edge mode changed (manual <-> armed).
    """
    return Event(
        type="edge_mode",
        payload={"mode": mode},
    )


def system_status(status: Dict[str, Any]) -> Event:
    """
    Periodic or on-demand system status update.
    """
    return Event(
        type="system_status",
        payload=status,
    )


# ----------------------------------------------------------------------
# Capture / ML events (placeholders, used later)
# ----------------------------------------------------------------------

def capture_started(info: Dict[str, Any]) -> Event:
    return Event(
        type="capture_started",
        payload=info,
    )


def capture_completed(info: Dict[str, Any]) -> Event:
    return Event(
        type="capture_completed",
        payload=info,
    )


def ml_classification(result: Dict[str, Any]) -> Event:
    """
    ML classification result for a capture.
    """
    return Event(
        type="ml_classification",
        payload=result,
    )
