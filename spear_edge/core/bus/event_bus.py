# spear_edge/core/bus/event_bus.py
from __future__ import annotations
import asyncio
from typing import Any, Callable, Dict, List


class EventBus:
    """
    Lightweight async pub/sub bus.

    - topic-based
    - subscribers get an asyncio.Queue
    - publishers push events without blocking (drops old if full)
    """

    def __init__(self):
        self._subs: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, maxsize: int = 8) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subs.setdefault(topic, []).append(q)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        async with self._lock:
            if topic in self._subs:
                self._subs[topic] = [qq for qq in self._subs[topic] if qq is not q]

    def publish_nowait(self, topic: str, event: Any) -> None:
        qs = self._subs.get(topic, [])
        for q in list(qs):
            try:
                if q.full():
                    _ = q.get_nowait()
                q.put_nowait(event)
            except Exception:
                pass
