import asyncio
from fastapi import WebSocket, WebSocketDisconnect

_TOPICS = (
    "tripwire_nodes",
    "tripwire_cue",
    "tripwire_auto_reject",
    "edge_mode",
    "task_update",
    "capture_start",
    "capture_progress",
    "capture_complete",
    "aoa_cone",  # AoA cone updates (v2.0)
    "classification_result",  # ML classification results
)


async def events_ws(websocket: WebSocket, orchestrator):
    await websocket.accept()

    # Subscribe to all topics and keep (topic, queue)
    subscriptions: list[tuple[str, asyncio.Queue]] = []
    for topic in _TOPICS:
        q = await orchestrator.bus.subscribe(topic, maxsize=50)
        subscriptions.append((topic, q))

    try:
        while True:
            # Create one task per topic queue
            task_to_topic = {
                asyncio.create_task(queue.get()): topic
                for topic, queue in subscriptions
            }

            # Wait until ANY topic produces an event
            done, pending = await asyncio.wait(
                task_to_topic.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel all unused tasks
            for task in pending:
                task.cancel()

            # Get the completed task
            task = next(iter(done))
            topic = task_to_topic[task]
            payload = task.result()

            # Always send typed envelope
            await websocket.send_json({
                "type": topic,
                "payload": payload,
                "ts": (
                    payload.get("timestamp")
                    if isinstance(payload, dict)
                    else None
                ),
            })

    except WebSocketDisconnect:
        pass

    finally:
        # Cleanly unsubscribe on disconnect
        for topic, queue in subscriptions:
            try:
                await orchestrator.bus.unsubscribe(topic, queue)
            except Exception:
                pass
