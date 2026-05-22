"""
JARVIS internal async event bus.

Provides decoupled pub/sub communication between subsystems (voice, AI, tasks,
Telegram, desktop automation, …).

Usage
-----
    from core.events import event_bus, EventType, Event

    # Subscribe
    sub_id = event_bus.subscribe(EventType.AI_RESPONSE_READY, my_handler)

    # Publish (fire-and-forget)
    await event_bus.publish(EventType.AI_RESPONSE_READY, {"text": "Hello!"})

    # Publish and wait for all handlers to finish
    await event_bus.publish_and_wait(EventType.TASK_CREATED, {"task_id": 42})

    # Unsubscribe when done
    event_bus.unsubscribe(sub_id)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from core.logger import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Event types                                                                   #
# --------------------------------------------------------------------------- #


class EventType(Enum):
    WAKE_WORD_DETECTED = auto()
    VOICE_COMMAND_RECEIVED = auto()
    AI_RESPONSE_READY = auto()
    ACTION_REQUESTED = auto()
    ACTION_COMPLETED = auto()
    TELEGRAM_MESSAGE_RECEIVED = auto()
    TASK_CREATED = auto()
    TASK_COMPLETED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


# --------------------------------------------------------------------------- #
# Event dataclass                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Event(type={self.type.name}, event_id={self.event_id!r}, "
            f"timestamp={self.timestamp:.3f})"
        )


# --------------------------------------------------------------------------- #
# Handler type alias                                                             #
# --------------------------------------------------------------------------- #

# Handlers may be plain coroutine functions or sync callables.
Handler = Callable[[Event], Any]


def _is_coroutine_callable(fn: Callable) -> bool:
    return asyncio.iscoroutinefunction(fn)


# --------------------------------------------------------------------------- #
# EventBus                                                                       #
# --------------------------------------------------------------------------- #


class EventBus:
    """Lightweight async pub/sub event bus.

    Thread-safety note
    ------------------
    The bus is designed for single-threaded async use.  If you need to publish
    events from a background thread use ``asyncio.run_coroutine_threadsafe``.
    """

    def __init__(self) -> None:
        # mapping: EventType -> {subscription_id: handler}
        self._subscribers: dict[EventType, dict[str, Handler]] = defaultdict(dict)

    # ---------------------------------------------------------------- subscribe

    def subscribe(
        self,
        event_type: EventType,
        handler: Handler,
    ) -> str:
        """Register *handler* for *event_type*.

        Returns
        -------
        str
            A unique subscription ID that can be passed to :meth:`unsubscribe`.
        """
        sub_id = str(uuid.uuid4())
        self._subscribers[event_type][sub_id] = handler
        log.debug(
            "Subscribed {handler} to {event} (id={sub_id})",
            handler=getattr(handler, "__name__", repr(handler)),
            event=event_type.name,
            sub_id=sub_id,
        )
        return sub_id

    # -------------------------------------------------------------- unsubscribe

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by its ID.

        Returns ``True`` if the subscription was found and removed.
        """
        for handlers in self._subscribers.values():
            if subscription_id in handlers:
                del handlers[subscription_id]
                log.debug("Unsubscribed id={sub_id}", sub_id=subscription_id)
                return True
        log.warning(
            "Unsubscribe called for unknown id={sub_id}", sub_id=subscription_id
        )
        return False

    # --------------------------------------------------------------- _dispatch

    async def _dispatch(
        self,
        event: Event,
        *,
        gather: bool = True,
    ) -> list[Any]:
        """Dispatch *event* to all registered handlers and return their results."""
        handlers = list(self._subscribers.get(event.type, {}).values())

        if not handlers:
            log.debug("No subscribers for {event}", event=event.type.name)
            return []

        log.debug(
            "Publishing {event} to {n} subscriber(s)",
            event=event.type.name,
            n=len(handlers),
        )

        coros: list[Coroutine] = []
        for handler in handlers:
            try:
                if _is_coroutine_callable(handler):
                    coros.append(handler(event))
                else:
                    # Wrap sync handler in a coroutine so gather works uniformly.
                    async def _sync_wrapper(h: Handler = handler) -> Any:
                        return h(event)

                    coros.append(_sync_wrapper())
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Error wrapping handler {h}: {exc}",
                    h=repr(handler),
                    exc=exc,
                )

        if not coros:
            return []

        if gather:
            results = await asyncio.gather(*coros, return_exceptions=True)
        else:
            results = []
            for coro in coros:
                result = await coro
                results.append(result)

        # Log any exceptions returned by gather (they don't propagate).
        for res in results:
            if isinstance(res, Exception):
                log.error(
                    "Handler raised exception for {event}: {exc}",
                    event=event.type.name,
                    exc=res,
                )

        return results

    # ----------------------------------------------------------------- publish

    async def publish(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Fire-and-forget: call all handlers concurrently via asyncio.gather.

        Exceptions inside handlers are logged but never re-raised.
        """
        event = Event(type=event_type, data=data or {})
        await self._dispatch(event, gather=True)

    # -------------------------------------------------------- publish_and_wait

    async def publish_and_wait(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Publish *event_type* and await all handler coroutines sequentially.

        Returns the list of handler return values (exceptions included).
        """
        event = Event(type=event_type, data=data or {})
        return await self._dispatch(event, gather=False)

    # --------------------------------------------------------------- clear

    def clear(self, event_type: EventType | None = None) -> None:
        """Remove all subscriptions, optionally scoped to *event_type*."""
        if event_type is None:
            self._subscribers.clear()
            log.debug("EventBus cleared — all subscriptions removed.")
        else:
            self._subscribers.pop(event_type, None)
            log.debug(
                "EventBus cleared subscriptions for {event}",
                event=event_type.name,
            )

    # ----------------------------------------------------------------- stats

    def subscription_count(self, event_type: EventType | None = None) -> int:
        """Return the number of active subscriptions."""
        if event_type is not None:
            return len(self._subscribers.get(event_type, {}))
        return sum(len(v) for v in self._subscribers.values())


# --------------------------------------------------------------------------- #
# Module-level singleton                                                         #
# --------------------------------------------------------------------------- #

event_bus: EventBus = EventBus()
