"""
JARVIS Assistant Runner — core orchestrator.

Ties together voice pipeline, AI agent, Telegram client, desktop automation,
task store, and the FastAPI server into a single async event loop.

State machine
-------------
    IDLE → LISTENING_FOR_WAKE → ACTIVATED → LISTENING → PROCESSING
         → SPEAKING → EXECUTING → IDLE (loop)

Usage
-----
    from assistant.runner import JarvisRunner
    runner = JarvisRunner()
    await runner.run()
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config.settings import settings
from core.database import close_database, init_database
from core.events import Event, EventType, event_bus
from core.logger import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# State enum                                                                    #
# --------------------------------------------------------------------------- #


class AssistantState(Enum):
    IDLE = "idle"
    LISTENING_FOR_WAKE = "listening_for_wake"
    ACTIVATED = "activated"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Response dataclasses                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class ActionResult:
    """Returned by _execute_action after an action completes."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Parsed response from JarvisAgent.process()."""

    speech: str
    action_type: str | None = None
    action_params: dict[str, Any] = field(default_factory=dict)
    follow_up: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# JarvisRunner                                                                   #
# --------------------------------------------------------------------------- #


class JarvisRunner:
    """
    Core orchestrator for the JARVIS desktop AI assistant.

    All subsystems are lazily initialised in ``initialize()``.  The main
    entry point is ``run()`` which blocks until a shutdown signal is received.
    """

    def __init__(self) -> None:
        self._state: AssistantState = AssistantState.IDLE
        self._state_lock = threading.Lock()
        self.is_running: bool = False

        # Lazy references to subsystems — populated by initialize()
        self._components: dict[str, Any] = {}

        # Asyncio server handle
        self._api_server: asyncio.Task | None = None  # type: ignore[type-arg]

        # Shutdown gate
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Event-bus subscription IDs (kept so we can unsubscribe cleanly)
        self._subscriptions: list[str] = []

    # ---------------------------------------------------------------- property

    @property
    def state(self) -> AssistantState:
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: AssistantState) -> None:
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            log.debug("State: {old} → {new}", old=old.value, new=new_state.value)

    # ---------------------------------------------------------------- initialize

    async def initialize(self) -> None:
        """
        Initialise all JARVIS subsystems in dependency order.

        Each step is logged.  Non-critical subsystems (Telegram) are caught so
        a missing Telegram config does not abort startup.
        """
        log.info("JARVIS initialising…")

        # 1. Database
        log.info("[1/9] Initialising database…")
        await init_database()
        self._components["db_ready"] = True

        # 2. AI agent
        log.info("[2/9] Loading AI agent…")
        self._components["agent"] = await self._load_agent()

        # 3. STT
        log.info("[3/9] Loading Speech-to-Text…")
        self._components["stt"] = await self._load_stt()

        # 4. TTS
        log.info("[4/9] Loading Text-to-Speech…")
        self._components["tts"] = await self._load_tts()

        # 5. Voice pipeline
        log.info("[5/9] Loading Voice pipeline…")
        self._components["voice_pipeline"] = await self._load_voice_pipeline()

        # 6. Wake word detector
        log.info("[6/9] Loading Wake-word detector…")
        self._components["wake_word"] = await self._load_wake_word()

        # 7. Desktop controller
        log.info("[7/9] Loading Desktop controller…")
        self._components["desktop"] = await self._load_desktop()

        # 8. Stores
        log.info("[8/9] Loading Task / Conversation stores…")
        self._components["task_store"] = await self._load_task_store()
        self._components["conv_store"] = await self._load_conv_store()

        # 9. Event subscriptions
        log.info("[9/9] Subscribing to event bus…")
        self._subscribe_events()

        # 10. Telegram (non-fatal)
        await self._try_connect_telegram()

        self.is_running = True
        log.info("JARVIS initialised and ready.")

    # ---------------------------------------------------------------- run

    async def run(self) -> None:
        """
        Main entry point.

        1. Initialise subsystems.
        2. Start FastAPI server in background.
        3. Start wake-word detection loop.
        4. Wait for shutdown.
        """
        await self.initialize()

        # Start FastAPI
        self._api_server = asyncio.create_task(
            self._start_api_server(), name="jarvis-api"
        )
        log.info(
            "FastAPI server starting on http://{host}:{port}",
            host=settings.API_HOST,
            port=settings.API_PORT,
        )

        # Start wake word loop (runs until shutdown)
        wake_task = asyncio.create_task(
            self._wake_word_loop(), name="jarvis-wake-word-loop"
        )

        # Block until shutdown signal
        await self._shutdown_event.wait()

        # Cancel background tasks
        for task in (wake_task, self._api_server):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        log.info("JARVIS event loop finished.")

    # ---------------------------------------------------------------- wake word loop

    async def _wake_word_loop(self) -> None:
        """
        Infinite voice interaction loop driven by wake-word detection.

        The loop runs until the shutdown event is set or the detector is
        stopped.
        """
        wake_word_detector = self._components.get("wake_word")
        stt = self._components.get("stt")
        tts = self._components.get("tts")
        agent = self._components.get("agent")
        voice_pipeline = self._components.get("voice_pipeline")

        while not self._shutdown_event.is_set():
            try:
                # ── Step 1: wait for wake word ──────────────────────────────
                self.state = AssistantState.LISTENING_FOR_WAKE

                if wake_word_detector is None:
                    # No wake-word detector — sleep and skip voice loop
                    await asyncio.sleep(1)
                    continue

                detected = await wake_word_detector.wait_for_wake_word()
                if not detected:
                    continue

                # ── Step 2: activation ──────────────────────────────────────
                self.state = AssistantState.ACTIVATED
                log.info("Wake word detected — activating.")

                if tts is not None:
                    try:
                        await tts.speak("Yes, sir?")
                    except Exception as exc:
                        log.warning("TTS activation sound failed: {exc}", exc=exc)

                # ── Step 3: record command ──────────────────────────────────
                self.state = AssistantState.LISTENING

                audio_data: bytes | None = None
                if voice_pipeline is not None:
                    try:
                        audio_data = await voice_pipeline.record_until_silence()
                    except Exception as exc:
                        log.warning("Voice recording failed: {exc}", exc=exc)

                if audio_data is None:
                    self.state = AssistantState.IDLE
                    continue

                # ── Step 4: transcribe ──────────────────────────────────────
                self.state = AssistantState.PROCESSING
                transcript = ""

                if stt is not None:
                    try:
                        transcript = await stt.transcribe(audio_data)
                        log.info("Transcript: {t!r}", t=transcript)
                    except Exception as exc:
                        log.warning("STT failed: {exc}", exc=exc)

                if not transcript.strip():
                    self.state = AssistantState.IDLE
                    continue

                # ── Step 5: process with agent ──────────────────────────────
                agent_response: AgentResponse | None = None
                if agent is not None:
                    try:
                        agent_response = await self._call_agent(transcript)
                    except Exception as exc:
                        log.error("Agent processing failed: {exc}", exc=exc)
                        agent_response = AgentResponse(
                            speech="I encountered an error processing that, sir."
                        )

                # ── Step 6: speak response ──────────────────────────────────
                self.state = AssistantState.SPEAKING

                if agent_response and tts is not None:
                    try:
                        await tts.speak(agent_response.speech)
                    except Exception as exc:
                        log.warning("TTS speak failed: {exc}", exc=exc)

                # ── Step 7: execute action ──────────────────────────────────
                if agent_response and agent_response.action_type:
                    self.state = AssistantState.EXECUTING
                    try:
                        result = await self._execute_action(
                            agent_response.action_type,
                            agent_response.action_params,
                        )
                        if result.message and tts is not None:
                            await tts.speak(result.message)
                    except Exception as exc:
                        log.error("Action execution failed: {exc}", exc=exc)

                # ── Step 8: loop back ───────────────────────────────────────
                self.state = AssistantState.IDLE

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Wake word loop error: {exc}", exc=exc)
                self.state = AssistantState.ERROR
                await asyncio.sleep(2)
                self.state = AssistantState.IDLE

        log.info("Wake word loop exited.")

    # ---------------------------------------------------------------- _handle_event

    async def _handle_event(self, event: Event) -> None:
        """
        Route incoming event-bus events to the appropriate handler.
        """
        etype = event.type
        data = event.data

        if etype == EventType.ACTION_REQUESTED:
            action_type = data.get("action_type", "")
            params = data.get("params", {})
            try:
                await self._execute_action(action_type, params)
            except Exception as exc:
                log.error("Event ACTION_REQUESTED handler error: {exc}", exc=exc)

        elif etype == EventType.TASK_CREATED:
            task_store = self._components.get("task_store")
            if task_store is not None:
                try:
                    await task_store.create(data)
                except Exception as exc:
                    log.warning("Task store create failed: {exc}", exc=exc)

        elif etype == EventType.TELEGRAM_MESSAGE_RECEIVED:
            # Only process if it directly mentions JARVIS or is a command
            text = data.get("text", "")
            if text and ("jarvis" in text.lower() or data.get("is_command")):
                try:
                    response = await self._call_agent(text)
                    tts = self._components.get("tts")
                    if tts is not None:
                        await tts.speak(response.speech)
                except Exception as exc:
                    log.warning(
                        "Telegram message processing failed: {exc}", exc=exc
                    )

        elif etype == EventType.SHUTDOWN:
            log.info("Shutdown event received.")
            await self.shutdown()

        else:
            log.debug("Unhandled event type: {t}", t=etype.name)

    # ---------------------------------------------------------------- _execute_action

    async def _execute_action(
        self, action_type: str, params: dict[str, Any]
    ) -> ActionResult:
        """
        Route an action to the correct subsystem and return the result.

        All action handlers are guarded; errors produce an ActionResult with
        success=False rather than propagating.
        """
        desktop = self._components.get("desktop")
        telegram = self._components.get("telegram")
        task_store = self._components.get("task_store")
        tts = self._components.get("tts")

        log.info(
            "Executing action: {action} params={params}",
            action=action_type,
            params=params,
        )

        try:
            if action_type == "SEND_TELEGRAM":
                if telegram is None:
                    return ActionResult(False, "Telegram is not connected, sir.")
                recipient = params.get("recipient") or params.get("name", "")
                message = params.get("message", "")
                await telegram.send_message_by_name(recipient, message)
                result = ActionResult(
                    True, f"Message sent to {recipient}, sir."
                )

            elif action_type == "OPEN_APP":
                if desktop is None:
                    return ActionResult(False, "Desktop automation is unavailable.")
                app_name = params.get("app", params.get("name", ""))
                await desktop.open_app(app_name)
                result = ActionResult(True, f"Opening {app_name}.")

            elif action_type == "TYPE_TEXT":
                if desktop is None:
                    return ActionResult(False, "Desktop automation is unavailable.")
                text = params.get("text", "")
                await desktop.type_text(text)
                result = ActionResult(True, "Text typed.")

            elif action_type == "OPEN_URL":
                if desktop is None:
                    return ActionResult(False, "Desktop automation is unavailable.")
                url = params.get("url", "")
                await desktop.open_url(url)
                result = ActionResult(True, f"Opening {url}.")

            elif action_type == "CREATE_TASK":
                if task_store is None:
                    return ActionResult(False, "Task store is unavailable.")
                task = await task_store.create(params)
                title = params.get("title", "task")
                result = ActionResult(True, f"Task '{title}' created, sir.")

            elif action_type == "COMPLETE_TASK":
                if task_store is None:
                    return ActionResult(False, "Task store is unavailable.")
                task_id = params.get("task_id") or params.get("id")
                title = params.get("title", "")
                await task_store.complete(task_id=task_id, title=title)
                label = title or str(task_id)
                result = ActionResult(True, f"Task '{label}' marked complete, sir.")

            elif action_type == "LIST_TASKS":
                if task_store is None:
                    return ActionResult(False, "Task store is unavailable.")
                tasks = await task_store.get_all()
                if not tasks:
                    speech = "You have no pending tasks, sir."
                else:
                    lines = [
                        f"{i + 1}. {t.get('title', 'Unknown')}"
                        for i, t in enumerate(tasks[:10])
                    ]
                    speech = "Your pending tasks: " + "; ".join(lines) + "."
                result = ActionResult(True, speech, {"tasks": tasks})

            elif action_type == "TAKE_SCREENSHOT":
                if desktop is None:
                    return ActionResult(False, "Desktop automation is unavailable.")
                path = await desktop.take_screenshot()
                result = ActionResult(
                    True, f"Screenshot saved to {path}.", {"path": path}
                )

            elif action_type == "SEARCH_WEB":
                if desktop is None:
                    return ActionResult(False, "Desktop automation is unavailable.")
                query = params.get("query", "")
                await desktop.search_web(query)
                result = ActionResult(True, f"Searching the web for '{query}'.")

            elif action_type == "SPEAK":
                # SPEAK actions are handled by the voice loop; nothing extra here.
                result = ActionResult(True, "")

            else:
                log.warning("Unknown action type: {a}", a=action_type)
                result = ActionResult(False, f"Unknown action: {action_type}")

        except Exception as exc:
            log.error(
                "Action {action} raised: {exc}", action=action_type, exc=exc
            )
            result = ActionResult(False, f"Action failed: {exc}")

        # Publish completion event
        await event_bus.publish(
            EventType.ACTION_COMPLETED,
            {
                "action_type": action_type,
                "success": result.success,
                "message": result.message,
            },
        )

        return result

    # ---------------------------------------------------------------- _start_api_server

    async def _start_api_server(self) -> None:
        """
        Start uvicorn in an asyncio-native way so it shares the event loop.
        """
        try:
            import uvicorn  # type: ignore[import]

            # Import the FastAPI app — defined in api/routes/__init__.py or similar
            app = self._get_fastapi_app()

            config = uvicorn.Config(
                app=app,
                host=settings.API_HOST,
                port=settings.API_PORT,
                log_level="warning",
                loop="none",           # reuse the running loop
                lifespan="off",        # we manage lifespan ourselves
            )
            server = uvicorn.Server(config)
            await server.serve()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("API server error: {exc}", exc=exc)

    def _get_fastapi_app(self):
        """
        Import and return the JARVIS FastAPI application.

        The app is defined in ``api/routes/__init__.py`` (or ``api/app.py``).
        We fall back to a minimal health-check app if the full app is not yet
        available so the runner stays functional during early development.
        """
        try:
            from api.routes import app  # type: ignore[import]

            return app
        except ImportError:
            pass

        try:
            from api.app import app  # type: ignore[import]

            return app
        except ImportError:
            pass

        # Minimal fallback app — health-check only
        from fastapi import FastAPI

        fallback = FastAPI(title="JARVIS", version=settings.VERSION)

        @fallback.get("/api/health")
        async def health():
            return {
                "status": "ok",
                "state": self.state.value,
                "version": settings.VERSION,
            }

        @fallback.post("/api/commands/chat")
        async def chat(body: dict):
            text = body.get("text", "")
            response = await self.text_command(text)
            return {"speech": response.speech, "action": response.action_type}

        return fallback

    # ---------------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        """
        Graceful shutdown.

        Stop each subsystem in reverse initialisation order, then signal the
        main event loop to exit.
        """
        if not self.is_running:
            return

        log.info("JARVIS shutting down…")
        self.is_running = False

        # Unsubscribe from event bus
        for sub_id in self._subscriptions:
            event_bus.unsubscribe(sub_id)
        self._subscriptions.clear()

        # Stop wake word detector
        wake_word = self._components.get("wake_word")
        if wake_word is not None:
            try:
                await wake_word.stop()
            except Exception as exc:
                log.warning("Wake word stop error: {exc}", exc=exc)

        # Stop TTS
        tts = self._components.get("tts")
        if tts is not None:
            try:
                await tts.stop()
            except Exception as exc:
                log.warning("TTS stop error: {exc}", exc=exc)

        # Disconnect Telegram
        telegram = self._components.get("telegram")
        if telegram is not None:
            try:
                await telegram.stop()
            except Exception as exc:
                log.warning("Telegram disconnect error: {exc}", exc=exc)

        # Close database
        try:
            await close_database()
        except Exception as exc:
            log.warning("DB close error: {exc}", exc=exc)

        # Signal event loop to exit
        self._shutdown_event.set()
        log.info("JARVIS shutdown complete.")

    # ---------------------------------------------------------------- text_command

    async def text_command(self, text: str) -> AgentResponse:
        """
        Process a text command directly (used by the REST API).

        Parameters
        ----------
        text:
            Raw command text from the user.

        Returns
        -------
        AgentResponse
            Parsed agent response including speech and optional action.
        """
        if not text.strip():
            return AgentResponse(speech="I didn't receive any input, sir.")

        try:
            response = await self._call_agent(text)

            if response.action_type:
                asyncio.create_task(
                    self._execute_action(response.action_type, response.action_params),
                    name="text-cmd-action",
                )

            return response
        except Exception as exc:
            log.error("text_command error: {exc}", exc=exc)
            return AgentResponse(
                speech="I encountered an error processing your request, sir."
            )

    # ---------------------------------------------------------------- private helpers

    def _subscribe_events(self) -> None:
        """Register handlers for relevant event types on the event bus."""
        sub_id = event_bus.subscribe(EventType.ACTION_REQUESTED, self._handle_event)
        self._subscriptions.append(sub_id)

        sub_id = event_bus.subscribe(EventType.TASK_CREATED, self._handle_event)
        self._subscriptions.append(sub_id)

        sub_id = event_bus.subscribe(
            EventType.TELEGRAM_MESSAGE_RECEIVED, self._handle_event
        )
        self._subscriptions.append(sub_id)

        sub_id = event_bus.subscribe(EventType.SHUTDOWN, self._handle_event)
        self._subscriptions.append(sub_id)

        log.debug("Event bus subscriptions registered ({n}).", n=len(self._subscriptions))

    async def _call_agent(self, text: str) -> AgentResponse:
        """
        Call the JarvisAgent and parse its JSON response into an AgentResponse.
        """
        agent = self._components.get("agent")
        if agent is None:
            return AgentResponse(speech="AI agent is not available, sir.")

        # The agent's process() method returns a raw dict or a string
        raw = await agent.process(text)

        if isinstance(raw, str):
            # Try to parse as JSON
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return AgentResponse(speech=raw)

        if not isinstance(raw, dict):
            return AgentResponse(speech=str(raw))

        speech = raw.get("speech", "")
        action_blob = raw.get("action")
        follow_up = raw.get("follow_up")

        action_type: str | None = None
        action_params: dict[str, Any] = {}

        if isinstance(action_blob, dict):
            action_type = action_blob.get("type")
            action_params = action_blob.get("params") or {}

        return AgentResponse(
            speech=speech or "Understood, sir.",
            action_type=action_type,
            action_params=action_params,
            follow_up=follow_up,
            raw=raw,
        )

    # ---------------------------------------------------------------- subsystem loaders

    async def _load_agent(self):
        """Load the JarvisAgent from ``ai`` module."""
        try:
            from ai.agent import JarvisAgent  # type: ignore[import]

            agent = JarvisAgent()
            log.info("JarvisAgent loaded.")
            return agent
        except ImportError:
            log.warning("ai.agent not found — AI agent unavailable.")
            return _StubAgent()
        except Exception as exc:
            log.error("Failed to load JarvisAgent: {exc}", exc=exc)
            return _StubAgent()

    async def _load_stt(self):
        """Load the SpeechToText engine."""
        try:
            from voice.stt import SpeechToText  # type: ignore[import]

            stt = SpeechToText(model=settings.WHISPER_MODEL)
            log.info("SpeechToText loaded (model={m}).", m=settings.WHISPER_MODEL)
            return stt
        except ImportError:
            log.warning("voice.stt not found — STT unavailable.")
            return None
        except Exception as exc:
            log.error("Failed to load SpeechToText: {exc}", exc=exc)
            return None

    async def _load_tts(self):
        """Load the TextToSpeech engine."""
        try:
            from voice.tts import TextToSpeech  # type: ignore[import]

            tts = TextToSpeech(
                voice=settings.TTS_VOICE,
                speed=settings.VOICE_SPEED,
            )
            log.info("TextToSpeech loaded (voice={v}).", v=settings.TTS_VOICE)
            return tts
        except ImportError:
            log.warning("voice.tts not found — TTS unavailable.")
            return None
        except Exception as exc:
            log.error("Failed to load TextToSpeech: {exc}", exc=exc)
            return None

    async def _load_voice_pipeline(self):
        """Load the VoicePipeline (record-until-silence)."""
        try:
            from voice.pipeline import VoicePipeline  # type: ignore[import]

            pipeline = VoicePipeline(
                sample_rate=settings.VOICE_SAMPLE_RATE,
                silence_threshold=settings.VOICE_SILENCE_THRESHOLD,
                silence_duration=settings.VOICE_SILENCE_DURATION,
            )
            log.info("VoicePipeline loaded.")
            return pipeline
        except ImportError:
            log.warning("voice.pipeline not found — voice pipeline unavailable.")
            return None
        except Exception as exc:
            log.error("Failed to load VoicePipeline: {exc}", exc=exc)
            return None

    async def _load_wake_word(self):
        """Load and start the WakeWordDetector."""
        try:
            from voice.wake_word import get_wake_word_detector  # type: ignore[import]

            detector = get_wake_word_detector(
                wake_word=settings.WAKE_WORD,
                sample_rate=settings.VOICE_SAMPLE_RATE,
            )
            await detector.start()
            log.info(
                "WakeWordDetector started (word={w!r}).", w=settings.WAKE_WORD
            )
            return detector
        except ImportError:
            log.warning("voice.wake_word not found — wake word detection unavailable.")
            return None
        except Exception as exc:
            log.error("Failed to load WakeWordDetector: {exc}", exc=exc)
            return None

    async def _load_desktop(self):
        """Load the DesktopController."""
        if not settings.DESKTOP_AUTOMATION_ENABLED:
            log.info("Desktop automation disabled by config.")
            return None

        try:
            from desktop.controller import DesktopController  # type: ignore[import]

            controller = DesktopController()
            log.info("DesktopController loaded.")
            return controller
        except ImportError:
            log.warning("desktop.controller not found — desktop automation unavailable.")
            return None
        except Exception as exc:
            log.error("Failed to load DesktopController: {exc}", exc=exc)
            return None

    async def _load_task_store(self):
        """Load the TaskStore."""
        try:
            from memory.task_store import TaskStore  # type: ignore[import]

            store = TaskStore()
            log.info("TaskStore loaded.")
            return store
        except ImportError:
            log.warning("memory.task_store not found — task store unavailable.")
            return _StubTaskStore()
        except Exception as exc:
            log.error("Failed to load TaskStore: {exc}", exc=exc)
            return _StubTaskStore()

    async def _load_conv_store(self):
        """Load the ConversationStore."""
        try:
            from memory.conversation_store import ConversationStore  # type: ignore[import]

            store = ConversationStore()
            log.info("ConversationStore loaded.")
            return store
        except ImportError:
            log.warning(
                "memory.conversation_store not found — conversation store unavailable."
            )
            return None
        except Exception as exc:
            log.error("Failed to load ConversationStore: {exc}", exc=exc)
            return None

    async def _try_connect_telegram(self) -> None:
        """
        Attempt to connect the Telegram client.

        Failure is non-fatal — JARVIS continues without Telegram if credentials
        are missing or the connection fails.
        """
        if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
            log.info("Telegram credentials not set — skipping Telegram integration.")
            return

        try:
            from telegram.client import JarvisTelegramClient  # type: ignore[import]

            client = JarvisTelegramClient(
                session_path=settings.TELEGRAM_SESSION,
                api_id=settings.TELEGRAM_API_ID,
                api_hash=settings.TELEGRAM_API_HASH,
                phone=settings.TELEGRAM_PHONE,
            )
            await client.connect()
            self._components["telegram"] = client
            log.info("Telegram client connected.")
        except ImportError:
            log.warning("telegram.client not found — Telegram unavailable.")
        except Exception as exc:
            log.warning(
                "Telegram connection failed (non-fatal): {exc}", exc=exc
            )


# --------------------------------------------------------------------------- #
# Stub implementations used when optional subsystems are absent                 #
# --------------------------------------------------------------------------- #


class _StubAgent:
    """Minimal stub used when the real JarvisAgent module is not yet written."""

    async def process(self, text: str) -> dict:
        return {
            "speech": (
                f"I received your message, sir: '{text[:80]}'. "
                "The AI engine is not yet fully loaded."
            ),
            "action": None,
            "follow_up": None,
        }


class _StubTaskStore:
    """In-memory task store stub used when the real TaskStore is absent."""

    def __init__(self) -> None:
        self._tasks: list[dict[str, Any]] = []
        self._next_id = 1

    async def create(self, data: dict) -> dict:
        task = {"id": self._next_id, **data}
        self._next_id += 1
        self._tasks.append(task)
        return task

    async def complete(self, task_id: int | None = None, title: str = "") -> bool:
        for t in self._tasks:
            if (task_id and t.get("id") == task_id) or (
                title and t.get("title") == title
            ):
                t["done"] = True
                return True
        return False

    async def get_all(self) -> list[dict]:
        return [t for t in self._tasks if not t.get("done")]
