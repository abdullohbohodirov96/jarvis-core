# JARVIS — Desktop AI Assistant

An always-on, voice-activated desktop assistant powered by OpenAI GPT-4o,
Whisper STT, Edge TTS, Telegram, and PyAutoGUI desktop automation.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Voice Setup](#voice-setup)
4. [REST API](#rest-api)
5. [Telegram Setup](#telegram-setup)
6. [Desktop Automation](#desktop-automation)
7. [Docker](#docker)
8. [Architecture](#architecture)

---

## Quick Start

**Requirements:** Python 3.11+, a microphone (for voice mode), an OpenAI API key.

```bash
# 1. Clone the repo and enter the jarvis/ directory
git clone <repo-url>
cd jarvis-core/jarvis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install OS-level audio dependencies (Linux only)
sudo apt-get install ffmpeg portaudio19-dev libsndfile1 libgl1

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Configure environment variables
cp .env.example .env
# Open .env and set at minimum: OPENAI_API_KEY

# 6. Run JARVIS
python main.py
```

On first launch JARVIS will:

- Create the SQLite database (`jarvis.db`)
- Download the Whisper `base` model (~150 MB, cached in `~/.cache/whisper`)
- Start listening for the wake word **"Jarvis"**

Say **"Jarvis"** and give a voice command.

---

## Configuration

All settings live in the `.env` file (copy from `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(none)_ | **[Required]** OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | Chat completion model |
| `OPENAI_MAX_TOKENS` | `2048` | Max tokens per response |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8765` | FastAPI listen port |
| `API_SECRET_KEY` | `change-me` | JWT signing secret |
| `DATABASE_URL` | `sqlite+aiosqlite:///./jarvis.db` | SQLAlchemy async URL |
| `DEBUG` | `false` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `ENVIRONMENT` | `production` | `development` or `production` |

See `.env.example` for the full list with inline documentation.

---

## Voice Setup

### Requirements

| Requirement | Notes |
|---|---|
| Microphone | Any USB or built-in mic works |
| `ffmpeg` | Required by Whisper for audio decoding |
| `portaudio19-dev` | Required by PyAudio / sounddevice |

### Whisper Model

Set `WHISPER_MODEL` in `.env` to control the accuracy/speed tradeoff:

| Model | RAM | Speed | Accuracy |
|---|---|---|---|
| `tiny` | ~75 MB | Very fast | Lower |
| `base` | ~150 MB | Fast | Good **(default)** |
| `small` | ~500 MB | Medium | Better |
| `medium` | ~1.5 GB | Slow | High |
| `large` | ~3 GB | Very slow | Best |

Models are downloaded automatically on first use and cached in `~/.cache/whisper`.

### Wake Word

The default wake word is **"Jarvis"** (case-insensitive).
Change it with `WAKE_WORD=<your word>` in `.env`.

### TTS Voice

JARVIS uses Microsoft Edge TTS (free, no API key).
List all available voices:

```bash
edge-tts --list-voices
```

Set your preferred voice with `TTS_VOICE` in `.env`, for example:

```
TTS_VOICE=en-GB-RyanNeural
```

### Disable Voice

```bash
python main.py --no-voice    # Text/API mode only
python main.py --api-only    # Same, plus skips all non-API tasks
```

---

## REST API

The FastAPI server starts automatically at `http://localhost:8765`.
Interactive docs: `http://localhost:8765/docs`

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns status, version, assistant state |
| `POST` | `/api/commands/chat` | Send a text command; returns AI speech + action |
| `POST` | `/api/tasks` | Create a task directly |
| `GET` | `/api/tasks` | List all pending tasks |
| `PATCH` | `/api/tasks/{id}/complete` | Mark a task complete |
| `GET` | `/api/state` | Current AssistantState (idle, listening, etc.) |

### Example: text command

```bash
curl -X POST http://localhost:8765/api/commands/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "What are my pending tasks?"}'
```

Response:

```json
{
  "speech": "You have 2 pending tasks: 1. Review PR; 2. Call dentist.",
  "action": null
}
```

---

## Telegram Setup

JARVIS uses a **Telethon userbot** — it logs in as *your* Telegram account
(not a bot token).  This lets it read messages and send replies just like you
would.

### Step-by-step

1. Go to [https://my.telegram.org/apps](https://my.telegram.org/apps) and log
   in with your phone number.
2. Create a new application (any name/platform).
3. Copy the **App api_id** and **App api_hash**.
4. Add them to `.env`:

```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
TELEGRAM_PHONE=+14155552671
TELEGRAM_SESSION=/tmp/jarvis_telegram
```

5. Start JARVIS.  On the **first run** Telegram will send a verification code
   to your phone.  Enter it when prompted:

```
Enter the code Telegram sent to +14155552671: 12345
```

A session file is saved to `TELEGRAM_SESSION` so subsequent launches do not
require re-authentication.

### Two-factor authentication (2FA)

If your Telegram account has 2FA enabled, you will be prompted for your
Telegram password after entering the SMS code.

### Disable Telegram

```bash
python main.py --no-telegram
```

or leave `TELEGRAM_API_ID=0` in `.env`.

---

## Desktop Automation

JARVIS uses PyAutoGUI to control the desktop.  Set
`DESKTOP_AUTOMATION_ENABLED=true` in `.env` (the default).

### Supported actions

| Action | Voice example | API params |
|---|---|---|
| Open application | *"Open Chrome"* | `{"app": "google-chrome"}` |
| Type text | *"Type 'Hello world'"* | `{"text": "Hello world"}` |
| Open URL | *"Open github.com"* | `{"url": "https://github.com"}` |
| Take screenshot | *"Take a screenshot"* | `{}` |
| Web search | *"Search for Python asyncio"* | `{"query": "Python asyncio"}` |

### Linux display

On a headless Linux machine (e.g. a remote server) you need a virtual display:

```bash
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
python main.py
```

Or use the Docker setup which starts Xvfb automatically.

---

## Docker

Docker runs JARVIS in **API-only** mode (no microphone access).
Use it to expose the REST API or run Telegram integration from a server.

### Build and start

```bash
cd jarvis/
cp .env.example .env         # fill in your keys
docker compose up --build
```

The REST API will be available at `http://localhost:8765`.

### Voice in Docker (Linux only)

Docker containers cannot access the host microphone by default.
For voice support on Linux, use host networking and expose `/dev/snd`:

```yaml
# In docker-compose.yml, replace the jarvis service definition with:
services:
  jarvis:
    build: .
    network_mode: host
    devices:
      - /dev/snd:/dev/snd
    env_file: .env
    command: ["python", "main.py"]
```

Then rebuild and start:

```bash
docker compose up --build
```

### Persisted data

The compose file creates three named Docker volumes:

| Volume | Contents |
|---|---|
| `jarvis_db` | SQLite database (`jarvis.db`) |
| `jarvis_screenshots` | Screenshot output files |
| `jarvis_telegram_session` | Telegram session file (avoids re-auth) |

---

## Architecture

```
jarvis/
├── main.py                  # Entry point — CLI args, banner, asyncio.run()
├── assistant/
│   └── runner.py            # JarvisRunner — master orchestrator / state machine
├── ai/
│   ├── agent.py             # JarvisAgent — OpenAI chat completions
│   └── prompts.py           # System prompts and helpers
├── voice/
│   ├── wake_word.py         # WakeWordDetector (Whisper-based)
│   ├── pipeline.py          # VoicePipeline — record_until_silence()
│   ├── stt.py               # SpeechToText wrapper (Whisper)
│   └── tts.py               # TextToSpeech wrapper (Edge TTS)
├── desktop/
│   └── controller.py        # DesktopController — PyAutoGUI actions
├── telegram/
│   └── client.py            # JarvisTelegramClient (Telethon)
├── memory/
│   ├── task_store.py        # TaskStore — async CRUD for tasks
│   └── conversation_store.py # ConversationStore — history management
├── api/
│   └── routes/              # FastAPI routers (health, chat, tasks, …)
├── core/
│   ├── database.py          # SQLAlchemy async engine + helpers
│   ├── events.py            # Internal async event bus
│   └── logger.py            # Loguru structured logging
└── config/
    └── settings.py          # Pydantic BaseSettings — all configuration
```

### State machine

```
IDLE → LISTENING_FOR_WAKE → ACTIVATED → LISTENING → PROCESSING
     → SPEAKING → EXECUTING → IDLE (loop)
```

### Event bus topics

| EventType | Published by | Consumed by |
|---|---|---|
| `WAKE_WORD_DETECTED` | WakeWordDetector | runner |
| `VOICE_COMMAND_RECEIVED` | VoicePipeline | runner |
| `AI_RESPONSE_READY` | JarvisAgent | runner |
| `ACTION_REQUESTED` | JarvisAgent | runner → subsystems |
| `ACTION_COMPLETED` | runner | (logging) |
| `TASK_CREATED` | agent / API | TaskStore |
| `TELEGRAM_MESSAGE_RECEIVED` | TelegramClient | runner |
| `SHUTDOWN` | signal handler | runner |
