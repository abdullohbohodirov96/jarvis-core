# JARVIS AI Assistant Platform

An intelligent personal AI assistant platform with voice interface, Telegram integration, long-term memory, task management, and a streaming chat API — all powered by GPT-4o.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Clients                           │
│  Voice (Whisper/Edge-TTS)  │  Telegram  │  HTTP/WS      │
└────────────┬───────────────┴─────┬──────┴───────┬───────┘
             │                     │              │
             ▼                     ▼              ▼
┌────────────────────────────────────────────────────────┐
│                    Nginx (TLS, proxy)                   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│              FastAPI  (Python 3.11, uvicorn)            │
│                                                        │
│   ┌─────────────┐  ┌───────────────┐  ┌─────────────┐ │
│   │  AI Agent   │  │ Voice Pipeline│  │  Telegram   │ │
│   │  (GPT-4o)   │  │ (Whisper+TTS) │  │  (Telethon) │ │
│   └──────┬──────┘  └───────────────┘  └─────────────┘ │
│          │                                             │
│   ┌──────▼──────┐  ┌───────────────┐                  │
│   │   Memory    │  │ Task Manager  │                  │
│   │  Manager   │  │ (CRUD + AI)   │                  │
│   └─────────────┘  └───────────────┘                  │
└──────────────┬─────────────────────────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌─────────┐       ┌──────────────────────────────────┐
│ Postgres│       │  Redis                           │
│  (data) │       │  DB0: cache  DB1: broker         │
└─────────┘       │  DB2: results                    │
                  └──────────────┬───────────────────┘
                                 │
                  ┌──────────────┴───────────────────┐
                  │       Celery Workers              │
                  │  • reminders      • ai_tasks      │
                  │  • voice_tasks    • telegram_tasks│
                  └───────────────────────────────────┘
```

---

## Features

- **AI Chat** — Streaming responses via GPT-4o with full conversation memory
- **Voice Interface** — Whisper STT (local or API) + Edge-TTS / ElevenLabs synthesis
- **Telegram Integration** — Telethon userbot; send/receive messages, scheduled sends
- **Task Management** — AI-powered task extraction from natural language
- **Reminders & Scheduling** — Celery beat for time-based triggers
- **Daily Summaries** — Automated morning briefings with tasks + weather + news
- **Long-term Memory** — Importance-scored memories with similarity search
- **WebSocket Chat** — Real-time bidirectional streaming in the browser
- **Observability** — Structured JSON logs, Prometheus metrics, Flower UI

---

## Quick Start

### Prerequisites

- Docker 24+ and Docker Compose v2
- An OpenAI API key
- (Optional) Telegram API credentials for the userbot

### 1. Clone the repository

```bash
git clone https://github.com/your-org/jarvis-core.git
cd jarvis-core
```

### 2. Run the setup script

The setup script installs Docker if missing, generates a secure `SECRET_KEY`,
and builds all images:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 3. Configure environment variables

```bash
# .env was created by setup.sh; fill in your API keys
nano .env
```

Required values to set:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key from https://platform.openai.com/api-keys |
| `TELEGRAM_API_ID` | From https://my.telegram.org/apps |
| `TELEGRAM_API_HASH` | From https://my.telegram.org/apps |
| `TELEGRAM_PHONE` | Your phone in E.164 format (e.g. `+12025551234`) |

### 4. Start the platform

```bash
./scripts/start.sh
```

This will:
1. Start all Docker services (Postgres, Redis, API, Celery, Flower)
2. Wait for Postgres to be ready
3. Run database migrations automatically

### 5. Access the services

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Health check | http://localhost:8000/health |
| Flower (Celery) | http://localhost:5555 |

---

## API Endpoints

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create a new user account |
| `POST` | `/api/v1/auth/login` | Obtain access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh an expired access token |
| `POST` | `/api/v1/auth/logout` | Revoke the current session |

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat/message` | Send a message, receive a full response |
| `GET` | `/api/v1/chat/stream` | SSE streaming response endpoint |
| `GET` | `/api/v1/chat/conversations` | List user conversations |
| `GET` | `/api/v1/chat/conversations/{id}` | Get conversation with message history |
| `DELETE` | `/api/v1/chat/conversations/{id}` | Archive a conversation |

### Tasks

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/tasks` | List tasks (filterable by status, priority) |
| `POST` | `/api/v1/tasks` | Create a task |
| `POST` | `/api/v1/tasks/extract` | AI-extract tasks from free text |
| `PATCH` | `/api/v1/tasks/{id}` | Update task fields |
| `DELETE` | `/api/v1/tasks/{id}` | Delete a task |

### Voice

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/voice/transcribe` | Upload audio, receive transcript |
| `POST` | `/api/v1/voice/synthesize` | Convert text to audio (returns binary) |

### Memory

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/memory` | List memory items |
| `POST` | `/api/v1/memory/search` | Semantic search over memories |
| `DELETE` | `/api/v1/memory/{id}` | Delete a memory item |

### Telegram

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/telegram/send` | Send a message via the userbot |
| `GET` | `/api/v1/telegram/chats` | List recent Telegram chats |
| `POST` | `/api/v1/telegram/schedule` | Schedule a message for later |

### WebSocket

```
ws://localhost:8000/ws/chat/{client_id}
```

**Client → Server:**
```json
{"message": "What tasks do I have today?", "conversation_id": "abc123"}
```

**Server → Client (streaming):**
```json
{"type": "token",  "content": "You have"}
{"type": "token",  "content": " 3 tasks today:"}
{"type": "done",   "conversation_id": "abc123"}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `JARVIS` | Application name shown in logs and UI |
| `ENVIRONMENT` | `production` | `development` / `staging` / `production` |
| `DEBUG` | `false` | Enable debug mode (exposes /docs in prod) |
| `SECRET_KEY` | — | JWT signing secret (min 32 chars, required) |
| `DATABASE_URL` | — | PostgreSQL asyncpg connection string |
| `REDIS_URL` | — | Redis connection string (DB 0) |
| `CELERY_BROKER_URL` | — | Redis for Celery broker (DB 1) |
| `CELERY_RESULT_BACKEND` | — | Redis for Celery results (DB 2) |
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o` | Chat completion model |
| `TELEGRAM_API_ID` | — | Telegram app ID |
| `TELEGRAM_API_HASH` | — | Telegram app hash |
| `TELEGRAM_PHONE` | — | Telegram phone number |
| `WHISPER_MODEL` | `base` | `base` / `small` / `medium` / `large` |
| `WHISPER_USE_API` | `false` | Use OpenAI Whisper API instead of local |
| `TTS_PROVIDER` | `edge` | `edge` (free) or `elevenlabs` (premium) |
| `EDGE_TTS_VOICE` | `en-US-AriaNeural` | Microsoft Edge TTS voice name |
| `ELEVENLABS_API_KEY` | — | ElevenLabs API key (if TTS_PROVIDER=elevenlabs) |
| `MAX_MEMORY_ITEMS` | `1000` | Memory items retained per user |
| `MEMORY_DECAY_HOURS` | `720` | Hours until memories are eligible for pruning |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON array) |
| `RATE_LIMIT_REQUESTS` | `100` | Requests per IP per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

---

## Development Setup (without Docker)

### Requirements

- Python 3.11
- PostgreSQL 15+
- Redis 7+
- ffmpeg (for audio processing)

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start Postgres and Redis locally, then:
cd backend

# Copy and configure .env
cp ../.env.example ../.env
# Set DATABASE_URL and REDIS_URL to localhost equivalents

# Run migrations
alembic upgrade head

# Start the API
uvicorn app.main:app --reload --port 8000

# In separate terminals:
celery -A app.workers.celery_app worker --loglevel=debug
celery -A app.workers.celery_app beat --loglevel=debug
```

### Running tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

---

## Deployment to Render

JARVIS ships with a `render.yaml` Blueprint for one-click Render deployment.

### Steps

1. Push the repository to GitHub (or GitLab).

2. In the [Render Dashboard](https://dashboard.render.com/), click **New → Blueprint**.

3. Connect your repository and select the branch to deploy.

4. Render detects `render.yaml` and shows a preview of all services it will create:
   - `jarvis-api` (Web Service — FastAPI)
   - `jarvis-worker` (Background Worker — Celery)
   - `jarvis-beat` (Background Worker — Celery Beat)
   - `jarvis-postgres` (Managed PostgreSQL)

5. Click **Apply** to provision resources.

6. After deployment, set the remaining secrets in the Render dashboard:
   - `OPENAI_API_KEY`
   - `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_PHONE`
   - `REDIS_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
     (use [Upstash](https://upstash.com/) or [Redis Cloud](https://redis.com/try-free/) free tiers)
   - `ELEVENLABS_API_KEY` (optional)

7. Trigger a new deploy after setting the secrets.

### Post-deployment: database migrations

Render does not automatically run migrations. After the first deploy:

```bash
# Using the Render Shell in the dashboard for jarvis-api:
alembic upgrade head
```

Or add it to the `buildCommand` in `render.yaml` (runs on every deploy):

```yaml
buildCommand: |
  pip install -r requirements.txt && alembic upgrade head
```

---

## Architecture Notes

### Async-first design

Every I/O operation (database, Redis, OpenAI, Telegram) uses async/await.
SQLAlchemy is configured with `asyncpg` driver; Redis uses `redis.asyncio`.
This allows a single Uvicorn worker to handle hundreds of concurrent connections.

### Agent-based AI

The AI layer is structured as a tool-calling agent.  On each turn, the agent
can invoke:
- `memory_search` — retrieve relevant past context
- `task_create` / `task_list` — manage the task backlog
- `telegram_send` — send a message via the userbot
- `web_search` — look up current information (optional plugin)

### Memory system

Memories are importance-scored (0–1) on creation by a lightweight GPT-4o-mini
classifier.  Items below `MEMORY_IMPORTANCE_THRESHOLD` (default 0.3) are
discarded.  Retrieval uses cosine similarity on OpenAI embeddings, returning
the top-K most relevant memories as additional context for each AI turn.

### Celery task queues

Five named queues isolate work by type:

| Queue | Purpose |
|---|---|
| `default` | General-purpose tasks |
| `ai_tasks` | Heavy OpenAI completion tasks |
| `voice_tasks` | Whisper transcription |
| `telegram_tasks` | Telegram send/receive |
| `reminders` | Time-triggered reminders + daily summaries |

### Security

- Passwords hashed with bcrypt (passlib)
- JWT access tokens (HS256, 30-minute TTL) + refresh tokens (7-day TTL)
- Rate limiting at both nginx and FastAPI middleware layers
- Non-root Docker user (`uid=10001`)
- All secrets managed via environment variables, never committed

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push: `git push origin feat/my-feature`
5. Open a Pull Request

Please run `pytest` and ensure all tests pass before opening a PR.

---

## License

MIT — see [LICENSE](LICENSE) for details.
