#!/usr/bin/env bash
# =============================================================================
# JARVIS — Application startup script
# =============================================================================
#
# Usage:
#   ./scripts/start.sh              # Standard start
#   ./scripts/start.sh --no-build   # Skip image rebuild
#   ./scripts/start.sh --prod       # Use production compose override
#
# What this script does:
#   1. Ensures .env exists (copies from .env.example if missing)
#   2. Builds and starts all Docker services
#   3. Waits for PostgreSQL to be ready
#   4. Runs Alembic database migrations
#   5. Prints a summary of available service URLs
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Colour

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}==> $*${NC}\n"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
NO_BUILD=false
PROD=false

for arg in "$@"; do
    case $arg in
        --no-build) NO_BUILD=true ;;
        --prod)     PROD=true ;;
        --help|-h)
            echo "Usage: $0 [--no-build] [--prod]"
            echo "  --no-build  Skip docker image rebuild"
            echo "  --prod      Use production compose override"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Working directory — always run from the repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Compose command
# ---------------------------------------------------------------------------
if $PROD; then
    COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
    info "Production mode: using docker-compose.prod.yml overlay"
else
    COMPOSE_CMD="docker compose -f docker-compose.yml"
fi

# ---------------------------------------------------------------------------
# Step 1: Ensure .env exists
# ---------------------------------------------------------------------------
header "Environment configuration"

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        warning ".env not found — copying from .env.example"
        cp .env.example .env
        warning "Please edit .env and fill in your API keys before proceeding!"
        echo ""
        echo "Required keys that MUST be set:"
        echo "  OPENAI_API_KEY, TELEGRAM_API_ID, TELEGRAM_API_HASH"
        echo "  TELEGRAM_PHONE, SECRET_KEY"
        echo ""
        read -rp "Press ENTER to continue anyway (or Ctrl+C to abort and edit .env first): "
    else
        error ".env.example not found — cannot create .env automatically."
        exit 1
    fi
else
    success ".env found"
fi

# Quick sanity check for the SECRET_KEY default
if grep -q "your-super-secret-key-change-this" .env 2>/dev/null; then
    warning "SECRET_KEY is still the example default — please change it before production use!"
fi

# ---------------------------------------------------------------------------
# Step 2: Build images
# ---------------------------------------------------------------------------
header "Building Docker images"

if $NO_BUILD; then
    info "Skipping build (--no-build specified)"
else
    info "Building images (this may take a few minutes on first run)..."
    ${COMPOSE_CMD} build --parallel
    success "Images built"
fi

# ---------------------------------------------------------------------------
# Step 3: Start services
# ---------------------------------------------------------------------------
header "Starting services"

info "Starting all services in detached mode..."
${COMPOSE_CMD} up -d --remove-orphans
success "Services started"

# ---------------------------------------------------------------------------
# Step 4: Wait for PostgreSQL
# ---------------------------------------------------------------------------
header "Waiting for PostgreSQL"

MAX_WAIT=90
INTERVAL=3
elapsed=0

info "Polling postgres container health (max ${MAX_WAIT}s)..."

until ${COMPOSE_CMD} exec -T postgres pg_isready -U jarvis -d jarvisdb -q 2>/dev/null; do
    if (( elapsed >= MAX_WAIT )); then
        error "PostgreSQL did not become ready within ${MAX_WAIT} seconds."
        echo ""
        echo "Diagnostics:"
        ${COMPOSE_CMD} logs --tail=30 postgres
        exit 1
    fi
    echo -n "."
    sleep "${INTERVAL}"
    (( elapsed += INTERVAL )) || true
done

echo ""
success "PostgreSQL is ready (${elapsed}s elapsed)"

# ---------------------------------------------------------------------------
# Step 5: Run Alembic migrations
# ---------------------------------------------------------------------------
header "Running database migrations"

info "Executing: alembic upgrade head"

${COMPOSE_CMD} exec -T api alembic upgrade head && {
    success "Database migrations applied"
} || {
    error "Migration failed — check the output above for details."
    echo ""
    echo "Common causes:"
    echo "  1. DATABASE_URL in .env is incorrect"
    echo "  2. The alembic/versions/ directory is empty (run 'alembic revision --autogenerate' first)"
    echo "  3. A migration script has a syntax error"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 6: Summary
# ---------------------------------------------------------------------------
header "JARVIS is ready"

echo -e "${GREEN}${BOLD}All services are running!${NC}\n"
echo -e "  ${CYAN}API (FastAPI)${NC}          http://localhost:8000"
echo -e "  ${CYAN}API Docs (Swagger)${NC}     http://localhost:8000/docs"
echo -e "  ${CYAN}API Docs (ReDoc)${NC}       http://localhost:8000/redoc"
echo -e "  ${CYAN}Health check${NC}           http://localhost:8000/health"
echo -e "  ${CYAN}Flower (Celery UI)${NC}     http://localhost:5555"
if ! $PROD; then
    echo -e "  ${CYAN}PostgreSQL${NC}             localhost:5432  (db=jarvisdb, user=jarvis)"
    echo -e "  ${CYAN}Redis${NC}                  localhost:6379"
fi

echo ""
echo -e "Useful commands:"
echo -e "  ${YELLOW}docker compose logs -f api${NC}               Tail API logs"
echo -e "  ${YELLOW}docker compose exec api bash${NC}             Shell into API container"
echo -e "  ${YELLOW}docker compose down${NC}                      Stop all services"
echo -e "  ${YELLOW}docker compose down -v${NC}                   Stop and delete volumes"
echo ""
