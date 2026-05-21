#!/usr/bin/env bash
# =============================================================================
# JARVIS — Database migration script
# =============================================================================
#
# Wraps Alembic with proper error handling, logging, and safety checks.
#
# Usage:
#   ./scripts/migrate.sh                          # Apply all pending migrations
#   ./scripts/migrate.sh --revision <id>          # Upgrade to specific revision
#   ./scripts/migrate.sh --downgrade -1           # Downgrade one step
#   ./scripts/migrate.sh --downgrade <id>         # Downgrade to specific revision
#   ./scripts/migrate.sh --status                 # Show current revision
#   ./scripts/migrate.sh --history                # Show full migration history
#   ./scripts/migrate.sh --create "description"   # Generate new migration file
#   ./scripts/migrate.sh --check                  # Check for pending migrations (CI)
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
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}==> $*${NC}\n"; }

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
ACTION="upgrade"           # Default: upgrade to head
REVISION="head"
DOWNGRADE_TARGET=""
MIGRATION_MESSAGE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --revision)
            ACTION="upgrade"
            REVISION="${2:?--revision requires a value}"
            shift 2
            ;;
        --downgrade)
            ACTION="downgrade"
            DOWNGRADE_TARGET="${2:?--downgrade requires a target (e.g. -1 or a revision id)}"
            shift 2
            ;;
        --status|--current)
            ACTION="status"
            shift
            ;;
        --history)
            ACTION="history"
            shift
            ;;
        --create)
            ACTION="create"
            MIGRATION_MESSAGE="${2:?--create requires a migration message}"
            shift 2
            ;;
        --check)
            ACTION="check"
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            sed -n '/^# Usage:/,/^# ===/{ s/^# //; p }' "$0" | head -n 20
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Determine execution context: inside Docker Compose or bare metal
# ---------------------------------------------------------------------------
# If the API container is running, exec into it. Otherwise run locally.
EXEC_CMD=""

if command -v docker &>/dev/null && docker compose ps api --status running 2>/dev/null | grep -q "running"; then
    EXEC_CMD="docker compose exec -T api"
    info "Running migration inside 'api' container"
else
    # Verify that alembic is available locally
    if ! command -v alembic &>/dev/null; then
        error "alembic not found locally and the 'api' container is not running."
        echo "  Start the stack first:  ./scripts/start.sh"
        echo "  Or install locally:     pip install alembic"
        exit 1
    fi
    info "Running migration locally (no running container detected)"
fi

# Timestamp for log prefixes
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S UTC')"

# ---------------------------------------------------------------------------
# Run migration
# ---------------------------------------------------------------------------
header "Database migration — ${TIMESTAMP}"

run_alembic() {
    if [[ -n "${EXEC_CMD}" ]]; then
        ${EXEC_CMD} alembic "$@"
    else
        alembic "$@"
    fi
}

case "${ACTION}" in

    upgrade)
        info "Applying migrations up to: ${REVISION}"
        run_alembic upgrade "${REVISION}" && {
            success "Migration complete (upgraded to ${REVISION})"
        } || {
            error "Migration failed — see output above."
            echo ""
            echo "Troubleshooting:"
            echo "  1. Check DATABASE_URL in .env points to the correct host"
            echo "  2. Ensure the database user has DDL privileges"
            echo "  3. Review the migration file in alembic/versions/ for errors"
            exit 1
        }
        ;;

    downgrade)
        warning "Downgrading to: ${DOWNGRADE_TARGET}"
        read -rp "Are you sure you want to downgrade? This may lose data. [y/N] " confirm
        if [[ "${confirm,,}" != "y" ]]; then
            info "Downgrade cancelled."
            exit 0
        fi
        run_alembic downgrade "${DOWNGRADE_TARGET}" && {
            success "Downgrade complete"
        } || {
            error "Downgrade failed."
            exit 1
        }
        ;;

    status)
        info "Current database revision:"
        run_alembic current --verbose
        ;;

    history)
        info "Migration history:"
        run_alembic history --verbose
        ;;

    create)
        info "Generating new migration: ${MIGRATION_MESSAGE}"
        run_alembic revision \
            --autogenerate \
            --message "${MIGRATION_MESSAGE}" && {
            success "Migration file created in alembic/versions/"
            echo ""
            warning "Review the generated file carefully before applying."
            warning "Autogenerate does NOT detect: renames, constraint changes on existing data."
        } || {
            error "Failed to generate migration."
            exit 1
        }
        ;;

    check)
        # Exit 1 if there are pending (unapplied) migrations — useful in CI pipelines
        info "Checking for pending migrations..."
        PENDING="$(run_alembic current 2>&1)"
        if echo "${PENDING}" | grep -q "(head)"; then
            success "Database is up to date — no pending migrations."
            exit 0
        else
            error "Pending migrations detected — run ./scripts/migrate.sh to apply them."
            echo ""
            run_alembic history -r "current:head" || true
            exit 1
        fi
        ;;

    *)
        error "Unknown action: ${ACTION}"
        exit 1
        ;;
esac
