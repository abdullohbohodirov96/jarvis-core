#!/usr/bin/env bash
# =============================================================================
# JARVIS — Initial setup script
# =============================================================================
#
# Idempotent first-time setup for the JARVIS AI assistant platform.
#
# What it does:
#   1. Detects the OS and installs Docker + Docker Compose if missing
#   2. Creates .env from .env.example (if .env doesn't exist)
#   3. Generates a cryptographically secure SECRET_KEY
#   4. Creates required local directories (data/sessions, etc.)
#   5. Builds Docker images
#   6. Prints a checklist of what to do next
#
# Supported OS:
#   Ubuntu / Debian, Fedora / RHEL / CentOS / Rocky, macOS (via Homebrew)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/.../setup.sh | bash
#   -- or --
#   chmod +x scripts/setup.sh && ./scripts/setup.sh
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
step()    { echo -e "${BOLD}[$1]${NC} $2"; }

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------
detect_os() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        OS_TYPE="${ID:-unknown}"
        OS_LIKE="${ID_LIKE:-}"
    elif [[ "$(uname)" == "Darwin" ]]; then
        OS_TYPE="macos"
    else
        OS_TYPE="unknown"
    fi
    echo "${OS_TYPE}"
}

OS="$(detect_os)"
info "Detected OS: ${OS}"

# ---------------------------------------------------------------------------
# Step 1: Docker installation
# ---------------------------------------------------------------------------
header "Step 1/5 — Docker"

install_docker_debian() {
    info "Installing Docker on Debian/Ubuntu..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg lsb-release
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    # Add current user to docker group (avoids sudo for docker commands)
    if [[ -n "${SUDO_USER:-}" ]]; then
        usermod -aG docker "${SUDO_USER}"
    fi
}

install_docker_fedora() {
    info "Installing Docker on Fedora/RHEL..."
    dnf -y install dnf-plugins-core
    dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
    dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    if [[ -n "${SUDO_USER:-}" ]]; then
        usermod -aG docker "${SUDO_USER}"
    fi
}

install_docker_macos() {
    info "Installing Docker Desktop on macOS..."
    if ! command -v brew &>/dev/null; then
        error "Homebrew is required on macOS. Install it first:"
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    brew install --cask docker
    info "Please start Docker Desktop from Applications and then re-run this script."
    exit 0
}

if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
    DOCKER_VERSION="$(docker --version)"
    COMPOSE_VERSION="$(docker compose version)"
    success "Docker already installed: ${DOCKER_VERSION}"
    success "Docker Compose: ${COMPOSE_VERSION}"
else
    warning "Docker not found — attempting installation..."

    # Check for root / sudo
    if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        error "Docker installation requires sudo privileges."
        echo "  Run: sudo ./scripts/setup.sh"
        exit 1
    fi

    case "${OS}" in
        ubuntu|debian|linuxmint|pop)      install_docker_debian ;;
        fedora|rhel|centos|rocky|almalinux) install_docker_fedora ;;
        macos)                             install_docker_macos ;;
        *)
            error "Unsupported OS: ${OS}"
            echo "  Please install Docker manually: https://docs.docker.com/get-docker/"
            exit 1
            ;;
    esac

    success "Docker installed successfully"
fi

# ---------------------------------------------------------------------------
# Step 2: Create .env
# ---------------------------------------------------------------------------
header "Step 2/5 — Environment configuration"

if [[ -f ".env" ]]; then
    success ".env already exists — skipping"
else
    if [[ ! -f ".env.example" ]]; then
        error ".env.example not found in ${PROJECT_ROOT}"
        exit 1
    fi
    cp .env.example .env
    success ".env created from .env.example"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate SECRET_KEY
# ---------------------------------------------------------------------------
header "Step 3/5 — Generating SECRET_KEY"

if grep -q "your-super-secret-key-change-this" .env 2>/dev/null; then
    # Generate 32 bytes of random hex (64 hex characters)
    if command -v openssl &>/dev/null; then
        NEW_KEY="$(openssl rand -hex 32)"
    elif command -v python3 &>/dev/null; then
        NEW_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    else
        error "Cannot generate SECRET_KEY — openssl and python3 both unavailable."
        error "Set SECRET_KEY manually in .env to a random 64-character hex string."
        exit 1
    fi

    # Replace the placeholder in .env
    # Use different sed syntax for macOS vs Linux
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "s|SECRET_KEY=.*|SECRET_KEY=${NEW_KEY}|" .env
    else
        sed -i "s|SECRET_KEY=.*|SECRET_KEY=${NEW_KEY}|" .env
    fi

    success "SECRET_KEY generated and written to .env (${#NEW_KEY} characters)"
else
    success "SECRET_KEY is already set"
fi

# ---------------------------------------------------------------------------
# Step 4: Create required directories
# ---------------------------------------------------------------------------
header "Step 4/5 — Creating directories"

DIRS=(
    "data/sessions"
    "data/whisper_cache"
    "docker/nginx/ssl"
    "docker/nginx/conf.d"
    "logs"
    "static"
    "alembic/versions"
)

for dir in "${DIRS[@]}"; do
    if [[ ! -d "${dir}" ]]; then
        mkdir -p "${dir}"
        info "Created: ${dir}/"
    else
        info "Exists:  ${dir}/"
    fi
done

# Create a .gitkeep in data/sessions so the directory is tracked
touch data/sessions/.gitkeep 2>/dev/null || true

success "Directories ready"

# ---------------------------------------------------------------------------
# Step 5: Build Docker images
# ---------------------------------------------------------------------------
header "Step 5/5 — Building Docker images"

info "This will take a few minutes on the first build (downloading base images + installing dependencies)..."
docker compose build --parallel
success "Docker images built successfully"

# ---------------------------------------------------------------------------
# Next steps checklist
# ---------------------------------------------------------------------------
header "Setup complete!"

echo -e "${GREEN}${BOLD}JARVIS setup finished.${NC}"
echo ""
echo -e "${BOLD}Before starting, fill in these required values in ${CYAN}.env${NC}${BOLD}:${NC}"
echo ""
echo -e "  ${YELLOW}OPENAI_API_KEY${NC}        OpenAI API key (https://platform.openai.com/api-keys)"
echo -e "  ${YELLOW}TELEGRAM_API_ID${NC}       Telegram API ID (https://my.telegram.org/apps)"
echo -e "  ${YELLOW}TELEGRAM_API_HASH${NC}     Telegram API hash"
echo -e "  ${YELLOW}TELEGRAM_PHONE${NC}        Phone number in E.164 format (e.g. +12025551234)"
echo ""
echo -e "${BOLD}Optional (for premium TTS):${NC}"
echo -e "  ${YELLOW}ELEVENLABS_API_KEY${NC}    ElevenLabs API key"
echo ""
echo -e "${BOLD}To start JARVIS:${NC}"
echo -e "  ${CYAN}./scripts/start.sh${NC}"
echo ""
echo -e "${BOLD}For production deployment:${NC}"
echo -e "  ${CYAN}./scripts/start.sh --prod${NC}"
echo ""
