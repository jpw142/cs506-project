SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CONTEXT_ROOT="$PROJECT_ROOT"

export VENV_PYTHON="$CONTEXT_ROOT/.venv/bin/python"
export INTR="$CONTEXT_ROOT/intermediary"
export PYS="$CONTEXT_ROOT/pys"
