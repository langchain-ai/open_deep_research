#!/usr/bin/env bash
set -euo pipefail

log(){ echo "[bootstrap] $*" >&2; }
die(){ echo "[bootstrap] ERROR: $*" >&2; exit 1; }

ROOT="$(pwd)"
[ -d "$ROOT/src" ] || die "Run from repo root (no ./src)."
[ -d "$ROOT/src/open_deep_research" ] || die "No ./src/open_deep_research found (wrong repo)."

# Ensure results dir exists
mkdir -p "$ROOT/results/llamator"

# Ensure .gitignore includes mandatory ignores (idempotent)
touch "$ROOT/.gitignore"
append_if_missing() { local line="$1"; local file="$2"; grep -qxF "$line" "$file" 2>/dev/null || echo "$line" >> "$file"; }
append_if_missing ".env" "$ROOT/.gitignore"
append_if_missing ".env.*" "$ROOT/.gitignore"
append_if_missing "results/" "$ROOT/.gitignore"
append_if_missing ".langsmith/" "$ROOT/.gitignore"
append_if_missing "*.log" "$ROOT/.gitignore"

# Patch src/open_deep_research/__init__.py to bootstrap env + logging early
INIT_PY="$ROOT/src/open_deep_research/__init__.py"
touch "$INIT_PY"
MARK="bootstrap_runtime_env"
if ! grep -q "$MARK" "$INIT_PY"; then
  cat >> "$INIT_PY" <<'EOF'

# --- security bootstrap (OpenRouter env + safe logging) ---
try:
    from .security.runtime_env import bootstrap_runtime_env
    bootstrap_runtime_env()
except Exception:
    # Fail-fast is expected if LLM_MODE=api but LLM_API_KEY is empty.
    raise
EOF
  log "Patched src/open_deep_research/__init__.py to call bootstrap_runtime_env()"
else
  log "src/open_deep_research/__init__.py already contains bootstrap hook; skipping"
fi

# Ensure executable bits for scripts
chmod +x "$ROOT/bootstrap.sh" || true
chmod +x "$ROOT/scripts/"*.sh 2>/dev/null || true

# Insert docs snippets into README.md / SECURITY.md using markers (non-destructive)
insert_snippet(){
  local target="$1"
  local snippet="$2"
  local start_marker="$3"
  local end_marker="$4"

  # If target doesn't exist, create it from snippet
  if [ ! -f "$target" ]; then
    cp "$snippet" "$target"
    log "Created $target from snippet"
    return
  fi

  # If markers exist, replace section
  if grep -q "$start_marker" "$target"; then
    # delete existing marked block
    python3 - <<PY
from pathlib import Path
t = Path("$target")
s = t.read_text(encoding="utf-8")
start = "$start_marker"
end = "$end_marker"
a = s.split(start)
if len(a) < 2:
    raise SystemExit(0)
before = a[0]
rest = start.join(a[1:])
b = rest.split(end)
after = end.join(b[1:]) if len(b) > 1 else ""
snippet = Path("$snippet").read_text(encoding="utf-8")
t.write_text(before + snippet + after, encoding="utf-8")
PY
    log "Updated snippet block in $target"
  else
    # append
    printf "\n\n" >> "$target"
    cat "$snippet" >> "$target"
    log "Appended snippet to $target"
  fi
}

insert_snippet "$ROOT/README.md" "$ROOT/docs/README_SNIPPET.md" "ODR_OPENROUTER_SNIPPET_START" "ODR_OPENROUTER_SNIPPET_END"
insert_snippet "$ROOT/SECURITY.md" "$ROOT/docs/SECURITY_SNIPPET.md" "ODR_SECURITY_SNIPPET_START" "ODR_SECURITY_SNIPPET_END"

log "Bootstrap complete."
log "Next: cp .env.example .env  (fill LLM_API_KEY + optionally LLAMATOR_ATTACK_API_KEY; keep LIVE_API=0 default)."
