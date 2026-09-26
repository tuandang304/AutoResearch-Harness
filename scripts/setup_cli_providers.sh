#!/usr/bin/env bash
# Check that the Claude Code / Codex / Antigravity CLIs are installed and logged in,
# and configure Antigravity so it can read images from AutoResearch-Harness's scratch dir.
#
#   bash scripts/setup_cli_providers.sh           # check all providers
#   bash scripts/setup_cli_providers.sh --no-test # skip the live "pong" calls
set -uo pipefail

TEST=1
[[ "${1:-}" == "--no-test" ]] && TEST=0
ok()   { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m✘\033[0m %s\n' "$*"; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "Claude Code (claude-code/<model>)"
if command -v claude >/dev/null; then
  ok "claude $(claude --version 2>/dev/null | head -1)"
  if (( TEST )); then
    out=$(cd "$TMP" && echo "Reply with exactly: pong" | timeout 120 claude -p --tools "" --safe-mode --no-session-persistence 2>&1)
    [[ "$out" == *pong* ]] && ok "logged in and responding" || bad "not responding (run 'claude' once and log in): ${out:0:200}"
  fi
else
  warn "not installed: npm install -g @anthropic-ai/claude-code  (then run 'claude' to log in)"
fi

echo "Codex (codex/<model>)"
if command -v codex >/dev/null; then
  ok "$(codex --version 2>/dev/null | head -1)"
  if (( TEST )); then
    out=$(cd "$TMP" && echo "Reply with exactly: pong" | timeout 180 codex exec --skip-git-repo-check --ephemeral --sandbox read-only -o "$TMP/codex.txt" - 2>&1 >/dev/null; cat "$TMP/codex.txt" 2>/dev/null)
    [[ "$out" == *pong* ]] && ok "logged in and responding" || bad "not responding (run 'codex login'): ${out:0:200}"
  fi
else
  warn "not installed: npm install -g @openai/codex  (then 'codex login')"
fi

echo "Antigravity (antigravity/<model>)"
if command -v agy >/dev/null; then
  ok "agy $(agy --version 2>/dev/null | head -1)"
  # agy is usually a snap: its home (and settings) live under ~/snap/antigravity-cli/common
  SNAP_HOME="$HOME/snap/antigravity-cli/common"
  if [[ -d "$SNAP_HOME" ]]; then
    AGY_HOME="$SNAP_HOME"; SCRATCH="${AI_SCIENTIST_CLI_TMP:-$SNAP_HOME/ai_scientist_tmp}"
  else
    AGY_HOME="$HOME"; SCRATCH="${AI_SCIENTIST_CLI_TMP:-${TMPDIR:-/tmp}/ai_scientist_cli}"
  fi
  SETTINGS="$AGY_HOME/.gemini/antigravity-cli/settings.json"
  mkdir -p "$SCRATCH" "$(dirname "$SETTINGS")"
  python3 - "$SETTINGS" "read_file($SCRATCH)" <<'EOF'
import json, os, sys
path, rule = sys.argv[1], sys.argv[2]
cfg = json.load(open(path)) if os.path.exists(path) and os.path.getsize(path) else {}
allow = cfg.setdefault("permissions", {}).setdefault("allow", [])
if rule not in allow:
    allow.append(rule)
    json.dump(cfg, open(path, "w"), indent=2)
    print(f"  \033[32m✔\033[0m added {rule} to {path}")
else:
    print(f"  \033[32m✔\033[0m {path} already allows {rule}")
EOF
  if (( TEST )); then
    for _try in 1 2; do  # agy occasionally hangs at startup; retry once
      out=$(cd "$SCRATCH" && printf '{"event":"user","message":{"content":"Reply with exactly: pong"}}\n' | timeout 90 agy -p= --input-format stream-json --output-format stream-json 2>&1)
      [[ "$out" == *pong* ]] && break
    done
    [[ "$out" == *pong* ]] && ok "logged in and responding" || bad "not responding (run 'agy' once and sign in): ${out:0:200}"
  fi
  echo "    models: agy models"
else
  warn "not installed: see https://antigravity.google/docs/cli (then run 'agy' to sign in)"
fi

echo "Write-up tools"
for t in pdflatex bibtex pdftotext chktex; do
  if command -v $t >/dev/null; then ok "$t"; else
    case $t in
      chktex) warn "chktex missing (LaTeX lint used during write-up): sudo apt install chktex" ;;
      pdftotext) bad "pdftotext missing: sudo apt install poppler-utils" ;;
      *) bad "$t missing: sudo apt install texlive-full  (or texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra)" ;;
    esac
  fi
done
