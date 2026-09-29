#!/usr/bin/env bash
# SessionStart: refresh the index in the background (incremental, only changed files).
# No config -> do nothing. Never blocks or fails the session.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CLAUDE_PROJECT_DIR:-$PWD}" 2>/dev/null || true
has_cfg=""
[ -n "${DEEPRECALL_CONFIG:-}" ] && has_cfg=1
d="$PWD"; while [ -z "$has_cfg" ] && [ "$d" != "/" ]; do [ -f "$d/.deeprecall/config.toml" ] && has_cfg=1; d="$(dirname "$d")"; done
[ -f "$HOME/.config/deeprecall/config.toml" ] && has_cfg=1
[ -z "$has_cfg" ] && exit 0
state="${XDG_STATE_HOME:-$HOME/.local/state}/deeprecall"; mkdir -p "$state"
(
  if command -v flock >/dev/null; then exec 9>"$state/index.lock"; flock -n 9 || exit 0; fi
  nohup "$ROOT/bin/deeprecall" index --quiet >>"$state/session-index.log" 2>&1
) </dev/null >/dev/null 2>&1 &
exit 0
