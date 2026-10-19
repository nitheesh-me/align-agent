#!/usr/bin/env bash
# Count stale edits in this machine's own Claude Code and Codex session logs.
# Prints one summary line per folder (counts only: no paths, prompts or code), then which of your own
# project folders and sessions hold the most stale edits (names stay on this machine). Nothing is uploaded.
#
# It needs the `align-agent` command. If that is not installed, it installs the version matching this plugin
# once, from PyPI, into its own virtual environment (~/.cache/align-agent/venv), and reinstalls only when the
# plugin's version changes. uvx and pipx run that same version. Errors are shown, not hidden.
# Works with Python 3.10 or later as python3, python or `py -3` (Windows, in Git Bash).

CACHE="${ALIGN_CACHE:-$HOME/.cache/align-agent}"
VENV="$CACHE/venv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# the package version this plugin ships with (plugin.json); unpinned only if it cannot be read
VER="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$HERE/../.claude-plugin/plugin.json" 2>/dev/null | head -n 1)"
SPEC="align-agent${VER:+==$VER}"

find_python() {
  for c in python3.13 python3.12 python3.11 python3.10 python3 python "py -3"; do
    # shellcheck disable=SC2086  # "py -3" is a command and its argument
    if $c -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

venv_python() {
  for b in "$VENV/bin/python" "$VENV/Scripts/python.exe" "$VENV/Scripts/python"; do
    [ -x "$b" ] && { echo "$b"; return 0; }
  done
  return 1
}

venv_install() {  # install $SPEC into the venv unless that version is already there
  vpy="$(venv_python)" || return 1
  if [ -n "$VER" ] && [ "$("$vpy" -c 'import importlib.metadata as m; print(m.version("align-agent"))' 2>/dev/null)" = "$VER" ]; then
    return 0
  fi
  "$vpy" -m pip install --quiet --disable-pip-version-check "$SPEC" >&2
}

# AA: the align-agent command as an array, so paths with spaces stay one word
align_command() {
  if command -v align-agent >/dev/null 2>&1; then AA=(align-agent); return 0; fi
  if venv_python >/dev/null; then
    venv_install || echo "align-agent: could not install $SPEC; using the one already in $VENV" >&2
  elif command -v uvx >/dev/null 2>&1; then AA=(uvx --quiet "align-agent${VER:+@$VER}"); return 0
  elif command -v pipx >/dev/null 2>&1; then AA=(pipx run --spec "$SPEC" align-agent); return 0
  else
    py="$(find_python)" || { echo "align-agent: no Python 3.10 or later found (tried python3, python, py -3). Install Python from https://www.python.org/downloads/ and run this again." >&2; return 1; }
    echo "align-agent: installing $SPEC from PyPI into $VENV (first run only)..." >&2
    # shellcheck disable=SC2086
    mkdir -p "$CACHE" && $py -m venv "$VENV" >&2 || { echo "align-agent: could not create a virtual environment in $VENV" >&2; return 1; }
    venv_install
  fi
  for b in "$VENV/bin/align-agent" "$VENV/Scripts/align-agent.exe" "$VENV/Scripts/align-agent"; do
    [ -x "$b" ] && { AA=("$b"); return 0; }
  done
  echo "align-agent: the install failed; see the messages above" >&2
  return 1
}

align_command || exit 1
mkdir -p "$CACHE" || exit 1  # last_error.txt goes here, also when align-agent came from PATH, uvx or pipx
found=0
for d in "$HOME/.claude/projects" "$HOME/.codex/sessions"; do
  [ -d "$d" ] || continue
  found=1
  printf '%s: ' "$(basename "$(dirname "$d")")"
  "${AA[@]}" analyze "$d" --quiet --top 5 -j 4 2>"$CACHE/last_error.txt" | sed '/^$/d'
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "(failed: $(tail -n 3 "$CACHE/last_error.txt" 2>/dev/null | tr '\n' ' '))"; }
done
[ "$found" = 1 ] || echo "no Claude Code or Codex session folders found (looked in ~/.claude/projects and ~/.codex/sessions)"
