#!/bin/sh
# Claude Code hook launcher: run hook.py with the first Python 3.8+ found as python3, python or
# `py -3` (Windows, Git Bash). With none, exit 0 silently, so a missing interpreter never shows a
# hook error on every tool call. Probes skip the Windows Store `python3` alias (it exits 9009) and,
# on macOS, the /usr/bin/python3 stub that asks to install the Command Line Tools.
# The interpreter found is cached in $CLAUDE_PLUGIN_DATA/python, so later calls start Python once.
here=$(dirname "$0")
cache="${CLAUDE_PLUGIN_DATA:-$HOME/.cache/align-agent}/python"
# line 1: the interpreter's path (it may hold spaces); line 2: its arguments (`-3` for py)
if [ -r "$cache" ] && { IFS= read -r exe && IFS= read -r args; } <"$cache"; then
  # the cached interpreter may have been uninstalled: probe again instead of failing every call
  if [ -n "$exe" ] && [ -x "$exe" ]; then
    exec "$exe" $args "$here/hook.py"
  fi
fi
for py in python3 python "py -3"; do
  set -- $py
  command -v "$1" >/dev/null 2>&1 || continue
  if [ "$(uname -s 2>/dev/null)" = Darwin ] && [ "$(command -v "$1")" = /usr/bin/python3 ] \
     && ! xcode-select -p >/dev/null 2>&1; then
    continue
  fi
  if "$@" -c 'import sys; sys.exit(sys.version_info < (3, 8))' </dev/null >/dev/null 2>&1; then
    exe=$(command -v "$1")
    shift
    mkdir -p "$(dirname "$cache")" 2>/dev/null && printf '%s\n%s\n' "$exe" "$*" >"$cache" 2>/dev/null
    exec "$exe" "$@" "$here/hook.py"
  fi
done
cat >/dev/null
exit 0
