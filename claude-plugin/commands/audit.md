---
description: Count stale edits in your own Claude Code and Codex sessions (counts only, nothing is uploaded)
---
Run `bash "${CLAUDE_PLUGIN_ROOT}/scripts/audit.sh"` with the Bash tool. For each session folder it prints a
counts-only summary line, then the project folders and sessions with the most stale edits. Show the output
unchanged. Then say in two sentences: the share of anchored edits that were stale and how many came through
the shell, and how concentrated they are (how many files have none, how much the top 10 files hold). Do not
open or quote any session file.
