#let termbg = rgb("#1f2328")
#let tdot(c) = box(width: 9pt, height: 9pt, radius: 4.5pt, fill: c)
#let term(head, body) = block(width: 100%, radius: 7pt, clip: true, stroke: 1pt + rule, fill: termbg, height: 150pt, {
  set block(spacing: 0pt)
  block(width: 100%, fill: rgb("#e9e8e4"), inset: (x: 9pt, y: 6pt), {
    tdot(rgb("#ff5f57")); h(4pt); tdot(rgb("#febc2e")); h(4pt); tdot(rgb("#28c840")); h(10pt)
    text(size: 9pt, fill: ink2, weight: "semibold", head) })
  block(width: 100%, fill: termbg, inset: 11pt, text(fill: rgb("#e6edf3"), body))
})
#let p = text(fill: rgb("#7ee787"), raw("$ "))
#let err(s) = text(fill: rgb("#ff7b72"), raw(s))
#let dim(s) = text(fill: rgb("#8b949e"), raw(s))
#card(block(width: 640pt)[
  #title("/align-write-through:audit", sub: [On a machine that already has `align-agent`, `uvx` or `pipx` and no `~/.cache/align-agent` yet.])
  #grid(columns: (1fr, 1fr), column-gutter: 12pt,
    term("0.4.9", {
      p; raw("bash scripts/audit.sh"); linebreak()
      raw(".claude: "); err("audit.sh: line 50:"); linebreak()
      err("  ~/.cache/align-agent/last_error.txt:"); linebreak()
      err("  No such file or directory"); linebreak()
      err("(failed: )"); linebreak(); linebreak()
      dim("# the redirect fails, so analyze never runs")
    }),
    term("with #1", {
      p; raw("bash scripts/audit.sh"); linebreak()
      raw(".claude: N trajectories, N anchored edits"); linebreak()
      raw("  to viewed files, N stale (…%) …"); linebreak()
      raw("Where they are: …"); linebreak(); linebreak()
      dim("# mkdir -p \"$CACHE\" before the redirect")
    }),
  )
])
