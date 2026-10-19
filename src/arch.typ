#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#let box2(t, s) = align(center, { text(weight: "semibold", size: 10pt, t); if s != none { linebreak(); text(size: 8.5pt, fill: ink2, s) } })
#let fix(body) = block(fill: goodbg, stroke: 0.8pt + good.lighten(40%), radius: 4pt, inset: (x: 6pt, y: 4pt),
  text(size: 8.5pt, fill: good.darken(15%), body))
#let hosts = ("Claude Code", "Codex", "Gemini CLI", "Cursor", "Copilot", "Cline", "OpenCode", "MCP clients")
#card(block(width: 820pt)[
  #title("Where the fixes land", sub: [One engine, many hosts. Green tags are what \#4 changes at each stage.])
  #v(4pt)
  #diagram(
    spacing: (34pt, 20pt),
    node-stroke: 0.8pt + rule, node-fill: white, node-corner-radius: 5pt, node-inset: 7pt,
    edge-stroke: 0.9pt + muted,
    // hosts
    ..hosts.enumerate().map(((i, h)) => node((0, i * 0.62), text(size: 9pt, h), name: label("h" + str(i)), width: 78pt)),
    node((1.15, 2.17), box2("Adapters", [`claude_code` · `hooks` \ `mcp_server` · `protocol`]), name: <ad>, width: 128pt),
    ..range(8).map(i => edge(label("h" + str(i)), <ad>, "-|>")),
    node((2.35, 0.9), box2("Guard", [before a shell command: \ `sed -i '88…'` on lines not \ seen as they are now?]), name: <guard>, width: 150pt),
    node((2.35, 3.45), box2("Write-through", [after any tool call: \ scan → changed files → diff, \ line-shift notes, uses left behind]), name: <wt>, width: 150pt),
    edge(<ad>, <guard>, "-|>", text(size: 8pt, fill: muted)[PreToolUse], label-side: left),
    edge(<ad>, <wt>, "-|>", text(size: 8pt, fill: muted)[PostToolUse \ (+ Failure)], label-side: left),
    node((3.75, 0.9), box2("Refuse once", [and show the lines \ as they are now]), name: <deny>, width: 110pt, fill: badbg, stroke: 0.8pt + bad.lighten(50%)),
    node((3.75, 3.45), box2("Appended to the result", [append-only, so the \ prompt cache is kept]), name: <ctx>, width: 110pt, fill: goodbg, stroke: 0.8pt + good.lighten(50%)),
    edge(<guard>, <deny>, "-|>"), edge(<wt>, <ctx>, "-|>"),
    node((4.85, 2.17), box2("Model", none), name: <model>, width: 60pt),
    edge(<deny>, <model>, "-|>"), edge(<ctx>, <model>, "-|>"),
    // fix tags
    node((1.15, 4.35), fix[fail open when missing · \ failure events · own patches \ not echoed back · partial-read \ ranges on every host], stroke: none, fill: none),
    node((2.35, -0.25), fix[shell cwd & `cd` · absolute paths · \ ranges per command · `nl -ba`], stroke: none, fill: none),
    node((2.35, 4.6), fix[91 s → 0.2 s diffs · capped scan · \ `\n`-only lines · CRLF · Windows paths], stroke: none, fill: none),
  )
])
