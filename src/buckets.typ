#let dot(c, sym) = box(width: 18pt, height: 18pt, radius: 9pt, fill: c, align(center + horizon, text(fill: white, size: 8pt, weight: "bold", sym)))
#let tl(..items) = box(items.pos().join(box(width: 14pt, align(horizon, line(length: 14pt, stroke: 1pt + muted)))))
#let V = dot(rgb("#6e7781"), "V")
#let OWN = dot(blue, "W")
#let OTHER = dot(orange, "W")
#let E = dot(ink, "E")
#let EX = dot(bad, "E")
#let chip(c, bg, body) = box(fill: bg, stroke: 0.8pt + c.lighten(35%), radius: 10pt, inset: (x: 8pt, y: 4pt), text(size: 9pt, fill: c.darken(10%), weight: "semibold", body))
#let gray = rgb("#6e7781")
#let row(story, line, old, new) = (align(horizon, line), align(horizon, text(size: 9.5pt, story)), align(horizon, old), align(horizon, new))
#card(block(width: 800pt)[
  #title("What \"stale\" counts", sub: [0.4.9 put very different events in one number and couldn't see the loudest one. #h(4pt) V = view · W = write (blue: the agent's own, orange: someone else's) · E = anchored edit · R = whole-file rewrite])
  #v(6pt)
  #grid(columns: (128pt, 1fr, 120pt, 200pt), column-gutter: 14pt, row-gutter: 13pt,
    [], [], text(size: 9pt, fill: muted, weight: "bold", "0.4.9"), text(size: 9pt, fill: muted, weight: "bold", "THIS PR"),
    grid.hline(stroke: 0.6pt + rule),
    ..row([The agent runs its own `sed`/python edit, then edits the file again without re-reading it.], tl(V, OWN, E),
      chip(orange, rgb("#fff1e8"), "stale"), chip(blue, rgb("#ddf4ff"), "stale · after its own unread write")),
    ..row([A subagent, formatter or script changes the file; the agent edits from its old view. #text(fill: ink2)[This is the case the guard exists for.]], tl(V, OTHER, E),
      chip(orange, rgb("#fff1e8"), "stale"), chip(bad, badbg, "stale · after a write it didn't make")),
    ..row([Claude Code refuses the edit with "File has been modified since read". (A rejected "String to replace not found" is now scored as a failed edit too.)], tl(V, OTHER, EX),
      chip(gray, rgb("#f0f0ee"), "not counted"), chip(good, goodbg, "scored · CAUGHT-BY-HARNESS")),
    ..row([A python one-liner prints the file (`print(open(p).read())`), then the agent edits it.], tl(V, OWN, V, E),
      chip(orange, rgb("#fff1e8"), "stale"), chip(good, goodbg, "not stale · printing is a view")),
    ..row([A `'x' in s` check, then `open(p,'w').write(…)`: a whole-file rewrite, not an anchored edit.], tl(V, OWN, dot(ink, "R")),
      chip(orange, rgb("#fff1e8"), "anchored edit"), chip(good, goodbg, "rewrite · not an opportunity")),
  )
  #v(8pt)
  #text(size: 9pt, fill: ink2)[The interval changes too: edits cluster within sessions, so `ci95` is now a bootstrap over sessions (the binomial interval is kept as `ci95_binomial`).]
])
