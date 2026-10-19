#let ln(n, code, hl: none, gone: false) = {
  let bg = if hl == "bad" { badbg } else if hl == "good" { goodbg } else if hl == "new" { rgb("#ddf4ff") } else { none }
  box(width: 100%, fill: bg, inset: (x: 4pt, y: 2.2pt), radius: 2pt,
    grid(columns: (22pt, 1fr), text(fill: muted, raw(str(n))),
      if gone { let n = code.len() - code.trim(at: start).len(); raw(" " * n); text(fill: bad, strike(raw(code.trim(at: start)))) } else { raw(code) }))
}
#let pane(head, step, body) = block(width: 205pt, height: 176pt, fill: white, stroke: 1pt + rule, radius: 6pt, inset: 9pt, {
  text(size: 9pt, fill: muted, weight: "bold", upper(step)); linebreak()
  text(weight: "semibold", head); v(3pt)
  block(fill: codebg, radius: 4pt, inset: 4pt, width: 100%, body)
})
#let arrow = align(horizon, text(size: 18pt, fill: muted, sym.arrow.r))

#card(block(width: 690pt)[
  #title("How a stale line number bites", sub: [The agent edits by line number from a view that is one write out of date. Nothing errors: the file is simply wrong.])
  #grid(columns: 5, column-gutter: 8pt,
    pane("The model reads the file", "step 1 · view", {
      ln(86, "def total(xs):"); ln(87, "    s = 0")
      ln(88, "    for x in xs:", hl: "good"); ln(89, "        s += x")
      v(2pt); text(size: 8.5pt, fill: ink2)[It plans to change line 88.]
    }),
    arrow,
    pane("A script inserts a line above", "step 2 · unseen write", {
      ln(86, "import math", hl: "new"); ln(87, "def total(xs):")
      ln(88, "    s = 0"); ln(89, "    for x in xs:")
      v(2pt); text(size: 8.5pt, fill: ink2)[Everything moves down one. The model never sees this.]
    }),
    arrow,
    pane([`sed -i '88s/.*/…/' stats.py`], "step 3 · edit by number", {
      ln(87, "def total(xs):"); ln(88, "    s = 0", gone: true)
      ln(88, "    for x in sorted(xs):", hl: "bad"); ln(89, "    for x in xs:")
      v(2pt); text(size: 8.5pt, fill: bad)[Replaced `s = 0` instead of the loop.]
    }),
  )
  #v(10pt)
  #block(width: 100%, fill: goodbg, radius: 6pt, inset: 10pt, stroke: 1pt + good.lighten(40%))[
    #text(weight: "bold", fill: good)[With align-write-through:] the step-2 write is appended to the next tool result as a diff with its new line numbers, and if the model still sends `sed -i '88s…'`, the guard refuses it once and shows lines 83–93 as they are now:
    #v(4pt)
    #block(fill: white, radius: 4pt, inset: 7pt, width: 100%, text(size: 9pt, raw("[STALE LINE NUMBERS] 'stats.py' changed since you last saw lines 88-88 numbered, so they may not be the lines you mean. Here they are as the file stands now (lines 83-93): …\nReissue the command with these line numbers if this is what you meant to change.")))
  ]
])
