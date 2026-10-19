// log-scale horizontal bars, 0.1 s .. 300 s
#let W = 420pt
#let lo = calc.log(0.1)
#let hi = calc.log(300)
#let x(s) = W * (calc.log(s) - lo) / (hi - lo)
#let bar(s, c, label, over: false) = box(width: W, height: 13pt, {
  if s != none {
    place(left + horizon, box(width: x(s), height: 11pt, fill: c, radius: (right: 4pt)))
    place(left + horizon, dx: x(s) + 5pt, text(size: 9pt, fill: ink2, label))
  } else { place(left + horizon, dx: 2pt, text(size: 9pt, fill: ink2, label)) }
})
#let row(name, sub, before, blabel, after, alabel) = (
  align(right + horizon, block(width: 165pt, { text(weight: "semibold", size: 10pt, name); linebreak(); text(size: 8.5pt, fill: muted, sub) })),
  { bar(before, orange, blabel); v(3pt); bar(after, blue, alabel) },
)
#let axis = box(width: W, height: 16pt, {
  for (s, l) in ((0.1, "0.1 s"), (1, "1 s"), (10, "10 s"), (100, "100 s")) {
    place(left + top, dx: x(s) - 10pt, text(size: 8.5pt, fill: muted, l))
  }
})
#card(block(width: 660pt)[
  #title("Time for one hook call", sub: [Claude Code kills a hook after 30 s, before it saves state, so a slow diff used to repeat on every later tool call.])
  #v(2pt)
  #box({
    // gridlines + timeout line behind the bars
    let plotx = 165pt + 12pt
    place(left + top, dx: plotx, dy: 0pt, box(width: W, height: 172pt, {
      for s in (0.1, 1, 10, 100) { place(left, dx: x(s), line(start: (0pt, 0pt), end: (0pt, 172pt), stroke: 0.6pt + rule)) }
      place(left, dx: x(30), line(start: (0pt, -4pt), end: (0pt, 172pt), stroke: (paint: bad, thickness: 1.2pt, dash: "dashed")))
      place(left, dx: x(30) + 4pt, dy: -6pt, text(size: 8.5pt, fill: bad, weight: "semibold")[hook timeout 30 s])
    }))
    grid(columns: (165pt, W), column-gutter: 12pt, row-gutter: 16pt,
      [], [],
      ..row("Diff a big lockfile", "package-lock.json, 21k lines", 91, [91 s], 0.2, [0.2 s]),
      ..row("Pathological diff", "886 KB, repeated lines", 180, [> 180 s (still running)], 0.2, [0.2 s]),
      ..row("Scan the tree, no git", [`$HOME`, 1.9M files], 33.3, [33 s on every call], none, [skipped (non-git walks stop at 20k files / 5 s)]),
      [], axis,
    )
  })
  #v(6pt)
  #text(size: 9pt)[#box(width: 9pt, height: 9pt, fill: orange, radius: 2pt) #h(2pt) 0.4.9 #h(14pt) #box(width: 9pt, height: 9pt, fill: blue, radius: 2pt) #h(2pt) this PR #h(14pt) #text(fill: muted)[log scale · measured on Linux, one run each]]
])
