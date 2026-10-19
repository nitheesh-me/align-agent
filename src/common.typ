// shared look: a light card that reads on both GitHub themes
#let surface = rgb("#fcfcfb")
#let ink = rgb("#0b0b0b")
#let ink2 = rgb("#52514e")
#let muted = rgb("#8a8984")
#let rule = rgb("#e4e3df")
#let blue = rgb("#2a78d6")
#let orange = rgb("#eb6834")
#let good = rgb("#1a7f37")
#let bad = rgb("#cf222e")
#let goodbg = rgb("#dafbe1")
#let badbg = rgb("#ffebe9")
#let codebg = rgb("#f3f2ee")
#set page(width: auto, height: auto, margin: 0pt, fill: none)
#set text(font: "Inter", size: 11pt, fill: ink)
#show raw: set text(font: "DejaVu Sans Mono", size: 9.5pt)
#let card(body) = block(fill: surface, stroke: 1pt + rule, radius: 10pt, inset: 18pt, body)
#let title(t, sub: none) = {
  text(size: 15pt, weight: "bold", t)
  if sub != none { linebreak(); text(size: 10pt, fill: ink2, sub) }
  v(4pt)
}
