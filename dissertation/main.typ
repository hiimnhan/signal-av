#import "@local/lu-msc-dissertation:0.1.0": appendix, dissertation, mainmatter, preamble
#import "@preview/citesugar:0.1.0": citesugar

// Strip leading given-name initials from a cite's rendering (e.g. "M. Spence" -> "Spence").
#let cite-surname(key, supplement: none, style: auto, form: "author") = {
  show regex("^([A-Z]\.\s*)+"): none
  cite(key, form: form, supplement: supplement, style: style)
}

// Same, but genitive: "D. Lewis'" -> "Lewis'".
#let cite-surname-genitive(key, supplement: none, style: auto) = {
  show regex("^([A-Z]\.\s*)+"): none
  show regex(".\s[\[\(]"): it => {
    show regex("[^\s\[\(]"): it => it + "'"
    show regex("[^s]'"): it => it + "s"
    it
  }
  cite(key, form: "prose", supplement: supplement, style: style)
}

// Typst's built-in "year" form silently drops any supplement. Render the year,
// then append the supplement as plain text ourselves instead of relying on it.
#let cite-year-suppl(key, supplement: none, style: auto) = {
  cite(key, form: "year", style: style)
  if supplement != none {
    [, ] + supplement
  }
}

#show cite: citesugar.with(
  forms: (
    // Override prose form itself: "M. Spence (1973)" -> "Spence (1973)".
    p: (key, suppl) => cite-surname(key, supplement: suppl, form: "prose"),
    // Override prose genitive: "D. Lewis'" -> "Lewis'".
    ps: (key, suppl) => cite-surname-genitive(key, supplement: suppl),
    // Author-only, no year: "Spence".
    a: (key, suppl) => cite-surname(key, supplement: suppl, form: "author"),
    // Year form, but keep the page/supplement instead of dropping it.
    y: (key, suppl) => cite-year-suppl(key, supplement: suppl),
  ),
)

#show: dissertation.with(
  title: "Emergent Behavioural Signalling in Cooperative Multi-Agent Highway Driving",
  author: "Nhan Nguyen",
  degrees: "",
  degree: "Master of Science",
  field: "Artificial Intelligence",
  supervisor: "Dr. Muhammad Bilal",
  date: datetime.today(),
)

// =================== PREAMBLE (roman numerals) ===================
#preamble[
  #include "declaration.typ"
  #pagebreak()

  #include "abstract.typ"
  #pagebreak()

  #include "acknowledgements.typ"
  #pagebreak()

  #outline(title: [Contents], depth: 4, indent: auto)
  #pagebreak()

  #outline(title: [List of Figures], target: figure.where(kind: image))
  #pagebreak()

  #outline(title: [List of Tables], target: figure.where(kind: table))
]

#mainmatter[
  = Introduction
  #include "chapters/introduction.typ"

  = Related Work
  #include "chapters/related_work.typ"

  = Methodology
  #include "chapters/methodology.typ"

  = Results <r-a>
  #include "chapters/result_analysis.typ"

  = Discussion
  #include "chapters/discussion.typ"

  = Conclusion
  #include "chapters/conclusion.typ"

  // ---------- References ----------
  #pagebreak()
  #bibliography("ref.bib", title: "References", style: "ieee-collapsed.csl")

  // ---------- Appendix ----------
  #appendix[
    = Reproducibility <app-reproducibility>
    #include "appendix/reproducibility.typ"

    = Statistical methodology <app-statistics>
    #include "appendix/statistics.typ"
  ]

]
