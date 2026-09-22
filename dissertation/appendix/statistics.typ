All intervals and error bars are 95% confidence intervals. For a metric summarised across the twenty seeds of a condition ($rho$, Speaker Consistency, CIC, CDE, collision rate, mean return), the interval is a Student-$t$ interval over the seed-level values; each seed contributes one value, itself an average over its 200 evaluation episodes. For the training curve in @fig-rho-training, we plot the mean of $rho$ across seeds at each logged iteration with a 95% confidence interval. Paired intervention-versus-baseline tests use the Wilcoxon signed-rank test with a Holm-Bonferroni correction. Cross-seed relationships use both Pearson and Spearman correlations. The _no-signal_ versus _bimodal_ baseline comparison uses the two-sided Mann-Whitney U test rather than a paired test, since the _no-signal_ condition is an independently trained population. Exact and adjusted $p$-values for every test reported in the thesis are given below.

#figure(
  caption: [Paired intervention-versus-baseline tests on the _bimodal_ family
    ($n = 20$ seeds; baseline collision rate $38.5%$, baseline return
    $201.4$). Each row is one intervention. $p$ is the two-sided Wilcoxon signed-rank test on the 20 paired seed-level differences between that intervention and the baseline; $p_"Holm"$ is the Holm-Bonferroni correction applied across the six interventions, separately within each metric. @tab-interventions reports the same adjusted values rounded, alongside the point estimates and confidence intervals.],
  text(size: 9pt, table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: (x: 4pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header(
      table.cell(rowspan: 2, align: left + horizon)[*Intervention*],
      table.cell(colspan: 2, align: center)[*Collision rate*],
      table.cell(colspan: 2, align: center)[*Return*],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*$p$*], [*$p_"Holm"$*], [*$p$*], [*$p_"Holm"$*],
    ),
    table.hline(stroke: 0.5pt),
    [`hide`], [0.220], [0.659], [0.0400], [0.0400],
    [`permute`],
    [$2.91 times 10^(-4)$],
    [$1.17 times 10^(-3)$],
    [$3.62 times 10^(-5)$],
    [$1.45 times 10^(-4)$],
    [`randomise`],
    [$8.83 times 10^(-5)$],
    [$5.30 times 10^(-4)$],
    [$3.81 times 10^(-6)$],
    [$1.91 times 10^(-5)$],
    [`desync`],
    [$8.83 times 10^(-5)$],
    [$5.30 times 10^(-4)$],
    [$1.91 times 10^(-6)$],
    [$1.14 times 10^(-5)$],
    [`hide_kin`],
    [0.550],
    [0.866],
    [$1.05 times 10^(-4)$],
    [$2.10 times 10^(-4)$],
    [`hide_all`],
    [0.433],
    [0.866],
    [$4.77 times 10^(-5)$],
    [$1.45 times 10^(-4)$],
    table.hline(stroke: 0.9pt),
  )),
)<tab-intervention-tests>

#v(1.5em)

#figure(
  caption: [Paired intervention-versus-baseline tests on the _same-reward_ family
    ($n = 20$ seeds; baseline collision rate $28.5%$, baseline return
    $165.7$). Each row is one intervention. $p$ is the two-sided Wilcoxon signed-rank test on the 20 paired seed-level differences between that intervention and the baseline; $p_"Holm"$ is the Holm--Bonferroni correction applied across the six interventions, separately within each metric. @tab-interventions reports the same adjusted values rounded, alongside the point estimates and confidence intervals.],
  text(size: 9pt, table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: (x: 4pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header(
      table.cell(rowspan: 2, align: left + horizon)[*Intervention*],
      table.cell(colspan: 2, align: center)[*Collision rate*],
      table.cell(colspan: 2, align: center)[*Return*],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*$p$*], [*$p_"Holm"$*], [*$p$*], [*$p_"Holm"$*],
    ),
    table.hline(stroke: 0.5pt),
    [`hide`], [0.324], [0.324], [0.784], [0.784],
    [`permute`],
    [0.001],
    [0.003],
    [$1.91 times 10^(-5)$],
    [$7.63 times 10^(-5)$],
    [`randomise`],
    [$6.34 times 10^(-4)$],
    [0.003],
    [$2.67 times 10^(-5)$],
    [$8.01 times 10^(-5)$],
    [`desync`],
    [$6.69 times 10^(-4)$],
    [0.003],
    [$6.29 times 10^(-5)$],
    [$1.26 times 10^(-4)$],
    [`hide_kin`],
    [$4.49 times 10^(-4)$],
    [0.002],
    [$1.91 times 10^(-6)$],
    [$1.14 times 10^(-5)$],
    [`hide_all`],
    [$1.91 times 10^(-6)$],
    [$1.14 times 10^(-5)$],
    [$1.91 times 10^(-6)$],
    [$1.14 times 10^(-5)$],
    table.hline(stroke: 0.9pt),
  )),
)<tab-intervention-tests-samereward>

#v(1.5em)

#figure(
  caption: [Paired intervention-versus-baseline tests on the _random-signal_ family
    ($n = 20$ seeds; baseline collision rate $62.4%$, baseline return
    $109.1$). Each row is one intervention. $p$ is the two-sided Wilcoxon signed-rank test on the 20 paired seed-level differences between that intervention and the baseline; $p_"Holm"$ is the same value after a Holm--Bonferroni correction applied across the six interventions, separately within each metric. The two left-hand columns are for the collision rate and the two right-hand columns for the mean episode return.],
  text(size: 9pt, table(
    columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: (x: 4pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header(
      table.cell(rowspan: 2, align: left + horizon)[*Intervention*],
      table.cell(colspan: 2, align: center)[*Collision rate*],
      table.cell(colspan: 2, align: center)[*Return*],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*$p$*], [*$p_"Holm"$*], [*$p$*], [*$p_"Holm"$*],
    ),
    table.hline(stroke: 0.5pt),
    [`hide`], [0.926], [1.000], [0.985], [1.000],
    [`permute`], [0.911], [1.000], [0.729], [1.000],
    [`randomise`], [0.456], [1.000], [0.674], [1.000],
    [`desync`], [0.295], [1.000], [0.388], [1.000],
    [`hide_kin`], [0.277], [1.000], [0.002], [0.008],
    [`hide_all`], [0.627], [1.000], [0.001], [0.006],
    table.hline(stroke: 0.9pt),
  )),
)<tab-intervention-tests-random>

#v(1.5em)

#figure(
  caption: [_no-signal_ versus _bimodal_ baseline, unpaired ($n = 20$ seeds each). Two-sided Mann-Whitney U test on the seed-level values.],
  text(size: 9pt, table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: (x: 4pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header([*Metric*], [*no-signal*], [*bimodal*], [*$U$*], [*$p$*]),
    table.hline(stroke: 0.5pt),
    [Collision rate], [$34.9%$], [$38.5%$], [203.0], [0.946],
    [Mean return], [$187.9$], [$201.4$], [265.0], [0.081],
    table.hline(stroke: 0.9pt),
  )),
)<tab-nosignal-mw>

#v(1.5em)

#figure(
  caption: [Cross-seed correlations on the _bimodal_ family ($n = 20$ seeds). The first four appear in
    @fig-rho-cic-outcomes.],
  text(size: 9pt, table(
    columns: (1.7fr, 1fr, 1.15fr, 1fr, 1.15fr),
    align: (left, right, right, right, right),
    inset: (x: 4pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header(
      table.cell(rowspan: 2, align: left + horizon)[*Relationship*],
      table.cell(colspan: 2, align: center)[*Pearson*],
      table.cell(colspan: 2, align: center)[*Spearman*],
      table.hline(start: 1, end: 3, stroke: 0.4pt),
      table.hline(start: 3, end: 5, stroke: 0.4pt),
      [*$r$*], [*$p$*], [*$r_s$*], [*$p$*],
    ),
    table.hline(stroke: 0.5pt),
    [$rho$ vs collision rate], [$+0.188$], [0.427], [$+0.184$], [0.437],
    [$rho$ vs mean return], [$-0.215$], [0.362], [$-0.188$], [0.427],
    [CIC vs collision rate], [$-0.271$], [0.248], [$-0.226$], [0.339],
    [CIC vs mean return], [$+0.261$], [0.266], [$+0.223$], [0.346],
    [CDE vs CIC],
    [$+0.896$],
    [$9.63 times 10^(-8)$],
    [$+0.756$],
    [$1.14 times 10^(-4)$],
    table.hline(stroke: 0.9pt),
  )),
)<tab-correlations>
