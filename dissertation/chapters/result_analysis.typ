#import "@preview/subpar:0.2.2"

The previous chapter set out the experimental conditions, the evaluation metrics, and the intervention battery this chapter draws on throughout. What follows reports what they showed: we begin with the separation coefficient, since it answers RQ1 and RQ2 directly, then turn to the intervention results and the receiver-side decoding measures.

// === Signalling is near-pooling and emerges only under type heterogeneity
// A.2 - answers RQ1 + RQ2
The distribution of the separation coefficient across the three signalling conditions shown in @fig-rho-condition answers both research questions. RQ1 asks whether type-contingent signalling emerges at all. In the bimodal condition, the token carried a small amount of type information, mean $rho = 0.091$ of a maximum of $1.0$. Moreover, it was statistically higher than both the same-reward and the random-signal controls (see @tab-correlations and the accompanying tests in @app-statistics). RQ1 was therefore answered: weak, type-associated signalling emerged, above the controls but far short of separation. On the operational cut used here ($rho < 0.1$) the mean sat just inside the pooling range, but the seeds straddled it: 7 of the 20 exceeded $0.1$, and the highest reached $0.227$. The population was better described as weakly pooled on average than as uniformly pooled. We use the term throughout to describe the measured distribution, not to claim that the population reached a pooling equilibrium.

A single end-of-training value cannot distinguish an emergence that never formed from one that formed and then collapsed back to pooling. We therefore tracked $rho$ over training (refer to @fig-rho-training). Averaged over seeds, $rho$ rose early and then settled onto a stable plateau near its final value of about $0.09$, without decaying back towards zero. Consequently, the weak type-signal association was a stable end state of training rather than a transient, and $rho$ stayed low throughout.

RQ2 asks whether the emergence is due to type-dependent reward heterogeneity rather than the signalling architecture itself. With the architecture and signalling head held fixed and only the reward difference between the types removed, $rho$ collapsed to near 0 ($approx 0.013$). RQ2 was thus answered: emergent signalling was contingent on that reward heterogeneity, and disappeared once it was removed.


#figure(
  placement: auto,
  image("../figures/rho_by_condition.png", width: 80%),
  caption: [$rho$ by condition. Bars are means over the 20 seeds and error bars are
    95% seed-level confidence intervals (Student-$t$ over the seed means); points
    are individual seeds.],
)<fig-rho-condition>

#figure(
  placement: auto,
  image("../figures/rho_over_training.png", width: 80%),
  caption: [$rho$ over training for the bimodal condition. Faint lines are the 20
    individual seeds; the solid line is the mean, and the shaded
    band is the 95% seed-level confidence interval at each logged iteration; the
    dashed line marks the pooling threshold $rho = 0.1$.],
)<fig-rho-training>

#figure(
  placement: auto,
  caption: [Evaluation-time interventions on the bimodal family ($N = 6$ agents,
    20 seeds). Point values are seed-level means. $Delta$ columns give the paired
    seed-level difference from baseline with a 95% $t$-confidence interval. $p$ is
    a two-sided Wilcoxon signed-rank test over the 20 paired seed-level
    differences, Holm-Bonferroni corrected across the six interventions
    separately within each metric. Exact unadjusted and adjusted $p$-values are
    given in @tab-intervention-tests.],
  text(size: 9pt, table(
    columns: (1fr, 0.52fr, 1.55fr, 0.62fr, 0.62fr, 1.68fr, 0.62fr),
    align: (left, right, right, right, right, right, right),
    inset: (x: 3pt, y: 4.5pt),
    stroke: none,
    fill: (_, y) => if y >= 2 and calc.even(y) { luma(247) },
    table.hline(stroke: 0.9pt),
    table.header(
      table.cell(rowspan: 2, align: left + horizon)[*Intervention*],
      table.cell(colspan: 3, align: center)[*Collision rate*],
      table.cell(colspan: 3, align: center)[*Return*],
      table.hline(start: 1, end: 4, stroke: 0.4pt),
      table.hline(start: 4, end: 7, stroke: 0.4pt),
      [*%*], [*$Delta$ (%)*], [*$p$*], [*mean*], [*$Delta$*], [*$p$*],
    ),
    table.hline(stroke: 0.5pt),
    [`baseline`], [38.5], [---], [---], [201.4], [---], [---],
    [`hide`],
    [40.5],
    [+2.1 \[−2.4, +6.5\]],
    [.659],
    [195.7],
    [−5.8 \[−13.0, +1.5\]],
    [.040],
    [`permute`],
    [63.6],
    [+25.1 \[+16.2, +34.1\]],
    [.001],
    [130.4],
    [−71.0 \[−92.6, −49.4\]],
    [< .001],
    [`randomise`],
    [71.0],
    [+32.5 \[+24.7, +40.3\]],
    [< .001],
    [110.3],
    [−91.1 \[−112.5, −69.7\]],
    [< .001],
    [`desync`],
    [71.7],
    [+33.2 \[+25.3, +41.1\]],
    [< .001],
    [110.5],
    [−90.9 \[−113.6, −68.3\]],
    [< .001],
    [`hide_kin`],
    [45.5],
    [+7.0 \[−11.7, +25.7\]],
    [.866],
    [66.4],
    [−135.0 \[−185.3, −84.7\]],
    [< .001],
    [`hide_all`],
    [36.0],
    [−2.5 \[−20.6, +15.7\]],
    [.866],
    [80.5],
    [−120.9 \[−169.9, −72.0\]],
    [< .001],
    table.hline(stroke: 0.9pt),
  )),
)<tab-interventions>


A low $rho$ does not, on its own, reveal how senders were emitting the token. Both types might be emitting more or less at random, or each might be settling firmly on one token while the two simply settle on the same one. Speaker Consistency tells these apart, asking how strongly a single type sticks to its favoured token regardless of what the other type does. Across the bimodal seeds, each type held to its preferred token about $62%$ of the time, far above the $33%$ (select one out of three tokens) that random emission would give (refer to @fig-sc-condition), so the choices were deliberate rather than noisy. Placed next to $rho approx 0.09$, the picture is pooling with consistency: every type commits to a token, but both commit to much the same one.

Read the other way round, the posterior $p(theta | m)$ in @fig-sig-heatmap gives the probability that a type is implied by a token. Each token is only weakly diagnostic: token 0 implied the cautious type about two-thirds of the time, while the others sat closer to the chance value of $0.5$. This confirms weak, type-associated signalling rather than a type-revealing code. Because seeds settle on different conventions, averaging the posterior across seeds dampens the per-seed structure, so this figure understates the separation present within any single run.

#figure(
  placement: auto,
  image("../figures/sc_by_condition.png", width: 100%),
  caption: [Speaker Consistency by intervention, bimodal family. The dashed line marks the chance level
    $1 slash #h(0pt) |cal(M)| = 0.33$.],
)<fig-sc-condition>

#figure(
  placement: auto,
  image("../figures/signal_heatmaps.png", width: 100%),
  caption: [Signal use in the bimodal condition, mean over 20 seeds. (Left)
    $p(m | theta)$, how each type distributes its tokens. (Right)
    $p(theta | m)$, the type implied by a received token. The $plus.minus$ values
    are 95% seed-level confidence intervals on each cell, computed as
    $1.96$ standard errors over the 20 seeds.
  ],
)<fig-sig-heatmap>



Although it carries so little type information, the signalling token still has a meaningful effect on the system. The token and the kinematic channel carried complementary loads. @tab-interventions shows that manipulating the token from its kinematic consequence in _permute_, _randomise_ and _desync_ raised the collision rate by 65% to 86% relative to baseline. Suppressing the neighbour kinematics (_hide_kin_) produced the opposite signature: collisions were statistically unchanged ($38.5%$ to $45.5%$, corrected $p=0.866$), while mean return fell by roughly two-thirds ($201.4$ to $66.4$, corrected $p < 0.001$).

Interestingly, hiding the token from receivers while leaving neighbour driving intact (_hide_) left safety essentially unchanged (collision rate $38.5%$ to $40.5%$, corrected $p=0.659$). @fig-corruption-controls shows a population trained with no token at all crashed at $34.9%$ against the communicating population's $38.5%$, a difference not detectable at twenty seeds per condition (refer to @tab-nosignal-mw). As noted in @experiment-controls,  this _no-signal_ condition is not a clean removal of the channel alone, and the comparison should be read with that in mind. Safety therefore does not depend on the token being present, or even on a binding existing in the first place. What raises the collision rate is specifically corrupting an established binding, so that the token and the behaviour it accompanies disagree. The presence or absence of either on its own does not.

A 65-to-86% rise in crashes when the token is scrambled could mean two different things: the agents genuinely rely on the token, or they are simply fragile to any disturbance at all. To tell these apart, we ran the same token interventions on two control families: the _random-signal_ and the _same-reward_. As shown in @fig-token-corruption, scrambling the token hurt safety a great deal in the _bimodal_, less in the _same-reward_, and not at all in the _random-signal_, where none of the token-corruption effects were significant. The agents that never learned to use the token were unharmed when it broke. The bimodal penalty is therefore genuine reliance on the token, not a general sensitivity to perturbation.


#figure(
  placement: auto,
  image("../figures/token_corruption_by_family.png", width: 100%),
  caption: [Change in collision rate under the three token-corrupting interventions,
    by training family. Bars are means over the 20 seeds of each family and error
    bars are 95% seed-level confidence intervals on the paired difference from that
    family's own baseline. The penalty tracks whether a family had learned to use
    the token: large in the _bimodal_, smaller in the _same-reward_, and absent in
    the _random-signal_, whose agents never bound the token to behaviour. Exact tests are
    in @app-statistics.],
)<fig-token-corruption>


#figure(
  placement: auto,
  grid(
    columns: 2,
    row-gutter: 1.2em,
    [
      #image("../figures/crash_by_condition.png", width: 80%)
    ],
    [
      #image("../figures/return_by_condition.png", width: 80%)
    ],
  ),
  caption: [Collision rate and return by condition under the unmodified baseline.
    Left: the collision rate. Right: the mean episode return. Bars
    are means over the 20 seeds of each condition, error bars are 95% seed-level
    confidence intervals, and points are individual seeds.],
)<fig-corruption-controls>

The interventions above alter the token only from the sender's perspective. To identify decoding from downstream behaviour, we evaluated the CIC and CDE. According to @fig-cde, in the _bimodal_ condition the mean CDE shifted by $0.126$ when the received token was scrambled, roughly four times that of the other two conditions. The shift was predominantly longitudinal (acceleration $0.099$ against steering $0.043$). CIC gave the same ordering: $0.053$ nats in the bimodal condition, about $4.8%$ of channel capacity (three tokens; $ln(3) approx 1.099$) and roughly seven times the other two controls ($approx 0.008$) as shown in @fig-cic. Together, they indicate that the token was decoded by the receiver. However, this is only a slight nudge to how the receiver actually acted.

#figure(
  placement: auto,
  image("../figures/cde_by_condition.png", width: 80%),
  caption: [Counterfactual Decoding Effect across conditions. Left: change in the receiver's mean
    action when the received token is altered with the scene frozen, by
    condition. Right: per-dimension change in the bimodal family. Bars are means over
    the 20 seeds, error bars are 95% seed-level confidence intervals, and points
    are individual seeds.],
)<fig-cde>

#figure(
  placement: auto,
  image("../figures/cic_by_condition.png", width: 80%),
  caption: [Causal Influence of Communication by condition (nats). Bars are means
    over the 20 seeds, error bars are 95% seed-level confidence intervals, and
    points are individual seeds.],
)<fig-cic>
Whether a seed developed more type information or stronger decoding did not translate into better driving. Across the twenty seeds, neither $rho$ nor CIC was significantly correlated with the collision rate or the mean return (refer to @fig-rho-cic-outcomes). The information the token carries and the strength with which receivers decode it are therefore decoupled from task performance, consistent with a channel whose safety load rests on the token-behaviour binding rather than on how much the symbol reveals.
#figure(
  placement: auto,
  image("../figures/rho_cic_outcomes.png", width: 80%),
  caption: [Seed-level relationships between the signalling metrics ($rho$, CIC) and
    task outcomes (collision rate, mean return) across 20 bimodal seeds. Each point
    is one seed and the line is an ordinary least-squares fit. Each panel is
    annotated with the Pearson $r$ and the Spearman $r_s$ over the 20 seeds, with
    the associated $p$-value. All four relationships are weak and none is
    significant. Exact values are given in @tab-correlations.],
)<fig-rho-cic-outcomes>



In summary, across twenty seeds per condition, the emergent token carried little information about hidden type yet its integrity was worth a 65-to-86% rise in the collision rate. The load-bearing element was the binding between the token and the driving it commits to, not the receiver's reading of the symbol. Hiding the token from receivers left safety untouched at a return cost under 3% (corrected $p=0.040$), whereas breaking the token-kinematic binding by any means raised collisions by 65% to 86%. The receiver-only counterfactual nonetheless confirmed that receivers do decode the token as a soft, speed-oriented cue decoupled from type separation. The kinematic channel carried the complementary efficiency load.
