#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 15pt)
#show math.equation: set text(font: "New Computer Modern Math")

#table(
  columns: (auto, auto, auto, 4.0in),
  align: (left, center, center, left),
  [*Intervention*], [*Token*], [*Kinematics*], [*What it tests*],

  [Baseline], [intact], [intact], [Unmodified baseline.],

  [Hide],
  [$bold(0)$],
  [intact],
  [Is the discrete token $m$ informationally necessary, or do kinematics
    alone suffice for receivers?],

  [Randomise],
  [$"Unif"(cal(M))$],
  [intact],
  [Does the type-signal convention carry load, or only the structural
    existence of the token channel?],

  [Permute],
  [intact],
  [mismatched],
  [Does the receiver rely on kinematic consistency with the token, or
    does it read the token symbolically?],

  [Desync],
  [intact],
  [random/step],
  [If the kinematic image is per-step noise but the token survives
    intact, does receiver behaviour degrade?],

  [Hide_kin], [intact], [$bold(0)$], [Is the kinematic channel necessary? (Direct complement of Hide.)],

  [Hide_all], [$bold(0)$], [$bold(0)$], [What is the total information value of the neighbour channel?],
)
