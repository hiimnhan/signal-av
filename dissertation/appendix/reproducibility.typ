// == Seeds <app-seed>
// we use 20 seeds which are ${1, 13, 33, 42, 59, 270, 515, 999, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200}$.
//
== Notation <app-notation>

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    [*Symbol*], [*Meaning*],

    [$Theta$, $theta$], [Set of latent driver types; a single type. $|Theta| = 2$.],

    [$cal(M)$, $m$], [Set of discrete tokens; a single emitted token. $|cal(M)| = 3$.],

    [$N$], [Number of controlled agents ($= 6$).],
    [$K$], [Number of observed neighbours per agent ($= 5$).],
    [$W$], [History window of past tokens observed ($= 5$).],
    [$L$], [Number of lanes ($= 3$).],
    [$o$], [Receiver's local observation (kinematics of neighbours).],

    [$p(theta, m)$], [Empirical joint frequency of type $theta$ emitting token $m$.],

    [$p(theta)$, $p(m)$], [Marginal type and token frequencies.],
    [$p(m | theta)$], [Emission distribution: token given type.],
    [$p(theta | m)$], [Posterior: type given token.],

    [$I(Theta; cal(M))$], [Mutual information between type and emitted token (bits).],

    [$rho$], [Separation coefficient, $I(Theta; cal(M)) \/ log |Theta| in [0, 1]$.],

    [SC], [Speaker Consistency, $frac(1, |Theta|) sum_theta max_m p(m | theta)$.],

    [CIC], [Causal Influence of Communication (nats).],
    [CDE], [Counterfactual Decoding Effect (action units).],

    [$pi^("drive")$], [Receiver's drive-action policy.],
    [$mu(o, m)$], [Mean of the receiver's drive-action distribution.],
    [$D_("KL")$], [Kullback--Leibler divergence.],
  ),
  caption: [Notation used throughout. Structural constants are repeated in
    @tab-hyperparameters.],
) <tab-notation>

== Hyperparameters <app-hyperparameters>

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, center),
    [*Parameter*], [*Value*],
    [Learning rate], [$3 times 10^(-4)$],
    [Rollout steps $T$], [512],
    [PPO epochs per iteration], [4],
    [Minibatch size], [128],
    [PPO clip ratio $epsilon$], [0.2],
    [GAE $lambda$], [0.95],
    [Discount $gamma$], [0.99],
    [Value coef $c_v$], [0.5],
    [Signal entropy start $c_m^0$], [0.08],
    [Signal entropy end $c_m^"end"$], [0.005],
    [Anneal iters $T_"anneal"$], [1200],
    [Aux coef $c_"aux"$], [0.5],
    [Gradient clip norm], [0.5],
    [Total iterations], [1600],
    [TypeEncoder hidden dimension $d$], [64],
    [DeepSets output dimension $d_"agg"$], [64],
    [Signal/response hidden dimension], [64],
    [Critic hidden dimension], [128],
  ),
  caption: [Hyperparameters.],
) <tab-hyperparameters>

== Hardware, software and simulation environment configuration <app-environment>

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, left),
    [*Component*], [*Specification*],
    [Processor], [Apple M4 Max (14 cores)],
    [Memory], [36 GB unified],
    [Machine], [MacBook Pro M4 Max],
    [Operating system], [macOS 26.5.2 (build 25F84)],
    [Accelerator], [Apple Metal (MPS); CUDA not used],
    [Python], [3.11.15],
    [PyTorch], [2.11.0],
    [highway-env], [1.10.2],
    [Gymnasium], [1.3.0],
  ),
  caption: [Hardware and software configuration.],
) <tab-environment>

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, left),
    [*Component*], [*Specification*],
    // table.cell(colspan: 2)[_Simulation environment_],
    [Controlled agents $N$], [6],
    [Background (IDM) vehicles], [0],
    [Lanes $L$], [3],
    [Episode duration], [40 s],
    [Control frequency], [5 Hz],
    [Observation], [Multi-agent kinematics, 6 vehicles per agent (ego plus $K = 5$ neighbours), relative, normalised],

    [Action space], [Continuous longitudinal $+$ lateral, clipped to $[-1, 1]$],
    [Reward speed range], [20--30 m/s],
    [Seeds per condition], [20],
  ),
  caption: [Simulation environment configuration.],
) <tab-sim-environment>
