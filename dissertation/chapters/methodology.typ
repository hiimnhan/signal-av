== Problem Formulation


The highway scenario is modelled as a Partially Observable Stochastic Game
(POSG). The game is defined by the tuple
$cal(G) = (cal(I), cal(S), cal(O), cal(A), cal(M),
  T, R, gamma)$,
where $cal(I) = {1, dots, N}$ is the set of $N$ controlled agents, $cal(S)$ is the joint environment state, $cal(O)$ is the observation space, $cal(A) = [-1, 1]^2$ is a continuous action (longitudinal acceleration, steering), $cal(M) = {0, 1, 2}$ is a discrete signal token, $T$ is the transition function, $R$ is the reward function, and $gamma in (0, 1)$ is the discount factor.

Each agent $i in cal(I)$ carries a latent type $theta_i in Theta = {0, 1}$ representing cautious and assertive styles of driving, respectively. It is drawn independently at episode reset from a fixed population distribution $p_Theta$. In the bimodal condition, $p_Theta(0) = p_Theta(1) = 0.5$. Types are private and persist for the full episode while there are no persistent agent identifiers across episodes.

At each decision step $t$, agent $i$ selects:
$
  (a_i^t, m_i^t) in cal(A) times cal(M)
$

The token $m_i^t$ is bound to an execution profile $pi(m_i^t)$ that
modifies the kinematic envelope within which $a_i^t$ is rendered. More details on the execution profiles are in @exec-profile.

The policy for agent $i$ decomposes into two heads sharing an encoder.
The signal head commits to a token first, the response head then conditions
on that committed token:

$
  pi_(i)(a_i^t, m_i^t | o_i^t, theta_i)
  = underbrace(
    sigma_(i)(m_i^t | o_i^t, theta_i),
    "signal head"
  )
  times
  underbrace(
    pi_i^("drive")(a_i^t | o_i^t, m_i^t),
    "response head"
  )
$

The signalling strategy $sigma_i: cal(O) times Theta ->
Delta(cal(M))$ maps local observation and own type to a distribution
over tokens. The response strategy $pi_i^"drive": cal(O) times cal(M) ->
Delta(cal(A))$ maps local observation and the committed signal token to
a continuous action distribution. Conditioning on $m_i^t$ lets the response head specialise to the
committed kinematic envelope. The own type $theta_i$ is not an input to the response head.


Agent $i$ maintains an implicit belief $b_(i)(theta_j | h_j)$ over the
type of each visible neighbour $j$, where $h_j$ is the observed
behavioural history. The analytical reference point is on-path Bayesian
updating:

$
  b_(i)(theta_j | m_j^t, h_j) prop
  sigma_(j)(m_j^t | s_j, theta_j) dot b_(i)(theta_j | h_j)
$

In this work, $b_i$ is not maintained explicitly. Instead, the type
encoder (@type-encoder) produces a distributed representation of
$b_i$ from the raw observation window, with gradients flowing from
the downstream loss.

== Experiments

=== Simulation Environment <simulation-environment>

The simulation environment is a multi-agent extension of the gym-based
`highway-env` simulator #cite(<leurent2018environment>). The road is a straight
three-lane highway. Each episode runs for 40 seconds at a control frequency
of 5 Hz, yielding 200 decision steps at a decision interval of
$Delta t = 0.2$ s.

#figure(
  image("../figures/sim_screenshot.png", width: 80%),
  caption: [Illustration of the simulation environment.],
)

All $N = 6$ vehicles on the road are policy-controlled learning agents, and
no background traffic is added. The study focuses solely on the behaviour that emerges among the controlled agents, and the
environment is kept as simple as possible so that scripted human-driven
traffic cannot introduce additional dynamics that would confound the
signal.

=== Observation Space <observation-space>

At each step $t$, agent $i$ observes its own state plus a short history for each of its $K = 5$ nearest neighbours:

$
  o_i^t = (o_i^"self", {O_(i j), M_(i j)}_(j=1)^K)
$

where $o_i^"self"$ is the agent's own speed, lane index, and type label $theta_i$.

For each neighbour $j$, $O_(i j) in RR^(W times 5)$ is a $W$-step history of five per-step kinematic features: whether the slot is occupied (presence), the longitudinal and lateral offset to the neighbour ($Delta x$, $Delta y$), and the neighbour's longitudinal and lateral velocity relative to agent $i$ ($Delta v_x$, $Delta v_y$). The window length equals the neighbour count, $W = K = 5$, so each observation spans the last second of interaction.

$M_(i j) in {-1, 0, 1, 2}^W$ is the matching $W$-step history of tokens emitted by neighbour $j$. When fewer than $K$ neighbours are within range, the unused slots are padded with $-1$.

=== Execution Profiles <exec-profile>

Token $m in {0, 1, 2}$ maps to a kinematic profile $pi(m)$. @tab-signal-profile shows the mapping.

This mapping is fixed by design, identical across all agents and seeds. Emitting a token deterministically selects the corresponding kinematic profile, whatever the sender's private type. What is learned is not this binding but two other things: (1) which token a sender chooses to emit (the signalling strategy $sigma_i$), and (2) how a receiver's driving responds to a received token (the response strategy $pi_i^"drive"$).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (center, left, center, center, center, center),
    [*Token*], [*Profile*], [*Headway Multiplier*], [*Jerk Cap (m/s³)*], [*Gap Accept (m)*], [*Lateral Drift*],

    [0], [Conservative], [1.5], [1.0], [15.0], [0.3],
    [1], [Neutral], [1.0], [2.0], [10.0], [0.6],
    [2], [Assertive], [0.7], [4.0], [6.0], [1.0],
  ),
  caption: [Signal token to kinematic execution profile mapping.],
)<tab-signal-profile>

Four quantities parameterise each profile. The Headway Multiplier
scales the desired following distance relative to a neutral baseline, so values
above 1 increase the target gap and values below 1 reduce it. The Jerk Cap
bounds the per-step change in acceleration, controlling
smoothness of longitudinal dynamics. The Gap Accept sets the base
threshold distance to the vehicle ahead below which forward acceleration
is suppressed. It is multiplied by the headway multiplier to give the
effective desired gap $g^*$. The Lateral Drift amplifies steering commands
whose magnitude exceeds a small threshold (0.1). Higher
values produce more decisive lane changes observable in the kinematic
history.

The longitudinal acceleration is jerk-capped each step:

$
  a_t = "clip"(tilde(a)_t,
    a_(t-1) - kappa Delta t,
    a_(t-1) + kappa Delta t)
$

where $tilde(a)_t$ is the raw policy output, $a_(t-1)$ is the previous
shaped acceleration, $kappa$ is the profile jerk cap, and
$Delta t = 0.2$ s. The cap enforces physically plausible acceleration
trajectories. Without it, the policy could produce instantaneous
acceleration reversals that real vehicles cannot execute.
When a front vehicle is within the desired gap
$g^* = "gap_accept" times "headway_mult"$, positive acceleration is
scaled by $g slash g^*$ where $g$ is the current bumper-to-bumper gap.

=== Reward Structure

The per-agent reward is a standard driving reward over collision avoidance, forward progress, lane preference, and on-road presence, weighted according to the agent's private type $theta_i$. The weight vectors per condition are shown in @tab-reward-weight. For agent $i$ at step $t$,

$
  r_i^t = w_(theta_i)^"col" c_i^"col" + w_(theta_i)^"spd" c_i^"spd"
  + w_(theta_i)^"lane" c_i^"lane" + w_(theta_i)^"road" c_i^"road"
$

where the four components are computed from the agent's own state alone:

$
  c_i^"col" = bb(1)["crashed"], quad
  c_i^"spd" = "clip"(frac(v_i - v_min, v_max - v_min), 0, 1), \
  c_i^"lane" = frac(ell_i, L - 1), quad
  c_i^"road" = bb(1)["on road"]
$

$c_i^"col"$ is the crash indicator for the current step. $c_i^"spd"$ is agent $i$'s speed normalised onto the reward speed range $[v_min, v_max] = [20, 30]$ m/s and clipped, so it saturates at 1 rather than paying for unbounded speed. $c_i^"lane"$ indexes the lane occupied, $ell_i in {0, ..., L-1}$ with $L = 3$, normalised so the leftmost lane scores 0 and the rightmost 1; this is a right-lane preference in the convention of the base simulator @leurent2018environment, not a lane-keeping penalty. $c_i^"road"$ indicates agent $i$ is still on the drivable surface. All four are bounded in $[0, 1]$, so the weights in @tab-reward-weight are directly comparable in scale.

The two types differ only in these weights. The cautious type pays more for a collision ($-2.0$ against $-1.5$) and is rewarded far less for speed ($0.2$ against $0.8$), which is what makes type behaviourally legible in the kinematics and is the type-dependent reward heterogeneity that RQ2 removes.

There is no reward term for the signal, which means no bonus for type-consistent emission, no penalty for type-inconsistent emission. This is a deliberate methodological commitment. If the reward directly incentivised truthful signalling, RQ1 would be answered by construction such that separation would emerge because we paid for it. Keeping the reward free of any signal-dependent term ensures that the regime observed is an endogenous consequence of interaction structure, not reward engineering.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    [*Type*], [$w^"col"$], [$w^"spd"$], [$w^"lane"$], [$w^"road"$],

    [0 (cautious)], [-2.0], [0.2], [0.2], [1.0],
    [1 (assertive)], [-1.5], [0.8], [0.05], [1.0],
  ),
  caption: [Reward weight vectors per type.],
)<tab-reward-weight>


== Network Architecture

@fig-arch gives an overview of the full per-agent pipeline including all components used. Each component is described in turn below.

=== TypeEncoder <type-encoder>

A single-layer GRU #cite(<cho2014learning>) with hidden size $d = 64$
maps the $W$-step sequence to a final hidden state, which is the
neighbour embedding.

For each neighbour $j in cal(N)_i$, the TypeEncoder maps the
$W$-step observation window to a fixed-size embedding. The input at
window step $w in {0, dots, W-1}$ is:

$
  x_(i j)^w = O_(i j)^w parallel "one-hot"(M_(i j)^w)
$

where $O_(i j)^w$ contains the five per-step kinematic features
as mentioned in @observation-space, $"one-hot"(M_(i j)^w)$ is the
standard one-hot encoding of the token, and $parallel$ denotes concatenation.


#figure(
  placement: auto,
  image("../figures/dissertation-architecture.png"),
  caption: [System architecture diagram. Neighbour kinematics
    and signal history pass through the shared TypeEncoder to produce a neighbour
    embedding, which the permutation-invariant DeepSets aggregator reduces to a
    neighbourhood aggregate. The Signal Head conditions on self-observation and
    own type to sample a token; the Response Head then conditions on
    self-observation, the neighbourhood aggregate, and that sampled token to
    produce the action distribution. The AuxHead supervises the neighbour
    embedding with a privileged type-prediction loss during training only, and
    is discarded at evaluation. The Centralised Critic pools all agents'
    self-observations, types, and neighbourhood aggregates to estimate the joint
    state value],
)<fig-arch>

Recurrent encoders over temporal trajectory windows
have been used for driver style inference in prior DRL work
#cite(<liu2024automatic>). GRU is chosen over LSTM because
the short window ($W = 5$) does not require LSTM's additional cell state to avoid vanishing gradients. Despite of this simplicity, it achieves comparable sequence modelling
with fewer parameters #cite(<cho2014learning>).
Self-attention is also not used as the quadratic cost and
positional-encoding overhead are unnecessary for sequences of length
five.

=== DeepSets Aggregator

The $K$ neighbour embeddings are aggregated with a
permutation-invariant DeepSets layer #cite(<zaheer2017deep>):

$
  z_i = rho(sum_(j in cal(N)_i) phi(hat(z)_(i j)))
$

where $phi: RR^64 -> RR^64$ is a Multilayer Perceptron (MLP) applied independently to each
neighbour embedding, $rho: RR^64 -> RR^64$ is an MLP applied to the
masked sum.
Absent neighbour slots are zeroed before summation.

A permutation-invariant aggregator is required in this setting. An agent's neighbours form an unordered, variable set: the slot in which a neighbour appears is arbitrary, and neighbours enter and leave as vehicles move. However, the policy must return the same decision under any reordering of the same set. Each neighbour moreover carries both its kinematics and its emitted token, and these pairs must be pooled symmetrically so that what a neighbour signals is read independently of where it happens to sit in the observation. DeepSets guarantees this invariance by construction.

DeepSets #cite(<zaheer2017deep>) is preferred over Social Attention
#cite(<leurent2019SocialAttentionAutonomous>) because the two architectures solve different problems. Social Attention routes selectively toward high-relevance neighbours via ego-as-query, neighbour-as-key/value attention. It is useful when the policy must identify a specific priority vehicle, such as one blocking a merge. However, the task here requires estimating the neighbourhood type distribution (how many cautious or assertive vehicles are nearby), a summary statistic for which uniform sum pooling, as in DeepSets, is provably sufficient. Moreover, DeepSets requires fewer parameters and less computation than its alternatives.

=== Signal Head

The signal head maps the concatenation of self-observation, own type,
and neighbourhood aggregate to a categorical distribution over tokens:

$
  sigma_i (dot | o_i^"self", theta_i) =
  "Softmax"(f_(sigma)([o_i^"self" parallel e_(theta)(theta_i) parallel z_i]))
  in Delta(cal(M))
$

where $e_theta: Theta -> RR^16$ is a learned type embedding, $f_(sigma)$ is an MLP, and $Delta(cal(M))$ is the probability simplex over tokens. The token is sampled as $m_i^t tilde sigma_i$.

=== Response Head

The response head maps self-observation, neighbourhood aggregate, and the
committed signal token to a bivariate action distribution:

$
  mu_i^a = f_(pi)([o_i^"self" parallel z_i parallel e(m_i^t)])
$

where $e: cal(M) -> RR^8$ is a learned token embedding, and $f_(pi)$ is an MLP. The token embedding allows the head to specialise its action distribution to each kinematic envelope without pre-assigning signal semantics.

=== Auxiliary Type Prediction Head

During training, an auxiliary head attached to the TypeEncoder
predicts the ground-truth type of each observed neighbour from its
embedding:

$
  hat(p)(theta_j | hat(z)_(i j)) = "Softmax"(f_("aux")(hat(z)_(i j)))
$

where $f_"aux"$ is an MLP with a two-class output. The training cross-entropy loss is:

$
  cal(L)_"aux" = -frac(1, sum_(i,j) bb(1)[theta_j >= 0])
  sum_(i in cal(I)) sum_(j in cal(N)_i, theta_j >= 0)
  log hat(p)(theta_j | hat(z)_(i j))
$

The mask $theta_j >= 0$ excludes unoccupied neighbour slots (type $= -1$).
The head is removed at evaluation; only the trained encoder weights are
retained.

The auxiliary objective serves two roles. First, the Proximal Policy Optimisation (PPO) @schulman2017PPO reward (collision penalty and speed) carries no direct gradient about why a neighbour behaved a certain way. Without explicit supervision the aggregate could encode proximity and velocity while ignoring latent type entirely. Dense cross-entropy supervision provides a direct gradient path into TypeEncoder and DeepSets, forcing the shared representation to become type-discriminative. Second, signalling problems often face the chicken-and-egg dilemma: the sender has no incentive to emit type-consistent tokens until the receiver can decode them, and the receiver representation does not encode type until the sender does so
consistently #cite(<eccles2019biases>). The auxiliary head breaks this deadlock by supervising the receiver-side representation from raw kinematics before any convention exists, mirroring the role-inference auxiliary tasks used in cooperative MARL #cite(<wang2020roma>) #cite(<jaderberg2017reinforcement>). This loss acts only on the receiver-side encoder and never touches the sender's signal head, so it does not prescribe which token a given type should emit. It makes type easier to represent from observed kinematics and past tokens.

=== Centralised Critic

Under decentralised execution, each agent's local value function faces a non-stationarity problem #cite(<lowe2017multi>). As all agents' policies co-evolve, the
transitions observed by agent $i$ appear non-Markovian.
Conditioning the critic on the joint state restores stationarity during
training, since the joint state is Markovian even as individual policies change. This centralised-training decentralised-execution (CTDE) principle was introduced for multi-agent actor-critic by #cite(<lowe2017multi>) and its effectiveness for cooperative tasks confirmed empirically by #cite(<yu2022surprising>). Joint state access also reduces advantage variance relative to local critics, since the value estimate is not confounded by unobserved co-agent behaviour.

The centralised critic takes the joint state of all $N$ agents:

$
  V_phi = f_(V)([o_1^"self" parallel e_V (theta_1) parallel z_1 parallel
      dots.h.c parallel o_N^"self" parallel e_V (theta_N) parallel z_N]) in RR
$

where $e_V: Theta -> RR^16$ is a critic-private learned type embedding
(separate from the policy's $e_theta$), $f_V$ is an MLP, and $parallel$ denotes concatenation. The aggregate vectors $z_i$ are computed with
the shared policy encoder and treated as constants when updating the
critic. The critic is used only during training.

== Training Procedure <training-procedure>


We train the agents with MAPPO #cite(<yu2022surprising>). Training runs in iterations. Each iteration first collects $T = 512$ environment steps. It then estimates how much better than average each action was, using generalised advantage estimation (GAE) #cite(<schulman2015high>) normalised to zero mean and unit variance. Finally, it updates the policy (minibatch size 128, clip ratio $epsilon = 0.2$).

The policy and the critic are trained with separate Adam optimisers
#cite(<kingma2015adam>). Keeping them separate isolates their gradients. The
policy update changes only policy parameters and the critic update changes only
critic parameters, so the critic's learning signal cannot leak into the
policy's representation.

To stop the agents from settling on one token too early, we add a
signal-entropy bonus to the policy loss. The signal entropy is the Shannon
entropy @shannonMathComm of the token distribution the signal head outputs: it is near zero when
an agent almost always emits the same token, and maximal when it spreads evenly
over all tokens. Rewarding higher entropy therefore discourages an early
collapse to a pooling equilibrium in which the token carries no information
about type.

The strength of this bonus is reduced linearly from $c_m^0 = 0.08$ to
$c_m^"end" = 0.005$ over the first $T_"anneal" = 1200$ iterations. Starting
high pushes the policy to try all three tokens early and lowering it later lets a stable convention settle once that exploration is complete. The bonus rewards spread across tokens, not any particular token-to-type pairing, so like the auxiliary head it discourages premature collapse without prescribing what the population should converge on.

All training hyperparameters are listed in @app-hyperparameters. Hardware, software and simulation configuration is shown in @app-environment.

== Evaluation Metrics
=== Experiment Controls <experiment-controls>
We conducted experiments under four conditions, each trained with twenty independent seeds. The principal condition, _bimodal_, assigned the two driver types distinct reward weights as in @tab-reward-weight. The _same-reward_ condition retained the full architecture but assigned both types identical weights (those of the cautious driver), removing the type-dependent reward heterogeneity while leaving the channel intact. The _random-signal_ condition retained that reward heterogeneity but replaced the emitted token with a uniform draw during training, severing the type-to-token link while keeping the channel present. The _no-signal_ condition retained the bimodal reward heterogeneity but removed the communication channel altogether: no token is emitted or received and agents observe only neighbour kinematics, as in a standard multi-agent driving setup, with every agent following the neutral dynamics profile since no token remains to ground one. It removes the channel to provide the kinematics-only task baseline. Because it carries no token, the signalling metrics ($rho$, SC, CIC, CDE) do not apply to the _no-signal_ condition.

These four controls are each needed for a different reason. The _bimodal_ condition is the baseline. The _same-reward_ and _random-signal_ conditions keep the channel intact; each isolates one factor, and together they rule out two alternative explanations for a positive $rho$ in _bimodal_. The _same-reward_ condition rules out the possibility that any signalling-shaped architecture would show apparent signal even without reward heterogeneity to explain: if $rho$ in the _bimodal_ condition is genuinely driven by having two types to distinguish (RQ2), removing only the reward-weight difference between types should collapse it. The _random-signal_ condition rules out the possibility that the measured $rho$ is an artefact of the architecture rather than something earned by learning. Breaking token-to-type binding, it gives the noise floor any real signal has to beat. The _no-signal_ condition answers a coarser question, whether the channel matters for task outcomes at all. It is worth noting that this condition is not a clean channel-only ablation. Because the channel is grounded, profile selection has no input besides the token, so we cannot remove the token while leaving profile choice intact unless we create a new system. However, its mechanism is still comparable to without-channel training in the sense that we choose a default action profile for everyone. This comparability also holds at the level of what is trained. In the _no-signal_ condition, the signal head and the response head's signal-embedding table are never invoked in the forward pass and so receive no gradient, while every component that produces driving behaviour, the type encoder, the DeepSets aggregator, the response head's driving trunk, the critic, and the auxiliary type-prediction head, is optimised under the same objective as in the _bimodal_ condition.


=== Task Outcomes

We report two task outcomes, one for safety and one for efficiency. The collision rate is the fraction of a seed's evaluation episodes in which at least one collision occurs, reported as a percentage; lower is safer. The mean return is the average reward an agent earns per episode, which is high when it makes steady progress at the target speed without crashing; higher is better. We compute each outcome per seed, averaging over that seed's evaluation episodes, and then summarise across the twenty seeds.

=== Type-Signal Mutual Information

The central measurement is the empirical mutual information @shannonMathComm between the latent type and the emitted signal, estimated from evaluation rollouts:

$
  I(Theta; cal(M)) = sum_(theta in Theta) sum_(m in cal(M))
  p(theta, m) log frac(p(theta, m), p(theta) p(m))
$

where $p(theta, m)$ is the empirical joint probability of type $theta$ emitting signal $m$, $p(theta)$ is the marginal probability of type $theta$ and $p(m)$ is the marginal probability of signal $m$.

We report $I(Theta; cal(M))$ in bits and normalise by $log |Theta|$ to give a separation coefficient $rho in [0, 1]$. Operationally, $rho < 0.1$ indicates _pooling_ (the signal carries essentially no information about type); $rho > 0.9$ indicates _separating_ (the signal determines type); intermediate values indicate _partial-pooling_. The distribution of $rho$ across seeds and conditions directly addresses RQ1 and RQ2.



=== Emergent Communication Metrics <supp-diag>


*Speaker Consistency (SC)* #cite(<jaques2019social>) measures how
consistently each type commits to a preferred token:

$
  "SC" = frac(1, |Theta|) sum_(theta in Theta) max_(m in cal(M)) p(m | theta)
$

High $"SC"$ is necessary but not sufficient for separation. A policy could achieve $"SC" = 1$ by always emitting the same token regardless of type (pooling equilibrium). SC and $rho$ together distinguish pooling-with-consistency from genuine type-signal separation.

*Causal Influence of Communication (CIC)* #cite(<lowe2019pitfalls>)
measures whether receivers actually change their actions in response to the received token:

$
  "CIC" = D_"KL" ( pi^("drive")(dot | o, m) parallel bb(E)_(m' tilde p(m)) [pi^("drive")(dot | o, m')] )
$

CIC is estimated by sampling action distributions under actual received
signals versus shuffled signals drawn from the empirical marginal $p(m)$.
A near-zero CIC indicates the response head ignores the token channel while elevated CIC confirms receivers condition meaningfully on received signals. CIC is reported in nats.

*Counterfactual Decoding Effect (CDE)* is a new metric developed for scrutinising the effect of the token on the receiver's next action kinematically. It measures the same receiver-side dependence directly in the action space.

Let $mu(o, m)$ denote the mean of the
receiver's drive-action distribution $pi^("drive")(dot | o, m)$, where $o$ is the receiver's local observation and $m$ is the tuple of tokens it receives from neighbours. At each recorded state the scene is held fixed (the sender's policy, action, and kinematics unchanged) and the received tokens are replaced by a counterfactual draw $m'$. CDE is the expected Euclidean shift in the mean action:

$
  "CDE" = bb(E)_(s, thin m' tilde "Unif"(cal(M))) thin
  norm(mu(o_s, m') - mu(o_s, m))_2 ,
$

taken over recorded states $s$ for which $m' eq.not m$. Because $mu$ is the
distribution mean, no action-sampling noise enters, and the shift is
attributable to decoding alone. We report the norm across the two action
dimensions and, separately, the per-dimension magnitude of the longitudinal
(acceleration) and lateral (steering) controls, in the action units.

The CDE is complementary to the CIC. Both isolate the receiver's response to the token, but they measure it on different scales. CIC quantifies the influence as an information-theoretic divergence. It reports how much the message reduces uncertainty about the action, without a physical scale or a direction. CDE instead holds the whole scene fixed and reads the change directly in the action space, so it is expressed in the units of the control the agent applies and resolves which action dimension moves. In short, CIC asks whether the token affects the next action of the receiver, whereas CDE asks how much, and along which axis, the action itself shifts.

=== Intervention Battery

A non-zero $rho$ shows the token and the driver's type are correlated. However, it does not tell us whether the token is genuinely coupled to behaviour or whether receivers act on it. The correlation could even be incidental: if cautious agents happen to emit token 1 for kinematic reasons unrelated to any communicative act, $rho$ would be elevated even though the token plays no signalling role. To test the relationship between the token and the kinematics it is meant to govern, and whether receivers exploit it, we apply the following six interventions, each evaluated on the bimodal condition over 200 episodes.

@tab-intervention-battery shows the intervention battery. The precise modification for each condition is as follows.

*Baseline.* Everything works as trained: agents drive, emit tokens, and observe neighbours normally. This is the unmodified baseline against which all ablations are compared.

*Hide.* Receivers are blind to the signal token. They can still see how
neighbours drive, but the symbol is hidden.

*Randomise.* A random draw replaces the token each agent emits. The agent choose a token, but that choice is discarded. For example, a cautious agent whose signal head outputs token 0 may have that output replaced with token 2 at this step; the receiver's signal window reflects the random draw, not the policy's intention. Because the replacement is independent of type, $rho$ is expected to be approximately near 0 by construction under this condition.



#figure(
  table(
    columns: (auto, auto, auto, auto),
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
  ),
  caption: [Intervention battery. Token and kinematics columns describe
    the state of each channel as seen by receivers or applied by the
    environment. $bold(0)$ denotes zeroing; mismatched/random\/step
    denotes kinematic execution decoupled from the emitted token.],
)<tab-intervention-battery>


*Permute.* Token labels are scrambled with a fixed code. For example,
under permutation $pi = {0 -> 2, 1 -> 0, 2 -> 1}$, an agent emitting token 0 (intending conservative) triggers the assertive execution profile, while receivers still observe token 0 in the signal window. The label reaches receivers intact but its physical meaning is wrong.

*Desync.* The token an agent emits is left untouched and stays visible to neighbours, but the execution profile it should trigger is re-drawn at random every step, independently of the token. For example, an agent emits token 0 (conservative) yet is driven with the assertive profile this step, then emits token 2 the next step but is driven as neutral. Neighbours see a genuine token history, but the kinematics behind it are pure noise.

Desync and Permute both decouple the token from its kinematic consequence, but differ in whether the decoupling is stable. Permute applies a fixed scramble consistently, so a systematic (but wrong) token-kinematic relationship remains learnable. Desync destroys that relationship entirely at every step. Greater degradation under Desync than Permute therefore indicates that receivers rely on the time-to-time kinematic image of the token rather than the token label alone.

*Hide\_kin.* Receivers can see the signal token but the driving behaviour of neighbours is hidden.

*Hide\_all.* Receivers see neither the token nor the driving behaviour of any neighbour.
