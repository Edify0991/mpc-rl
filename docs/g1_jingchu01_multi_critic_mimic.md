# G1 / Jingchu01 MPC-Mimic Multi-Critic PPO

## Scope

Only the reference-motion tasks that simultaneously use CD-MPC landmarks and
whole-body imitation use multi critic:

- `Mjlab-MPC-RL-Mimic-Contact-G1-29DOF`;
- `Mjlab-Hierarchical-HybridMimic-MPC-G1-29DOF`;
- `Mjlab-MPC-RL-Mimic-Contact-Jingchu01-28DOF`;
- `Mjlab-Hierarchical-HybridMimic-MPC-Jingchu01-28DOF`.

THEMIS and the robot-specific paper-compatible velocity/loco-manipulation
tasks remain on the upstream single-critic PPO.  The Phase-2 student also
stays single-critic because its MPC landmark and future-contact rewards are
removed by design.

## Return decomposition

The scalar environment reward is unchanged.  After each simulator step, the
robot-local runner reconstructs the already weighted and time-scaled reward
terms from `RewardManager._step_reward` and forms a disjoint partition:

\[
r_t=r_t^{\mathrm{mpc}}+r_t^{\mathrm{mimic}}+
    r_t^{\mathrm{task}}+r_t^{\mathrm{reg}}.
\]

| Critic | Included terms |
|---|---|
| `mpc_landmark` | every `mpc_*` term and `future_contact_plan` |
| `mimic` | every dense `motion_*` joint/velocity/anchor/body imitation term |
| `task` | uprightness, posture, foot/contact behaviour, collision penalties, and `tracking_success` |
| `regularization` | joint-limit, action-rate, hybrid-torque, residual-action, and hierarchical-MPC-parameter penalties |

The runner checks at every step that the four channel sum equals MJLab's
scalar reward (within numerical tolerance).  Thus this is a critic-target
decomposition, not a reward reweighting or an extra reward source.

For each channel \(g\), a distinct privileged critic estimates
\(V_g(s_t)\), and its GAE is

\[
\delta_t^g=r_t^g+\gamma(1-d_t)V_g(s_{t+1})-V_g(s_t),\qquad
A_t^g=\delta_t^g+\gamma\lambda(1-d_t)A_{t+1}^g.
\]

The PPO actor uses the original total-return objective through

\[
A_t=\sum_g A_t^g,
\]

followed by the usual batch normalization.  Each critic is fitted to its own
GAE return.  Consequently, the policy gradient remains the gradient of the
unchanged total reward while value regression does not force physically
different MPC, imitation, stability, and regularization signals into one
value target.

## Implementation

Each robot package owns a copy of `multi_critic.py`; it contains the
feed-forward rollout storage, `MultiCriticPPO`, and
`MultiCriticVelocityOnPolicyRunner`.  The four critics use the existing
privileged `critic` observation group, have independent MLP parameters, and
are saved under `critics_state_dict`.  `finetune.py` recognizes this state
dictionary and shape-filters all four heads when resuming.

`tracking_success` in each robot's `mimic_mdp.py` is a sparse task outcome:
joint RMS error, anchor position error, and projected-gravity uprightness
must all pass their thresholds.  It complements the dense `motion_*` terms
instead of replacing them.

This implementation intentionally supports the existing feed-forward PPO
configuration only.  Recurrent multi-critic policies, RND, and symmetry
augmentation require an explicit sequence-aware extension rather than being
silently applied to decomposed returns.
