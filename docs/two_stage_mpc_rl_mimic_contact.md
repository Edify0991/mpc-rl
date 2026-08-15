# 两阶段非周期动作模仿：MPC-RL Teacher 与因果 RL Student

本文档对应两个独立任务：

- `Mjlab-MPC-RL-Mimic-Contact-Themis`：Phase-1 full-preview MPC-RL teacher；
- `Mjlab-MPC-RL-Mimic-Student-Themis`：Phase-2 无 MPC、仅输出关节目标的因果 student。

它们不改变已有的 `Mjlab-Hierarchical-HybridMimic-MPC-Themis`。后者仍保留慢尺度
16 维 MPC 参数动作和原有 phase-based 设计，不能与本文的非周期 teacher 混称。

## Phase 1：reference-plan MPC-RL teacher

### 信息与动作

在策略时刻 \(t\)，teacher actor 只接收机器人本体状态和一个即将到来的参考关节帧
\([q^{ref}_{t+1},\dot q^{ref}_{t+1}]\)，而不是把 \(H\) 帧 preview 泄露给 actor。其动作是

\[
a_t=[\Delta q_t,\;A^\sigma_t],\qquad
\Delta q_t\in\mathbb R^{29},\quad A^\sigma_t\in\mathbb R^{H\times2}.
\]

策略的 29 维关节分量是归一化的关节目标残差；实际执行关节目标为

\[
q^{des}_t=q^{ref}_t+0.30\Delta q_t.
\]

接触部分不是从零生成日程，而是对 reference schedule 的有界残差。对每足、每个
horizon stage：

\[
\sigma^{mpc}_{t,k,i}=
\operatorname{clip}\left(
\sigma^{ref}_{t,k,i}+0.75\tanh(A^\sigma_{t,k,i}),0,1
\right).
\]

因此 raw action 为零时严格恢复参考接触计划；策略仅在参考不可行、扰动或接触时序
偏差出现时修正它。

为避免探索动作把一个 nominal 支撑阶段的两足同时关闭，默认的 pre-QP guard 会恢复参考中
激活最大的那只足至 \(\sigma=0.5\)。reference 本来就是 flight 的阶段不作该限制。该投影在
QP 之外执行且结果被冻结；它是可行性 safeguard，不会改变 QP 的线性/凸性。

### 参考接触计划与位置

motion loader 提供完整 future horizon：

\[
\{c^{ref}_{t:t+H},l^{ref}_{t:t+H},k^{ref}_{t:t+H},
 r^{ref}_{L,t:t+H-1},r^{ref}_{R,t:t+H-1},
 \sigma^{ref}_{t:t+H-1}\}.
\]

`contact_state[T,2]` 可通过 `reference_contact_key` 从 NPZ 显式读取，顺序固定为
`[left, right]`。未提供标签时，加载器针对每只候选足端计算相对最低高度和中心差分速度，
再执行：短 swing gap 填补、短 stance run 删除。最小 stance/swing 长度分别由
`reference_contact_min_stance_frames` / `reference_contact_min_swing_frames` 控制。

这是仅基于运动学的 nominal contact extractor，适合作为 Phase-1 基线；它不是接触真值。
正式实验应优先使用压力、力板或 retargeted-simulator contact 标签，并扩展至 toe/heel/edge
micro-contact 点。参考触地点 \(r^{ref}\) 在每一次 QP 调用中固定，绝不作为 QP 决策变量。
基础 biped task 使用 `FOOT_L` / `FOOT_R` 两个候选点；可通过
`contact_point_offsets_b={"FOOT_L": (...), "FOOT_R": (...)}` 将点移到鞋尖/鞋跟的局部位置。
若要同时建模 toe 与 heel，应把 CD-MPC 从当前的两接触 wrench 模型扩展为四个接触点，而不是把
两个点的力错误地合并为一个可任意移动的位置。

### CD-MPC

状态与控制为

\[
x=[c,l,k],\qquad u=[f_L,\tau_L,f_R,\tau_R].
\]

对固定数值步长 \(h=0.07\,\mathrm{s}\) 和固定
\((r^{ref},\sigma^{mpc},\bar c)\)，每个 QP 解决：

\[
\min_{x,u}\sum_{k=0}^{H-1}
\|x_{k+1}-x^{ref}_{k+1}\|_Q^2+
\|u_k\|_R^2+
\|u_k-u_{k-1}\|_{R_\Delta}^2.
\]

其中 \(x^{ref}\) 的 CoM、线动量和角动量都由 motion loader 的质量一致性重建结果给出，
`u_ref=0`。当前权重为 `Q_c/Q_l/Q_k`、`R_f_foot/R_tau_foot/R_delta`。

Phase-1 令 `run_every_n_steps=1`，使每个 full-horizon 接触计划进入下一次 QP；这不改变
数值 MPC 的固定 \(h\)。critic 观察并存储 CoM、角动量、接触力和完整 contact-plan landmark，
actor 不观察这些 \(H\) 帧特权量。

### 奖励

总回报分为：

\[
r=r_{mimic}+r_{MPC-landmark}+r_{contact-plan}+r_{reg/safety}.
\]

- `motion_joint_pos`、`motion_joint_vel`、`motion_anchor`、`motion_body`：mimic；
- `mpc_com_tracking`、`mpc_com_vel_tracking`、`mpc_ang_mom`、`mpc_grf_tracking`：MPC landmark；
- `FutureContactPlanTracking`：将计划 \(\sigma^{mpc}_{t,k}\) 置入 FIFO，并于真实时间
  \(t+k\) 以经过法向力 on/off hysteresis 滤波的 MuJoCo foot contact 标签比较；
- 力矩、动作、关节限位和碰撞项：regularization/safety。

future-contact 项是“未来预测一致性”辅助奖励。由于 MPC 会 receding-horizon replan，远期
计划并不单独因果决定未来接触；不能把该奖励解释为严格的因果接触控制证明。episode reset
会清除 FIFO valid mask，跨 clip 的预测不参与评分。对非循环 clip，`mpc_contact_plan_valid`
标记超过片段末尾的 horizon stage，`FutureContactPlanTracking` 会将这些 stage mask 掉。

### 条件凸性审查

当前设计没有把 CD-MPC 变成非线性问题：在 QP 调用前，
\(A^\sigma_t\)、\(\sigma^{mpc}\)、接触位置 \(r^{ref}\)、线性化 CoM \(\bar c\) 都被冻结。
因此 centroidal dynamics 对 wrench \(u\) 仿射，摩擦、CoP、wrench 上界保持线性，目标是
凸二次项。连续 \(\sigma\) 仅缩放已固定的约束系数。

若以后让 QP 同时优化接触位置或接触时间，则
\((r-c)\times f\) 会出现双线性项；必须采用 SCP/冻结线性化点，不能沿用此凸性结论。

## Phase 2：因果 DAgger student

student 环境删除 `loco_mpc` command、contact-plan action、MPC reward 和 MPC observation。
它只输出 29 维 \(\Delta q\)，执行相同的
\(q^{des}=q^{ref}+0.30\Delta q\)。`q^{ref}` 是外部命令/观测，不是 student action。

teacher 拥有完整 reference horizon、MPC 和 \(2H\) 接触计划；student 只保留本体状态和
单帧 future reference。因此应在 **student 实际访问的状态** 上 query frozen teacher：

\[
\mathcal D\leftarrow\mathcal D\cup
\{(o_t^{student},a_{t,T}^{joint},z_{t,T}^{MPC})\}.
\]

使用的监督损失为

\[
\mathcal L_{DAgger}=
\lambda_q\|a^{joint}_S-a^{joint}_T\|^2+
\lambda_z\|\hat z_S-z_T^{MPC}\|^2.
\]

其中 \(a^{joint}\) 是环境动作坐标下的 29 维 normalized residual；因为 teacher/student 使用同一
reference 和 `0.30` 缩放，它等价于匹配 \(q^{des}\)，但避免把 raw Gaussian 输出直接与绝对关节角
做 MSE。\(\hat z_S\) 是仅训练期的 optional auxiliary landmark head，部署时删除。student 同时继续接受
PPO 的 mimic/regularization 回报；建议先以较大 DAgger 权重初始化，再逐步降低该权重并增加
student rollout 占比。teacher/student 当前 actor observation 都是因果单帧 reference，便于在同一
状态上 query；teacher 的完整 preview 只进入其 MPC/critic，不进入 actor。

`g1_training.dagger_distillation` 与 `jingchu01_training.dagger_distillation` 已提供：

- `DaggerReplayBuffer`：保存 student observation、teacher joint-action target、可选 MPC landmark；
- `teacher_joint_action_from_full_action`：从 49 维 teacher action 中取前 29 维，显式丢弃
  仅训练期使用的 contact-plan 分量；
- `dagger_distillation_loss`：目标损失；
- `update_student`：单个监督 update。

当前 `VelocityOnPolicyRunner` 原生只有 PPO，没有 frozen-teacher query hook。因此 DAgger
collector 必须在每次 student rollout 后、PPO update 前调用 teacher 并使用上述组件插入 supervised
minibatch；这是一条明确的 runner 扩展接口，而不是已由普通 `mjlab train` 自动完成的步骤。

完整的 runner 时序、snapshot 同步、\(\beta\) teacher/student source mixing、PPO mask、
\(\lambda_{DAgger}\) 退火、landmark normalization、deterministic teacher mean 和验证清单见
[`dagger_ppo_integration_plan.md`](dagger_ppo_integration_plan.md)。在该专用 runner 实现前，
student 任务仍只能进行普通 PPO，`dagger_distillation.py` 只能被外部训练循环显式调用。

## 参考结束与遥操作

两任务均将 `motion.loop=False`，并在终端帧 reset。非循环 clip 的 future-contact reward 已 mask
掉超出数据末端的 stage；MPC 在这些 stage 使用已实现的静态 tail：末帧 CoM/contact 几何保持，
CoM 速度、线动量和角动量置零。若末帧仍处于动态腾空或单脚切换，必须离线追加恢复/站立 tail；
不能把动态最后帧无限复制成未来动作。

实时遥操作若只有一帧 \(q^{ref}\)，student 可直接部署。teacher/MPC 路径则需要采用当前命令
保持或短时速度外推后的 terminal reference；它不能获得真实 future contact。若希望在该模式下
保留 MPC，应训练一个因果 reference-completion module，或保守地缩短 preview 影响范围。
