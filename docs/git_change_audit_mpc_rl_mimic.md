# MPC-RL Mimic / Contact 扩展：Git 工作区 17 文件审计

> 路径说明（后续重构）：此文为历史审计。通用实现现位于
> `training_common`/`mjlab_tools`，机器人私有训练配置位于
> `g1_training`、`jingchu01_training`；以下旧路径仅用于追溯当时的差异。

本文件基于当前 `git status --short` 编写，覆盖工作区中的 17 个改动文件：9 个已跟踪
修改、6 个新增 Python 模块、2 个新增 Markdown 文档。它说明每个文件的职责、所依据的数学
关系，以及“已接入运行时”和“仅提供后续扩展接口”的边界。

## 结论与范围

当前代码形成三条相关但不同的路径：

1. `Mjlab-MotionTracker-Themis`：BeyondMimic 风格的纯动作跟踪器；
2. `Mjlab-Hierarchical-HybridMimic-MPC-Themis`：一个 PPO actor 的快慢尺度 action，慢尺度
   参数化接触时钟、落足残差/方差和动量参考；
3. `Mjlab-MPC-RL-Mimic-Contact-Themis` / `...-Student-Themis`：本轮实现的两阶段非周期
   teacher/student。teacher 使用未来 (H) 帧参考驱动 CD-MPC，student 去除 MPC 与 contact-plan
   action。

“已实现”不等于“已证明全身稳定”：目前实际具备的是**固定 MPC 参数条件下的凸 QP**、接触计划
与 landmark 奖励。chance constraint、协方差闭环传播、终端集/终端控制器以及 DAgger 与 PPO 的
自动 runner 集成仍是后续工作。

## 17 个文件逐项说明

| # | 文件 | 状态 | 修改内容与运行时作用 |
|---:|---|---|---|
| 1 | `README.md` | 修改 | 登记 5 个 mimic 相关任务；解释 reference PD、两阶段 teacher/student、MPC parameter adaptor、wrench landmark 与 retargeting 前提。 |
| 2 | `pyproject.toml` | 修改 | 增加 `export-reference-centroidal` 命令行入口。 |
| 3 | `src/themis_mpc/contact_schedule.py` | 修改 | 增加非周期 `make_reference_contact_schedule`；扩展 phase-walking schedule 以接受慢尺度 timing、落足均值/方差和接触意图；在 QP 前冻结计划。 |
| 4 | `src/themis_training/__init__.py` | 修改 | 注册 tracker、hybrid、hierarchical、Phase-1 teacher 与 Phase-2 student 五个任务 ID。 |
| 5 | `src/themis_training/env_cfgs.py` | 修改 | 定义上述任务的场景、动作、观测、奖励、MPC 与终止配置。 |
| 6 | `src/themis_training/finetune.py` | 修改 | 让 fine-tune 入口也识别本项目的 `MotionReferenceCommandCfg`，从而接受本地 reference NPZ 或 artifact。 |
| 7 | `src/themis_training/mpc_grf_mdp.py` | 修改 | 把 motion reference、慢尺度 MPC 参数、快尺度接触计划、MPC trajectory/wrench landmark、接触监督奖励接入 `LocoMPCCommand`。 |
| 8 | `src/themis_training/rl_cfg.py` | 修改 | 增加 hierarchical、teacher 与 student 的 PPO rollout/熵系数配置。 |
| 9 | `src/themis_training/themis/themis_constants.py` | 修改 | 增加 effort-motor THEMIS articulation，供显式 PD 力矩控制使用。 |
| 10 | `docs/stochastic_contact_mpc_and_joint_mimic.md` | 新增 | 给出随机落足、chance constraint、协方差传播、条件稳定性、局部接触 patch 与层级 RL 的理论/扩展说明。 |
| 11 | `docs/two_stage_mpc_rl_mimic_contact.md` | 新增 | 给出 Phase-1/Phase-2 的精确接口、接触计划、奖励、terminal-tail、DAgger 边界和训练流程。 |
| 12 | `src/themis_training/dagger_distillation.py` | 新增 | 提供 DAgger buffer、teacher 49 维 action 的关节部分切片、监督损失与单步更新函数。 |
| 13 | `src/themis_training/export_reference_centroidal.py` | 新增 | 从 reference NPZ 与匹配 MJCF 导出逐帧 CoM、动量及候选接触位置。 |
| 14 | `src/themis_training/hybrid_mimic.py` | 新增 | 实现参考 PD 执行器；支持 joint residual、每脚 contact state/full-horizon contact plan、以及 held 高层 MPC 参数 action。 |
| 15 | `src/themis_training/mimic_mdp.py` | 新增 | 实现 BeyondMimic-format motion loader、严格名称校验、参考质心量、运动学接触候选/平滑、mimic 观测/奖励与 clip 尾部处理。 |
| 16 | `src/themis_training/mpc_parameter_net.py` | 新增 | 实现 GRU 型 MPC parameter adaptor、物理边界解码和仅用于低频候选评估的高斯落足采样。 |
| 17 | `src/themis_training/reference_centroidal.py` | 新增 | 用 MuJoCo inertial parameters 从整段参考运动重建质量一致的 CoM、线/角动量和接触点相对 CoM 向量。 |

以下按功能链路展开。

## A. 参考动作、质心量与接触计划

### 文件 15：`mimic_mdp.py`

新增 `MotionReferenceCommandCfg` 和 `MotionReferenceCommand`。它读取 BeyondMimic 格式 NPZ
中的 `joint_pos`、`joint_vel`、body pose/velocity、joint/body name，并把参考名称与当前 MuJoCo
robot 名称逐项匹配；G1 reference 不能直接拿去控制 THEMIS，必须先 retarget 或更换实体。

运行时功能包括：

- `motion_reference`：当前 (q^{ref},\dot q^{ref})；
- `motion_reference_preview`：actor 使用的一帧 future preview；
- `reference_centroidal_horizon`：MPC 使用的 (H+1) 帧 CoM/动量参考；
- `reference_contact_horizon`：MPC 使用的 (H) 帧 nominal contact schedule；
- `motion_joint_error_exp`、`motion_joint_vel_error_exp`、anchor/body reward：whole-body mimic；
- `motion_clip_complete`：有限 reference 末尾的 episode reset；
- terminal tail：非循环 clip 的越界 stage 保持末帧几何，但令速度、线动量与角动量归零，并通过
  `reference_horizon_valid` 标记接触监督无效。

接触标签优先从 NPZ 的 `reference_contact_key` 读取。无标签时使用候选接触点的平地运动学规则：

\[
y_{t,i}^{cand}=\mathbb{1}\left[
z_{t,i}-\min_s z_{s,i}\le h_{thr}\ \land\
\left\|\frac{p_{t+1,i}-p_{t-1,i}}{2}\,f_{fps}\right\|\le v_{thr}
\right].
\]

之后按 run-length hysteresis 填补短 swing gap，删除短 stance run。它是 MPC 的 nominal
schedule，不是物理接触真值；真实接触仍由仿真 sensor 提供。

### 文件 17：`reference_centroidal.py`

新增参考质心量重建。对第 (j) 个刚体，令局部惯性 CoM offset 为 (r_j^b)，世界 body orientation
为 (R_j)，则

\[
p_j^{com}=p_j+R_jr_j^b,\qquad
v_j^{com}=v_j+\omega_j\times(R_jr_j^b).
\]

总质量 (m=\sum_jm_j) 下，代码计算

\[
c=\frac{1}{m}\sum_jm_jp_j^{com},\qquad
l=\sum_jm_jv_j^{com},
\]

\[
k=\sum_j\left(I^w_j\omega_j+
(p_j^{com}-c)\times m_jv_j^{com}\right).
\]

这说明参考动量是参考构型、刚体速度、质量、惯性和惯性坐标系的函数；地面反力不需要作为“额外
动量项”加入该时刻定义。地面 wrench 决定的是下一时刻的动量变化。

接触点按 reference body origin 或局部 offset 计算：

\[
r_i=p_{body(i)}+R_{body(i)}d_i^b,\qquad r_i^{rel}=r_i-c.
\]

### 文件 13 与 2：`export_reference_centroidal.py`、`pyproject.toml`

提供离线导出器和 `export-reference-centroidal` CLI。它读取匹配 MJCF 的 `body_mass`、`body_ipos`、
`body_inertia`、`body_iquat`，调用上述质心计算，再把 
`com_pos_w`、`com_vel_w`、`linear_momentum_w`、`angular_momentum_w`、接触位置写回 NPZ。

这条离线路径用于检查 data contract；训练时 `mimic_mdp.py` 仍会从当前仿真模型重建，以避免
reference 与模型质量参数失配。

## B. 显式 PD 执行与两类策略 action

### 文件 9：`themis_constants.py`

增加 `THEMIS_EFFORT_ARTICULATION` 与 `get_themis_effort_robot_cfg`。原位置 actuator 不能接受
代码直接计算的 effort target；新实体改用 torque motor，保留每组 joint 的 effort limit 与 armature。

### 文件 14：`hybrid_mimic.py`

新增 `HybridMimicAction`。它不把 MPC wrench 经 (J^\top f) 加到关节力矩；实际执行始终是参考
PD：

\[
q_t^{des}=q_t^{tracker}+s_q\Delta q_t,
\]

\[
\tau_t=K_p(q_t^{des}-q_t)+K_d(\dot q_t^{ref}-\dot q_t),
\]

其中当前 Phase-1/Phase-2 使用 (s_q=0.30)。这符合 MPC-RL landmark 指导而非 force-feedforward
的控制分解。

该 action term 有三种可组合分量：

| 分量 | 维度/频率 | 用途 |
|---|---|---|
| `Δq` | 29，每步 | 形成 (q^{des})；Phase-2 唯一部署 action。 |
| contact state 或 plan | 2 或 (2H)，每步 | 形成 QP 前固定的 contact schedule；不是独立 GRU。 |
| high-level MPC parameter | 16，每 `high_level_decimation=5` 步保持 | 形成时钟、落足、动量参考的慢尺度参数。 |

`last_q_des`、`last_tau_pd`、raw contact plan 被保存，分别支持 DAgger target、力矩正则和接触计划
landmark。

## C. CD-MPC schedule、动力学、约束与 landmark

### 文件 3：`contact_schedule.py`

新增 `make_reference_contact_schedule`。Phase-1 以参考 schedule 为中心，策略输出 full horizon
残差 (A^\sigma\in\mathbb R^{H\times2})：

\[
\sigma_{t,k,i}^{mpc}=
\operatorname{clip}\left(
\sigma_{t,k,i}^{ref}+s_\sigma\tanh(A^\sigma_{t,k,i}),0,1
\right),\qquad s_\sigma=0.75.
\]

`r_LF/r_RF` 直接传入 reference 的 future contact positions，故它们不是 QP 决策变量。为避免探索
把一个 nominal stance 的所有接触删除，`preserve_nominal_support=True` 在 QP 外恢复参考中最强
支撑点到 (sigma=0.5)；reference flight stage 保持允许无接触。

旧 `make_walking_schedule` 也扩展为可接收：phase-rate scale、duty offset、touchdown mean residual、
touchdown std metadata、reference touchdown trajectory 与 2D fast contact intention。该路径主要用于
hierarchical phase-based task。

### 文件 7：`mpc_grf_mdp.py`

`LocoMPCCommandCfg` 新增以下开关和数据链路：

- `motion_command_name`：从 `MotionReferenceCommand` 取 (x^{ref})；
- `parameter_network_path`、history 和 bounds：加载 optional TorchScript GRU adaptor；
- `use_hierarchical_parameters`：使用 held PPO 高层 16D action；
- `use_policy_contact_state`：兼容旧每脚 2D action；
- `use_reference_contact_schedule` 与 `use_policy_contact_plan`：Phase-1 的 reference (+\) full-
  horizon residual schedule；
- `_com_traj/_vel_traj/_k_traj/_u_traj/_sigma_traj`：保存 MPC trajectory；
- `_contact_plan_valid`：mask reference 末尾越界的 delayed-contact 监督。

它将动作/网络输出复制并 detach 到 command buffer，在每个 QP solve 前解码；PPO 不穿过 MPC 或
MuJoCo 反向传播。

当前 centroidal state/control 为

\[
x_k=[c_k^\top,l_k^\top,k_k^\top]^\top,qquad
u_k=[f_{L,k}^\top,\tau_{L,k}^\top,f_{R,k}^\top,\tau_{R,k}^\top]^\top.
\]

对于固定 (h=0.07\,\mathrm{s})、接触位置 (r_{i,k})、activation (sigma_{i,k}) 与 CoM
linearization point (ar c_k)，实现的离散动力学是

\[
\begin{aligned}
c_{k+1}&=c_k+\frac{h}{m}l_k+\frac{h^2}{2m}\sum_i\sigma_{i,k}f_{i,k}+\frac{h^2}{2}g,\\
l_{k+1}&=l_k+h\sum_i\sigma_{i,k}f_{i,k}+hmg,\\
k_{k+1}&=k_k+h\sum_i\sigma_{i,k}
\left((r_{i,k}-\bar c_k)\times f_{i,k}+\tau_{i,k}\right).
\end{aligned}
\]

Phase-1 参考由 motion loader 写入

\[
x_{k}^{ref}=[c_k^{ref},l_k^{ref},k_k^{ref}],\qquad u_k^{ref}=0.
\]

求解目标为

\[
\min\sum_{k=0}^{N-1}\left(
\|x_{k+1}-x_{k+1}^{ref}\|_Q^2+
\|u_k-u_k^{ref}\|_R^2+
\|u_k-u_{k-1}\|_{R_\Delta}^2
\right),
\]

末端状态使用 (Q_f=10Q)。当前默认

\[
Q=\operatorname{diag}(Q_c,Q_l,Q_k),\quad
Q_c=(100,100,200),\ Q_l=(10,10,20),\ Q_k=(50,50,50),
\]

且 (R_f=R_\tau=10^{-4},R_\Delta=10^{-3})。每只 active foot 的局部 wrench 满足线性 friction/
CoP 近似：

\[
|f_x|\le\mu f_z,\quad |f_y|\le\mu f_z,\quad 0\le f_z\le f_z^{max},
\]

\[
|\tau_x|\le y_hf_z,\quad
-x_{toe}f_z\le\tau_y\le x_{heel}f_z,\quad
|\tau_z|\le\mu_zf_z.
\]

inactive contact 的完整 6D wrench 被置零。`mpc_grf_tracking` 使用 simulator foot force 和 MPC
force landmark 的 stance-gated RMS 误差；moment landmark 仅作为 critic reference，因为默认
contact sensor 不测 6D wrench。

`FutureContactPlanTracking` 把时刻 (t) 的 (sigma^{mpc}_{t,k}) 放入 FIFO，并在 (t+k) 与
hysteresis-filtered physical contact (y_{t+k}) 比较：

\[
r_t^{plan}=
\frac{\sum_{a=0}^{H-1}\gamma_c^a v_{t-a,a}
\exp\left(-\operatorname{mean}_i(\sigma^{mpc}_{t-a,a,i}-y_{t,i})^2/s_c^2\right)}
{\sum_{a=0}^{H-1}\gamma_c^a v_{t-a,a}}.
\]

这里 (v) 是 FIFO/clip-valid mask。该项是 delayed prediction-consistency reward；由于 receding-
horizon 会 replan，它不是“旧计划唯一导致未来接触”的因果证明。

### 条件凸性

一次 QP 内，网络/action 输出、(sigma)、(r)、(ar c) 都先固定。于是

\[
x_{k+1}=Ax_k+B_k(\bar c_k,r_k,\sigma_k)u_k+d
\]

对决策变量 ((x,u)) 为仿射等式，所有 friction/CoP/wrench bounds 为线性不等式，目标为凸二次。
若 (R\succ0)，控制块严格凸，因此在非空可行域上控制解唯一。接触位置或时刻若改为 QP 决策，
((r-c)\times f) 会产生双线性项，以上结论不再成立；必须改用固定线性化/SCP。

## D. 慢尺度参数网络与随机落足接口

### 文件 16：`mpc_parameter_net.py`

实现 `MPCParameterNet`：输入为 ([B,T,29]) 的 state/reference/contact history，经 GRU 和 MLP
输出 16 个 raw parameter。head 零初始化，因此未训练模型严格输出 nominal 参数。解码为

\[
\begin{aligned}
s_\phi&=s_{min}+(s_{max}-s_{min})\operatorname{sigmoid}(a_0),\\
\Delta d&=d_{max}\tanh(a_1),\\
\mu_{r,i,xy}&=r_{max}\tanh(a_{2:6}),\\
\sigma_{r,i,xy}&=\sigma_{min}+(\sigma_{max}-\sigma_{min})\operatorname{sigmoid}(a_{6:10}),\\
[\Delta l,\Delta k]&=[l_{max}\tanh(a_{10:13}),k_{max}\tanh(a_{13:16})].
\end{aligned}
\]

当前 bounds 为 (s_\phi\in[0.8,1.2])、(|\Delta d|\le0.1)、
(|\mu_{r,xy}|\le0.18\) m、(sigma_{r,xy}\in[0.005,0.1]) m、
(|\Delta l|\le12)、(|\Delta k|\le4)。数值 `mpc_dt` 不由网络改变。

`sample_touchdown_candidates` 以

\[
r_i^{(j)}=r_i^{ref}+\mu_{r,i}+\epsilon_i^{(j)},\qquad
\epsilon_i^{(j)}\sim\mathcal N(0,\operatorname{diag}(\sigma_{r,i}^2))
\]

构造镜像候选，适用于低频候选打分/Monte-Carlo 评估。它**没有**在每次 QP 内随机采样，因而当前
MPC 保持确定性 landmark。

## E. 任务配置、注册与训练入口

### 文件 5：`env_cfgs.py`

该文件新增/补充以下配置：

| 配置函数 | 内容 |
|---|---|
| `_apply_mpc_grf_features` | 增加 force 与 moment critic landmark observation。 |
| `themis_motion_tracker_env_cfg` | 纯 motion mimic reward，用于 tracker 预训练。 |
| `themis_hybrid_mimic_env_cfg` | effort robot + motion command + explicit PD action + mimic/MPC reward 基线。 |
| `themis_hierarchical_hybrid_mimic_env_cfg` | 29D joint residual、2D fast contact、16D 每 5 步 held parameter action；关闭冲突的 phase gait reward。 |
| `themis_mpc_rl_mimic_contact_env_cfg` | Phase-1。`Δq + 2H` action，`run_every_n_steps=1`，reference schedule、critic MPC landmarks、future-contact reward、finite clip termination。 |
| `themis_mpc_rl_mimic_student_env_cfg` | Phase-2。删除 `loco_mpc` command、MPC rewards/observations 和 contact-plan action，仅留 29D joint action 与 causal preview。 |

Phase-1 reward 的实际组织为

\[
r=r_{mimic}+r_{MPC\text{-}landmark}+r_{contact\text{-}plan}+r_{reg/safety},
\]

其中 mimic 包含 joint position/velocity、anchor/body tracking，MPC landmark 包含 CoM、CoM velocity、
angular momentum、GRF tracking，contact-plan 是上节 FIFO 项。`foot_gait` 被置零，避免 phase clock
标签与 reference schedule 冲突。

基础 task 的 reference contact point 默认是 `FOOT_L/FOOT_R` body origin；可用
`contact_point_offsets_b` 移至 toe 或 heel。它仍是**每脚一个 6D wrench**模型；若同时需要 toe/heel
patch，必须扩展为多个 point contact，而不能让单个接触点在 QP 内自由移动。

### 文件 4、8、6：注册、PPO 配置和 fine-tune

- `__init__.py` 注册 5 个任务；所有当前均使用 `VelocityOnPolicyRunner`；
- `rl_cfg.py` 为 hierarchy/teacher/student 分设 experiment name、40 step rollout 和 entropy coefficient；
- `finetune.py` 将 `MotionReferenceCommandCfg` 纳入 tracking-task 判断，确保
  `--env.commands.motion.motion-file` 能用于新增任务。

## F. Phase-2 DAgger

### 文件 12：`dagger_distillation.py`

teacher 的 full action 是 ([a^{joint}_{29},A^\sigma_{2H}])，student 只保留前 29 维。函数
`teacher_joint_action_from_full_action` 明确执行此切片。监督放在归一化 action 坐标：

\[
\mathcal D\leftarrow\mathcal D\cup
\{(o_t^{student},a_{t,T}^{joint},z_{t,T}^{MPC})\},
\]

\[
\mathcal L_{DAgger}=
\lambda_q\|a_{S}^{joint}-a_T^{joint}\|^2+
\lambda_z\|\hat z_S-z_T^{MPC}\|^2.
\]

因为两阶段使用同一 reference 和 `joint_target_residual_scale=0.30`，匹配 (a^{joint}) 与匹配
(q^{des}) 等价，但前者不会把 Gaussian policy raw action 与绝对 joint angle 直接做 MSE。

已实现：ring buffer、sample、loss、optimizer update、optional MPC landmark auxiliary head。

未实现：现有 `VelocityOnPolicyRunner` 没有 frozen teacher query hook；普通 PPO 训练不会自动收集
DAgger 样本或插入 supervised minibatch。实际 Phase-2 训练需要在 student rollout 后、PPO update 前
调用本模块，或新增专用 runner。

## G. 文档性理论与未来扩展

### 文件 10：`stochastic_contact_mpc_and_joint_mimic.md`

该文档不是额外运行时代码，而是把已实现和未实现理论边界写清楚。

1. 高斯落足：(r_i=\bar r_i+\epsilon_i\), 
   (epsilon_i\sim\mathcal N(0,\Sigma_{r,i}))。均值表示修正方向，方差表示不确定性，二者不能混用。
2. 半空间 chance constraint：对可落足多边形 (Hr\le h)，风险分配后可用保守 back-off

\[
H_j\bar r_i+\Phi^{-1}(1-\epsilon_j)
\sqrt{H_j\Sigma_{r,i}H_j^\top}\le h_j.
\]

   当前没有 contact surface half-space (H,h)，因此该约束**尚未实现**。
3. 协方差传播：在线性化 ((\bar r,\bar f,\bar c)) 下，

\[
\Sigma_{x,k+1}=A_{cl,k}\Sigma_{x,k}A_{cl,k}^\top+
\sum_iE_{r,i,k}\Sigma_{r,i,k}E_{r,i,k}^\top+W_k,
\]

   其中 
   (E_{r,i,k}=[0,0,-h[\bar f_{i,k}]_\times]^\top)。当前没有 (W,K,\Sigma) 的校准，故不能声称
   chance-safety。
4. 条件 recursive feasibility/practical ISS：文档给出了需要终端集、终端控制器、模型误差界和
   whole-body wrench realizability 的充分条件。当前没有这些硬约束，所以不能把该定理作为现有代码
   的实验结论。
5. 局部脚板接触：文档提出 toe/heel/edge micro-contact/patch 的后续模型。当前 sensor 的 foot mesh
   任一点接触即可产生 foot-level label，但 MPC 仍用完整 foot wrench cone，不能正确表达 point/line
   support。

### 文件 11：`two_stage_mpc_rl_mimic_contact.md`

该文档对应本轮已接入的 Phase-1/Phase-2 task，包含：49D/29D action、reference contact
preprocessing、MPC cost、critic landmark、delayed contact reward、terminal tail、DAgger 接口和遥操作
下只有一帧 reference 时应部署 student 的说明。

## H. 用户文档

### 文件 1：`README.md`

README 是上述任务的入口说明：任务表、PD 控制形式、reference NPZ contract、THEMIS/G1 retargeting
限制、parameter adaptor 的数值边界、wrench landmark 的传感器边界，以及两份理论/实现文档链接。

## 建议的阅读与验证顺序

1. 先读 `reference_centroidal.py` 与 `mimic_mdp.py`，确认 reference 的坐标、body 名称、质量参数和
   contact label；
2. 再读 `contact_schedule.py`、`mpc_grf_mdp.py`，确认哪些量在 QP 前固定；
3. 查看 `env_cfgs.py` 中选中的任务，避免把 hierarchical phase task 与非周期 Phase-1 teacher 混用；
4. Phase-2 训练前，先实现/接入 DAgger teacher-query runner hook；
5. 若研究 toe/heel/edge 或概率安全，按 `stochastic_contact_mpc_and_joint_mimic.md` 的 contact-patch/
   chance-constraint 路线扩展，而不要直接把接触位置改成 QP 决策变量。
