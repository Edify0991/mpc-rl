# CD-MPC–RL Motion Mimic：理论、代码与 RAL 投稿评估

**审查日期：2026-08-19。** 本文以当前 `main`（`88dc1cd`）为依据，区分已经
落地的实现、可以严格成立的理论命题和仍只是研究假设的部分。结论先行：工作有
一条值得投入的 RAL 级主线，但当前代码与证据尚不足以投稿；必须把“关节可行域
内嵌”和“纠偏度驱动的自适应融合”实现为可验证的方法，而不能仅作为动机。

## 1. 研究问题与建议的论文定位

原始 [MPC-RL](https://arxiv.org/abs/2606.05687) 使用训练期的 centroidal-dynamics
MPC 产生 CoM、动量、接触 wrench 等 landmark 来引导 RL。该降阶模型具有快速、
可并行求解的优点，但它不表示全身关节状态、执行器力矩或速度约束。

建议不要把论文定位成“BeyondMimic + MPC + multi-critic”的拼接。更准确、也更有
机会的定位是：

> **Reference-consistent, actuator-aware CD-MPC guidance for whole-body
> humanoid motion tracking under randomized dynamics.**

它应回答一个清晰问题：对于不一定动力学可行的重定向运动，如何在保持固定接触
mode 的 batched CD-QP 凸性和吞吐量的前提下，(i) 从全身运动获得一致的参考
centroidal 量，(ii) 以经过线性化且可校准的关节可行域约束限制 MPC wrench，
并且 (iii) 仅在 MPC 确实需要偏离参考时放宽局部关节 mimic？

## 2. 当前代码事实：已完成、部分完成、未完成

| 目标 | 当前证据 | 判定 |
|---|---|---|
| 保留原仓库任务 | `themis_training` 保留原任务；G1、JC01 分为独立 entry point；原始速度/locomanip MDP 留在各 `mpc_grf_mdp.py` | 已完成，边界清楚 |
| 精确参考 (c,l,k_G) | `training_common/reference_centroidal.py` 由 MuJoCo mass、inertial offset、惯量和 link 运动计算 CoM、线动量、关于系统 CoM 的角动量 | 已完成，但“精确”仅相对于模型、命名/速度语义正确的 reference body 集合 |
| 参考接触序列 | `mpc_grf_mimic_mdp.py` 优先读取标签，否则以足端高度、速度和最短支撑/摆动长度推断；每次 QP 固定该 horizon schedule | 已完成为启发式/标签流程；尚非力接触真值或可行性保证 |
| MPC landmark 与 mimic 奖励分离 | 独立 `mimic_mdp.py` 的 `motion_joint_pos/vel`；独立 mimic MPC 与 `MpcExactCentroidalLandmarkTracking` | 已完成；没有追踪全局 root pose 的密集 reward，符合预期设计 |
| 多 critic | 四个回报通道的独立 critic；通道和被检查等于原标量 reward | 已完成为**固定权重的值函数分解** |
| 关节速度/力矩硬约束进入 MPC | 当前 MPC 只有 wrench cone、CoP、力界；没有 Jacobian、(M(q))、(ddot q)、关节约束矩阵 | 未实现 |
| DR 同步进入 MPC 动力学 | G1/JC01 locomotion/mimic 现于 reset/solve boundary 从 simulator 读取每环境 mass、foot friction、body inertia/actuator 快照，并 batched 重建 CD-QP 的 `A,B,d,G,h` | 已实现质量/足端摩擦同步；惯量/actuator 仍需 joint-feasibility 线性化才可进入硬约束；独立 LocoManip solver 未覆盖 |
| 用“纠偏度”自适应融合 mimic/MPC | Jingchu01 现为固定 `actor_advantage_weights=(1.5,1,1,1)`；没有 correction metric、时变权重或带权 rollout return | 未实现 |

两个容易混淆的事实必须明确写入论文：

1. `MimicLocoMPCCommand.current_centroidal_state()` 使用完整仿真刚体运动学，且
   参考量在相同模型惯性参数下重建。这比根部近似严格得多；但 reference body
   漏配、retarget 速度噪声、接触点偏移或模型参数误差仍会破坏“物理真值”。
2. 当前 `HybridMimicAction` 的 wrench **不**映射成 actuator feed-forward torque；
   实际命令仍是显式 PD 加 residual。`torque_limit` 默认是 `None`，而逐关节的
   effort/velocity limits 虽已在 constants 中定义，但没有成为 MPC 约束。

## 3. 三项创新的理论判断

### 创新一：从重定向运动重建 reference centroidal/contact，再以关节 mimic 补全降阶模型

**结论：理论上成立，且是当前最成熟的贡献；单独新颖性中等。**

给定每个刚体的惯性 CoM 速度，使用

\[
c=\frac{1}{m}\sum_jm_jp_{C,j},\quad l=\sum_jm_jv_{C,j},\quad
k_G=\sum_j\{I_j^W\omega_j+(p_{C,j}-c)\times m_jv_{C,j}\}
\]

计算目标是正确的。统一的初始 anchor yaw/xy 规范化再计算动量，也避免了“位置、
接触点、角动量来自不同坐标系”的常见错误。固定接触日程和位置后，CD dynamics
对 wrench 是仿射的；MPC 输出的 landmark 可作为 RL reward，不会改变 QP。

不严格模仿 base 的绝对全局位姿也合理：全局 root 平移/朝向可由 CoM、动量和接触
landmark 约束，而关节 position/velocity mimic 保留动作语义。这里必须报告 root
drift、heading、joint error 和 centroidal error，不能因为 reward 中不含 root 项而
回避它。BeyondMimic 已是强 motion-tracking 基线，且其工程含 motion preprocessing、
tracking reward 和 DR [代码库](https://github.com/HybridRobotics/whole_body_tracking)，因此
“使用 joint mimic”本身没有新颖性。

建议把可复现性提升为方法的一部分：每个 clip 给出质量守恒检查、
\(l-m\dot c\) 误差、reference contact label 与实际接触的 F1/时序偏差，以及在线
whole-body centroidal state 和 reference 的坐标系审计。

### 创新二：从 contact wrench 到关节可行域的线性/仿射 MPC 约束

**结论：可行，但用户当前关于“Jacobian 直接把 wrench 映射成关节速度”的表述不成立；
正确版本可保持每次 QP 凸，但它是局部、保守约束而非全局保证。**

浮基刚体动力学为

\[
M(q)\ddot q+h(q,\dot q)=S^T\tau+J_c(q)^T\lambda .\tag{1}
\]

Jacobian transpose 的虚功关系给出的是 contact wrench 对**广义力/关节力矩**的贡献，
不是 joint velocity。若冻结 \((q,\dot q)\)、接触 mode 和一个全身加速度/姿态名义轨迹，
可写成

\[
\tau_k=\bar\tau_k+T_{\lambda,k}\lambda_k,\qquad
\dot q_{k+1}=\bar{\dot q}_{k+1}+V_{\lambda,k}\lambda_k,
\qquad q_{k+1}=\bar q_{k+1}+P_{\lambda,k}\lambda_k,\tag{2}
\]

其中矩阵由 (1) 的受约束前向动力学、离散积分和冻结的 (J_c,M,h) 获得；简单的
\(-S J_c^T\lambda\) 只是在**固定 \(\ddot q\)**时的力矩近似，不能同时声称它预测
速度。于是每个 horizon node 可加入

\[
\underline\tau\le\bar\tau+T_\lambda\lambda\le\bar\tau^{+},\quad
\underline{\dot q}\le\bar{\dot q}+V_\lambda\lambda\le\bar{\dot q}^{+},\quad
\underline q\le\bar q+P_\lambda\lambda\le\bar q^{+}.\tag{3}
\]

把软 slack \(s\ge0\) 加入 (3)，并用 \(\rho\lVert s\rVert^2\) 处罚，可避免 reference
不可行时 QP 完全失效。只要 schedule、(M,J,h)、名义轨迹和坐标变换在一次 solve
内固定，(3) 是仿射不等式；加上半正定二次代价，原问题仍为凸 QP。若使用电机
torque–speed envelope，应先采用保守的多面体内近似；直接使用乘积/非线性饱和会失去
QP 性质。

这个方向有充分先例，不能把“在 QP 中有 torque/velocity constraint”当作新发现。
例如全身 QP 文献显式由 (1) 写 torque constraints，并指出 joint position/velocity
需通过状态加速度/数值积分表达 [Wang et al., 2023](https://journals.sagepub.com/doi/10.1177/02783649231198558)；
centroidal + full-kinematics planning 也已同时优化 posture、velocity、wrench 和
joint limits，但为非线性优化 [Dai et al., 2014](https://doi.org/10.1109/ICRA.2014.6907603)。
可发表的差异应是：**在 massively parallel、固定-contact CD-QP 的 wrench 决策变量上
构造/校准低开销的全身可行域 surrogate，并量化紧性、违反率与吞吐量。**

需要在论文中给出：

- 约束推导、坐标/符号和每个矩阵的计算频率；
- 线性化残差上界或至少 held-out 实测 calibration（预测可行时实际不违反的概率）；
- 关节力矩、速度、位置的最大/95% 分位违反率和 QP infeasible/slack 使用率；
- 固定 schedule、冻结线性化、无接触互补性、无全局递归可行性的明确边界。

### 创新三：DR 内的 MPC 纠偏度驱动 mimic–landmark 自适应融合与 multi-critic

**结论：动机合理，当前实现尚未满足；方法新颖性低到中等，必须靠“物理量定义、
正确的时变 return 估计和实证”获得区分度。**

对域参数 \(\theta_d\)（质量/惯量、摩擦、重力、actuator scale 等），应让 simulator
和本环境的 MPC 同时使用该环境的同一个 \(\theta_d\)。对 reference \(x^{ref}\)，定义

\[
D_t=\frac1N\sum_{i=1}^N\|W_x(x_{t+i}^{MPC}(\theta_d)-x_{t+i}^{ref})\|^2
     +\frac1N\sum_{i=0}^{N-1}\|W_u(u_{t+i}^{MPC}-u_{t+i}^{ref})\|^2,\tag{4}
\]

并使用限幅/滞回的 exogenous 系数 \(\alpha_t=\sigma(a(D_t-b))\)。
\(D_t\) 是“当前状态、参考及随机化模型下最优 CD plan 对 reference 的必要偏离”，
不是 MPC 的普遍正确性指标；它还会受权重、horizon、slack 和接触标签影响。因此应
报告其与真实 joint-limit/slip/fall 和 reference tracking error 的相关性，不能把它自动
称为“纠偏真值”。

训练上应使用

\[
r_t=\alpha_t r_t^{MPC}+(1-\alpha_t)r_t^{mimic}+r_t^{task}+r_t^{reg}.\tag{5}
\]

**不能**先对各原始长期 advantage 做完 GAE，再以当前 \(\alpha_t\) 加权；未来每步
\(\alpha_{t+k}\) 会改变 return。正确做法是将每一 transition 的带权 reward 写入
rollout，再对带权通道各自计算 GAE，或训练一个总 value。若 \(\alpha_t\) 依赖 policy
action，应 detach/停止梯度并保存其数值，避免 actor 通过操纵权重而逃避任务；更稳妥
的是让它只依赖 MPC/reference/model 状态，并加最小 mimic 权重和变化率限制。

当前 multi-critic 的四通道加和严格等于原 scalar reward，因而其 actor gradient 对**现有
固定 reward**是正确的；它并没有 adaptive fusion。多 critic 也不是新概念：
[RobotKeyframing](https://arxiv.org/abs/2407.11562) 已以分组 critic 处理 dense/sparse
reward，并以加权 advantage 做 PPO。你的差异只能是由 (4) 导出的、随物理偏离变化的
权重，以及在不引入 PPO bias 的实现与结果。

## 4. 竞争格局与 RAL 新颖性判断

最接近的竞争工作是 [HybridMimic](https://arxiv.org/abs/2603.06775)：它已经将 RL、
连续 contact state、centroidal controller 和 human motion mimicking 结合，并在 Booster
T1 硬件展示；其摘要明确声称 policy 调制连续接触状态和 centroidal velocity，及由
centroidal control 生成可行 feed-forward torque。因此，当前 repository 中的
`HybridMimicAction` 名称、contact-plan residual 和此类架构会被审稿人直接比较。

| 主张 | 新颖性判断 | 使其足够强的必要条件 |
|---|---|---|
| 全身重定向参考的精确 centroidal landmark | 中等 | 证明相较 root approximation / kinematic mimic 可显著提升高动态、跨机器人/跨质量表现 |
| 关节可行域投影到 batched CD-QP wrench | 中等偏高（若真正低成本且有校准） | 推导、凸性命题、safety calibration、与 WBC/非线性方案的质量–速度–吞吐量比较 |
| 纠偏度自适应 reward + multi-critic | 低至中等 | 正确时变 GAE、因果消融、显示固定权重和单 critic 的明显失败模式 |
| “连续接触动作调制 CD-MPC” | 低 | 已被 HybridMimic 覆盖，不应作为核心贡献 |
| “MPC 参数化/学习适应 model mismatch” | 低 | 同期已有 parameterized centroidal MPC 工作，例如 [Cost-Matching MPC](https://arxiv.org/abs/2603.28243) |

**投稿判断：以今天状态，不建议投 RAL。** 原因不是方向不对，而是创新二、三尚未实现，
且 HybridMimic 已占据相邻问题的硬件证据。若完成下节的最小闭环、在至少两种机器人
和真实硬件/高保真 sim2sim 上给出强消融，RAL 仍有合理可能；若没有关节 hard/surrogate
constraint 的证据，定位为“工程化 reward 组合”通常不够。

## 5. 代码审计：逻辑冲突、冗余与优先级

### P0：投稿前必须修正

1. **DR–MPC 不一致。** `MPCConfig` 在 `LocoMPCCommand.__init__` 一次性以标量
   mass/friction 构建并预计算矩阵；G1/JC01 mimic config 也写入常量质量。原 THEMIS
   配置有 body inertia DR，但 G1/JC01 当前只显式改足端 friction 配置，且没有把每个
   environment 的随机化惯量、摩擦、actuator 参数同步给 MPC。应建立 per-env
   `MPCModelParameters`，在 reset 后读取 simulator 实际参数并重建/批量传入 (A,B,d,G,h)。
   若不做，不能声称“DR 下为同一 reference 优化不同动力学 landmark”。
2. **创新二没有代码入口。** `g1_mpc/centroidal_mpc.py` / `jingchu01_mpc/centroidal_mpc.py`
   没有 joint affine constraint；`hybrid_mimic.py` 也未使用逐关节 `EFFORT_LIMIT`、
   `VELOCITY_LIMIT`。先为 reference MPC 新增 `JointFeasibilityLinearization`，再扩展
   PiMPC/JAX/ADMM 三个后端；否则不同 solver 的约束集会悄然不一致。
3. **自适应 multi-critic 没有实现。** 当前 `MultiCriticPPO.compute_returns()` 用固定
   `actor_advantage_weights`；必须存储 α、重建带权 reward channels，并测试其总和等于
   (5)。不要仅在 `update()` 中重乘 advantage。

### P1：实验可信度与效率

4. **接触真值与 QP schedule 要分开记录。** 目前 height/speed fallback 只是 kinematic
   heuristic；对每个 clip 比较 explicit label、heuristic 和 simulator contact sensor，
   尤其是 toe/heel/edge contact。论文中将其称为 “motion-derived nominal schedule”，而非
   “ground-truth contact”。
5. **centroidal state 重复计算。** landmark reward、`mpc_com_ref`、`mpc_ang_mom_ref`
   均可能在同一步重新遍历所有刚体。将状态按 environment step 缓存在 MPC command，
   既消除冗余又保证同一步 observation/reward 使用完全同一个值。
6. **线动量与 CoM velocity reward 重复。** 恒质量下 (l=m\dot c)。当前 landmark
   同时奖励两者；消融中应有仅速度、仅动量、二者的归一化组合，避免只是隐式加大同一
   误差的权重。
7. **复制维护风险。** G1 与 JC01 的 `multi_critic.py`、MPC MDP、solver 大量镜像复制。
   当前物理隔离是对的，但数学实现应提取至 `training_common`/`mpc_common`，机器人包只
   提供 XML、joint/contact maps 和参数。先用回归测试锁住行为，再去重；不能在没有测试
   的阶段大规模重构。

### 已通过与未覆盖的检查

- 已通过：`python -m compileall -q src`、`git diff --check`、`git fsck --no-dangling
  --no-reflogs`；工作树在审查开始时干净。
- 未覆盖：MJLab/RSL-RL 的真实环境构造、单步 rollout、三 solver 的数值等价、训练收敛。
  当前本地 Python 环境没有 `torch`、`mjlab`、`rsl_rl`、`tensordict`；`uv run` 还因全局
  cache 的只读 `.git` 失败。因此上述通过不等于任务可训练。

## 6. 推荐的最小可发表方法与实施顺序

1. **冻结一个参考任务。** 选择 3--5 个有明确接触的 clip；完成离线 centroidal/contact
   audit 和 raw reference replay。先关闭 policy contact-plan residual，得到 fixed-reference
   single-critic control。
2. **实现并校准 joint-feasibility QP。** 每个 MPC update 从 MuJoCo 获取/线性化
   (M,J,h)，在 (N=10) 上构造 (2)--(3)；先只约束 12 个腿关节 torque/velocity，带
   slack。确保 PiMPC、JAX PiMPC 和 ADMM 的相同约束结果在小 batch 上一致，再 benchmark
   4096 env 的 wall time。
3. **实现 parameter-consistent DR。** reset 采样并记录 \(\theta_d\)，既更新仿真又
   更新 MPC；把 \(\theta_d\)、QP slack、(D_t) 放进 critic only observation 和日志。
4. **实现自适应 reward 的正确 return。** 使用 (4)--(5)，冻结/滞回 \(α)，对带权通道
   单独 GAE；保留原有 exact-additive multi-critic 作为 ablation，而不是替换基线。
5. **最后才加 policy contact residual / hierarchical parameter action。** 这些是提高
   performance 的扩展，不应遮蔽“约束与自适应”主张。

## 7. RAL 所需消融与指标

最低实验矩阵应包含：

| 对照 | 回答的问题 |
|---|---|
| BeyondMimic-style joint mimic | 是否只是常规 imitation？ |
| 原始 MPC-RL landmark | exact motion reference 是否优于 velocity/root approximation？ |
| fixed-reference MPC + joint mimic | reference centroidal/contact 的纯收益 |
| + joint affine constraint（无 adaptive fusion） | 约束本身是否减少 violation 而非只降低 reward？ |
| + fixed reward fusion / single critic | multi-critic 是否仅是 value-function 复杂化？ |
| + adaptive fusion（完整方法） | (D_t) 的因果贡献 |
| HybridMimic 可复现配置或逐项公平比较 | 与最接近先例的真实差异 |

每个方法至少给出 success/fall、joint (q,\dot q,\tau) violation、slack、friction/CoP
violation、MPC feasibility、centroidal tracking、joint tracking、root drift、能耗/热代理、
MPC latency 与吞吐量；对 DR 范围内和范围外扰动分别报告均值、方差、成功率及置信区间。
硬件若只跑一台机器人，另一台至少应有独立 high-fidelity sim2sim；若声称 actuator
constraint，则硬件日志必须包含可比较的 torque/velocity/temperature 或可靠代理。

## 8. 可安全写入论文的结论与不可写的结论

现在可以写：使用模型质量分布从 retargeted reference 重建 centroidal targets；对固定
contact schedule 的 CD-MPC 产生 detached RL landmarks；现有 multi-critic 对原 reward
进行精确加性分解。

现在不应写：MPC 已显式保证关节 torque/velocity 约束；DR 参数已在 MPC 内同步；multi-
critic 已按纠偏度自适应融合；全身 recursive feasibility、chance-constrained safety 或
硬件安全性已被证明。完成 (2)--(5) 和校准实验后，前三项才可改为方法性主张；递归可行性
还需终端集、鲁棒误差界和可实现 wrench 的全身条件，不能由“QP 是凸的”推出。
