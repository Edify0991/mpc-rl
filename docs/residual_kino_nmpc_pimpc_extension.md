# Residual kino-dynamic NMPC 向 \(\pi^n\)-MPC 的迁移方案

## 1. 结论、边界与目标

本文针对将 [Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control](https://arxiv.org/html/2510.12717) 的 GPU kino-dynamic NMPC/Residual-RL 思路，迁移到本仓库的 `CentroidalMPC` / `jax_pimpc.py` 框架的方案。此处的“PiMPC”均指 \(\pi^n\)-MPC（parallel-in-horizon MPC），不是 Pinocchio；Pinocchio 可以作为模型量计算的离线/CPU 原型，但不能成为每个训练环境、每个 horizon 节点的 Python 调用后端。

**结论是有条件地正确：**在一个固定接触模态上，对非线性 kino-dynamic OCP 作一次 SQP/RTI 线性化，得到的确是局部凸 QP；在满足本文件第 4 节的结构条件后，它可以交给扩展后的 \(\pi^n\)-MPC 做 GPU 的环境批量和 horizon 节点并行。Residual MPC 本身也采用“一次 SQP 子问题”的 RTI 路径，将原始 NLP 在一次控制调用内近似为 QP [Residual MPC](https://arxiv.org/html/2510.12717)。

但下列三句话都**不正确**，不能写入论文或代码注释：

1. “任意 NMPC 线性化后与当前 `PiMPCSolver` 完全等价。”当前实现只接受公共常量 `A`、逐节点 `B_k`、公共仿射项 `e` 和逐节点控制 box；它没有时变 \(A_k,d_k\)，也没有接触速度、摩擦锥、摆脚高度、关节/力矩等一般线性不等式的投影。
2. “线性化 QP 等价于原始 NMPC。”它仅在名义轨迹邻域内是一阶一致的 SQP 子问题；接触切换、强非线性、大 residual 或差的 warm start 都会破坏这一近似。
3. “horizon 可以被拆成彼此独立的 batch。”动力学约束仍将 \(k\) 与 \(k+1\) 耦合。正确表述是：经 \(\pi^n\)-MPC/ADMM 变量分裂后，局部矩阵运算、投影和 dual 更新在张量维度 \((B_{\rm env},N)\) 上并行执行；时间耦合由有限次并行迭代中的代数更新处理，而非逐时刻独立求解。

本次建议的终态是一个**单独注册的新任务**：Residual kino-NMPC 直接生成全身动力学一致的名义关节力矩，RL 只输出有界 residual；它与现有“质心 QP 只产生 landmark、绝不注入 \(J^\top f\)”的 mimic 任务并存，不能静默替换后者。

---

## 2. 原始非线性 kino-dynamic OCP

令 \(q\in\mathcal Q\) 为浮基广义构型，\(v\in\mathbb R^{n_v}\) 为广义速度，\(a\in\mathbb R^{n_v}\) 为广义加速度，\(\lambda\in\mathbb R^{n_\lambda}\) 为全部候选接触力/力矩。使用局部流形增量 \(q\boxplus\delta q\)，而不是直接相加 quaternion。对 batch 中第 \(b\) 个环境和 horizon 节点 \(k\)，定义

\[
x_k=(q_k,v_k),\qquad u_k=(a_k,\lambda_k).
\]

浮基投影动力学为

\[
\Psi(q_k,v_k,a_k,\lambda_k)
=M_b(q_k)a_k+h_b(q_k,v_k)-J_b(q_k)^\top\lambda_k=0. \tag{1}
\]

其中 \(M_b,h_b,J_b\) 是浮基六维方程的相应块；这是完整刚体动力学
\(M(q)a+h(q,v)=S^\top\tau+J(q)^\top\lambda\)
消去关节驱动力矩后的基座方程。离散化可采用半隐式 Euler（生产版本建议与 Residual MPC 对齐并进行步长/积分器消融）

\[
v_{k+1}=v_k+\Delta t\,a_k,\qquad
q_{k+1}=q_k\boxplus(\Delta t\,v_{k+1}). \tag{2}
\]

给定本次 MPC 调用前冻结的接触激活 \(\sigma_{k,j}\in\{0,1\}\)，考虑目标

\[
\begin{aligned}
\min_{x,u}\quad &\sum_{k=0}^{N-1}
 \tfrac12\|r_k(x_k,u_k;x_k^{\rm ref})\|_{W_k}^2
 +\tfrac12\|u_k-u_{k-1}\|_{R_\Delta}^2
 +\tfrac12\|r_N(x_N;x_N^{\rm ref})\|_{W_N}^2,\\
\text{s.t.}\quad& (1),(2),\quad x_0=\hat x_0,\\
& \sigma_{k,j}J_j(q_k)v_k=0, \tag{3a}\\
& F_j\lambda_{k,j}\le b_j\sigma_{k,j},\quad \lambda_{k,j}=0\ \text{if}\ \sigma_{k,j}=0, \tag{3b}\\
& r_{j,z}(q_k)=h_{j}^{\rm swing}\ \text{for swing},\quad
 q_{\min}\le q_k\le q_{\max},\ v_{\min}\le v_k\le v_{\max},\\
& \tau_{\min}\le\tau_{\rm ID}(q_k,v_k,a_k,\lambda_k)\le\tau_{\max}. \tag{3c}
\end{aligned}
\]

\(F_j\lambda\le b_j\) 必须使用摩擦**多面体**（而非精确二阶锥）才会在 QP 内保持线性；足端/手端接触坐标、法向和摩擦系数同样在一次 RTI 子问题中冻结。若令 \(\sigma\)、接触位置或 complementarity \(\lambda_n\phi(q)=0\) 成为决策变量，问题重新成为混合整数/双线性非凸问题，不能称为当前 QP。

Residual MPC 所使用的 kino-dynamic 形式同样含基座动力学、接触速度、摩擦、摆脚高度和关节状态边界；其公开 formulation 采用固定接触相位并在每次调用仅解一个 SQP QP [Residual MPC](https://arxiv.org/html/2510.12717)。

---

## 3. 从 NMPC 到 RTI-QP 的严格推导

取上一控制周期 warm start \(\bar z=\{\bar x_k,\bar u_k\}\)，其中
\(\delta x_k=(\delta q_k,\delta v_k)\)、\(\delta u_k=(\delta a_k,\delta\lambda_k)\)。对由 (1)、(2) 定义的离散映射 \(x_{k+1}=f_k(x_k,u_k)\)，在局部坐标中一阶展开：

\[
\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k+d_k,\quad
d_k=f_k(\bar x_k,\bar u_k)-\bar x_{k+1}. \tag{4}
\]

\(A_k=\partial f_k/\partial x\)、\(B_k=\partial f_k/\partial u\) 含 \(M_b,J_b,h_b\) 对 \(q,v\) 的导数。它们可由 GPU 上的解析/自动微分 kernel、codegen 或稀疏数值线性化给出；不可在 IsaacLab 训练环中逐环境调用 CPU Pinocchio。

写全部等式/不等式为 \(c_k(z_k)=0\)、\(g_k(z_k)\le0\)，其可行域一阶近似为

\[
E_k\delta z_k=e_k,\qquad C_k\delta z_k\le r_k, \tag{5}
\]

其中 \(\delta z_k=(\delta x_k,\delta u_k)\)，
\(E_k=\nabla c_k(\bar z_k)\)、\(e_k=-c_k(\bar z_k)\)，
\(C_k=\nabla g_k(\bar z_k)\)、\(r_k=-g_k(\bar z_k)\)。对最小二乘残差采用 Gauss--Newton（GN）近似：

\[
H_k=J_{r,k}^\top W_kJ_{r,k}+\rho_{\rm reg}I\succeq\rho_{\rm reg}I,\qquad
g_k=J_{r,k}^\top W_kr_k(\bar z_k). \tag{6}
\]

因此单次 RTI 子问题为

\[
\begin{aligned}
\min_{\delta x,\delta u}\ &
\sum_{k=0}^{N-1}\bigl(\tfrac12\delta z_k^\top H_k\delta z_k+g_k^\top\delta z_k\bigr)
+\tfrac12\delta x_N^\top H_N\delta x_N+g_N^\top\delta x_N,\\
\text{s.t.}\ &\delta x_0=0,\ (4),\ (5). \tag{QP-RTI}
\end{aligned}
\]

控制增量平滑项 \(\|u_k-u_{k-1}\|_{R_\Delta}^2\) 只在相邻阶段产生带状耦合；通过将前一控制并入增广状态或通过 \(\pi^n\) 的共识变量分裂，仍可保留阶段并行结构。

### 命题 1：局部 QP 的一阶一致性

假设 \(f,c,g,r\) 在 \(\bar z\) 邻域二阶连续，且姿态使用有效局部 retraction。则 (4)--(6) 的约束残差和 GN 梯度残差分别与原 NLP 相差 \(O(\|\delta z\|^2)\)。若 \(W_k\succeq0\)、\(\rho_{\rm reg}>0\)，则 (QP-RTI) 的目标严格凸于未被等式消去的方向。

**证明。**对每个 \(C^2\) 映射作 Taylor 展开，余项由二阶导数有界性为 \(O(\|\delta z\|^2)\)，得到 (4)、(5)。最小二乘项的精确 Hessian 等于 \(J_r^\top WJ_r+\sum_i r_i\nabla^2r_i\)；GN 去掉后项，因此保留半正定项，再加 \(\rho_{\rm reg}I\) 即严格正定。注意：这证明的是局部模型一致性和 QP 凸性，并非原非凸 NLP 的全局凸性或全局最优性。\(\square\)

### 命题 2：一次 RTI 的适用条件

若最优解邻域内满足 LICQ、二阶充分条件、严格互补，并且参考/状态在相邻控制周期的变化有界，上一周期解可作为 warm start，则一次 SQP/RTI QP 给出对移动最优解的局部跟踪修正。若活动集切换、接触模式改变、残差超过 trust region，或 QP 不可行，则不存在该局部保证。

**说明。**这是 RTI 的标准局部结论，不能被扩大为“RL residual 下的全局稳定性”。工程上必须保留 line-search/trust-region、上次可行解/安全 PD fallback、QP residual 和约束违反监控。

---

## 4. 何时可进入 \(\pi^n\)-MPC，何时不行

### 4.1 需要的可分结构

对每个环境，(QP-RTI) 必须改写为

\[
\begin{aligned}
\min_{\delta x,\delta u}\;& \sum_{k=0}^{N-1} f_k(\delta x_k,\delta u_k)
 +f_N(\delta x_N),\\
\text{s.t.}\;&\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k+d_k,\\
& (\delta x_k,\delta u_k)\in\mathcal Z_k,
\end{aligned}\tag{7}
\]

其中每个 \(\mathcal Z_k=\{z:E_kz=e_k,C_kz\le r_k\}\) 是非空闭凸多面体；所有维度和约束行数按最大接触数静态 padding，inactive row 用 mask 放宽。用本地副本 \(w_k,z_k\) 和共识约束 \(w_k=z_k\) 分裂后，增广拉格朗日的 Hessian 在 \(k\) 上为 block diagonal。故以下操作可对全部 \((b,k)\) 同时执行：

- 每阶段 GN/Riccati/增广矩阵块的组装与小矩阵分解；
- \(\Pi_{\mathcal Z_k}\) 的 polyhedral projection；
- primal/dual 的 ADMM 更新、残差归约和固定次数迭代。

动态链只通过 (7) 的相邻项出现在等式更新中。\(\pi^n\)-MPC 的意义是把这条链改写为同步的矩阵操作；它不是把 \(N\) 个节点各自当作独立控制问题。

### 4.2 当前实现与目标实现的差距

| 项目 | 当前 `jax_pimpc.py` | kino-RTI 所需 |
|---|---|---|
| 动力学 | 常量 \(A\)，batch/阶段 \(B_k\)，全 horizon 共用 \(e\) | batch/阶段 \(A_{b,k},B_{b,k},d_{b,k}\) |
| 代价 | 固定 \(W_y,W_u,W_f\)，控制差分 | 阶段 GN \(H_{b,k},g_{b,k}\)，终端项，仍可保留差分 |
| 可行集 | 控制变量逐节点 box clip | 一般线性等式与不等式的阶段投影/分裂 |
| 接触 | centroidal wrench mask | 固定 mode 下的摩擦 pyramid、接触速度、摆脚、力矩/状态约束的线性化 |
| 状态 | 9D \([c,l,k]\) | 局部 \([q,v]\)，不可直接对 quaternion 相加 |
| GPU | JAX AOT，\((B,N)\) 张量算子 | 同样静态 shape 的 JAX/XLA 或 CUDA kernel；模型/导数也必须在设备端 |

因此第一阶段不能把 `CentroidalMPC._solve_jax_pimpc` 的输入直接替换为全身 NMPC；必须新增一个不破坏旧 API 的一般 LTV-RTI solver。

### 4.3 ADMM 收敛的准确表述

对于固定线性化、\(H\succeq0\)、\(\mathcal Z_k\) 闭凸且存在鞍点的 QP，精确子问题解的标准两块 ADMM 收敛到 QP 的 primal-dual 解。实际训练为固定迭代次数、warm start 和近似投影，故只能报告 KKT/primal/dual residual，不能声称每步精确收敛。若希望保持该理论，应将一般约束设计为一个凸阶段投影，不要在 ADMM 内引入未冻结的接触二值变量或非凸足端距离约束。

---

## 5. residual 与原仓库 mimic 框架的关系

新控制律建议为

\[
\begin{aligned}
\tau_{\rm kino}&=\tau_{\rm ID}(q_0^{\star},v_0^{\star},a_0^{\star},\lambda_0^{\star})
+K_p(q_0^{\star}-q)+K_d(v_0^{\star}-v),\\
\tau_{\rm cmd}&=\Pi_{[\tau_{\min},\tau_{\max}]}
\left(\tau_{\rm kino}+\alpha\,\tau_{\rm res}^{\rm RL}\right). \tag{8}
\end{aligned}
\]

\(\tau_{\rm ID}\) 应由完整逆动力学/RNEA 或等价的致动关节求解得到。它不是把 centroidal 力粗略映射为 \(J^\top f\) 的前馈项；后者正是当前 landmark 任务刻意没有采用的近似。Residual MPC 也将模型控制力矩与 RL residual 混合，而用 GPU 并行 MPC 保持训练吞吐 [Residual MPC](https://arxiv.org/html/2510.12717)。

建议 residual actor 输出受限的关节 residual（可为 \(\Delta q,\Delta\dot q\) 经 PD 转力矩，或直接 tanh-squashed \(\Delta\tau\)），并采用 \(\alpha\) warm-up/anneal、\(\|\tau_{\rm res}\|^2\) 和 saturation penalty。MPC 仍以 centroidal/全身参考优化；mimic 奖励仍施加在仿真的全身姿态/速度上。

**重要安全界：**(8) 的力矩 clip 仅保证力矩 box，不保证计划接触力、摩擦或未来状态约束仍成立。因此 residual 对 NMPC 的闭环可行性没有无条件保证。若需要安全论断，至少要有：收紧后的接触/状态约束、模型误差界、可实现 wrench/torque 集、终端不变集或 backup controller，并使 residual 落入经过鲁棒不变集推导的允许集合。当前代码和本方案第一版都不能声称该强结论。

---

## 6. 推荐代码改造方案

### 6.1 新模块，不回归现有 CD-MPC

保持下列旧路径不变：`src/themis_mpc/centroidal_mpc.py`、`jax_pimpc.PiMPCSolver`、现有 `LocoMPCCommand` 与 `HybridMimicAction`。它们继续执行固定 \(dt\) 的 centroidal QP，只输出 landmark，不注入力矩。

新增：

1. `src/themis_mpc/kino_nmpc.py`
   - `KinoNMPCConfig`、`KinoNMPCInput`、`KinoNMPCOutput`；
   - `KinoRTIPiMPC.solve()`：shift warm start → GPU model evaluation/linearization → 一次或受限次数 SQP → recover nominal \((q,v,a,\lambda,\tau)\)；
   - 成功标志、KKT/constraint residual、求解耗时、fallback 标志必须随输出保存。
2. `src/themis_mpc/kino_dynamics_backend.py`
   - 明确接口 `evaluate_and_linearize(q,v,a,lambda,contact_data)`；输出 \(M_b,h_b,J_b\)、kinematics、导数/线性化块与 \(\tau_{\rm ID}\)；
   - 先以小 batch 的 Pinocchio 数值差分作**离线验证 oracle**，生产训练换成 JAX/Warp/CUDA 可 batch 的解析或 AD kernel。禁止 Python for-loop 的 per-env Pinocchio。
3. `src/themis_mpc/jax_rti_pimpc.py`
   - 独立于旧 `jax_pimpc.py` 的 `RTIPiMPCSolver`；静态输入 shape：
     \[
     A:[B,N,n_x,n_x],\ B:[B,N,n_x,n_u],\ d:[B,N,n_x],
     \ H/g/C/E:[B,N,\ldots].
     \]
   - 使用 `jax.jit`、`lax.fori_loop` 和固定 `N,n_c` AOT compile；cache key 至少包含 \((B,N,n_x,n_u,n_{ineq},n_{eq},dtype)\)。
4. `src/themis_mpc/stage_projection.py`
   - 实现 \(\Pi_{\mathcal Z_k}\)。第一版可使用固定步数的 batched dual projected-gradient / Dykstra / 小型 ADMM；所有约束必须是线性凸的，且输出 projection residual。
5. `src/themis_training/kino_mpc_mdp.py`
   - 读取 articulation state、参考全身轨迹和冻结接触 schedule，维护 solver/warm start/landmarks；
   - 不能复用只含 \([c,l,k]\) 的 `LocoMPCCommand` buffer。
6. `src/themis_training/residual_kino_action.py`
   - 按 (8) 组合 kino-MPC torque、RL residual 和 actuator limit；
   - 记录 `tau_kino`、`tau_residual`、clip fraction、solver-failure flag，供 critic/reward/日志使用。
7. 新环境配置与注册（例如 Jingchu01 首先落地）：
   `Mjlab-Residual-KinoNMPC-Mimic-Jingchu01-28DOF`。原任务 ID 和 checkpoint 绝不改变。

### 6.2 约束投影的实现策略

为保留阶段可并行，避免先形成整个 horizon 的巨大稠密 KKT。将每步不等式打包为
\(C_{b,k}z_{b,k}\le r_{b,k}\)，采用静态最大行数并以 inactive-row mask 实现。摩擦使用四/八棱锥；接触速度、摆脚高度等线性化等式进入 \(E\)。

若某些约束需要跨时刻（例如 footstep 连续性、力率），应：

- 将一阶差分增广到状态，或
- 引入相邻副本和额外 consensus edge。

不得为了方便把这些约束悄悄忽略；这样虽然仍“并行”，但与 kino-dynamic NMPC 已不是同一个问题。

### 6.3 内存与吞吐

全身变量下，直接物化稠密 \(H\in\mathbb R^{B\times N\times n_z\times n_z}\) 会迅速耗尽 GPU 显存。应优先存储 block/对角权重、Jacobian block 或定义 Hessian-vector product；只对每阶段小块分解。常量/稀疏 pattern AOT 缓存，数值仅更新。基准必须分离首次编译时间和 steady-state solve 时间，并扫描 \(B\in\{256,1024,4096\}\)、\(N\in\{8,12,16,24\}\)。

---

## 7. 分阶段实施与验收

### 阶段 A：一般 LTV \(\pi^n\)-QP（不接全身模型）

- 扩展 solver 支持 \(A_{b,k},B_{b,k},d_{b,k}\)；
- 退化测试：令 \(A_{b,k}=A\)、\(d_{b,k}=e\)、仅控制 box 时，输出与当前 `jax_pimpc.solve_batch` 在容差内一致；
- 用小随机 QP 与 CPU 稀疏 OSQP/高精度解比较 objective、primal/dual residual。

### 阶段 B：kino-dynamic RTI QP

- 用单环境 Pinocchio oracle 验证 \(A,B,d,C,E\) 的有限差分误差；验证 quaternion tangent/retraction；
- 固定参考接触 schedule，先完成基座动力学、摩擦 pyramid、接触速度、状态界和关节 torque recovery；
- 加 trust region \(\|\delta z_k\|_\infty\le\Delta\) 与失败 fallback，确认 RTI 不是在大修正时发散。

### 阶段 C：GPU 与 Residual-RL 集成

- 将模型和线性化迁到 GPU 可 batch backend；验证任何 host synchronization 都不在 rollout 热路径；
- 注册独立 residual-kino task；先固定 \(\alpha=0\) 验证纯 MPC，再逐步 anneal residual；
- 奖励至少包含 mimic、参考/landmark 跟踪、residual energy、torque saturation、MPC constraint/KKT residual；不得把 solver failure 当作无信号样本。

### 阶段 D：科学评估

报告并作消融：

1. centroidal-landmark MPC-RL（当前基线）；
2. kino RTI MPC，无 residual；
3. kino RTI + residual；
4. 无 warm start、无 trust region、无接触速度约束；
5. 不同 \(B,N\) 的延迟/吞吐与 KKT residual。

指标包括全身 mimic error、CoM/动量误差、接触滑移/摩擦违反、关节 torque saturation、fall rate、QP/NMPC failure rate、单控制周期 GPU 时间。只有当第 2--5 项均满足约束残差和实时预算时，才可声称该迁移有效。

---

## 8. 可写入论文的理论表述

可严谨声明：

> 在固定接触模式、有效局部构型坐标、\(C^2\) 动力学和局部正则条件下，所提出的 RTI-SQP 将 kino-dynamic NMPC 转化为带时变线性动力学和阶段凸约束的 GN-QP。经共识分裂，QP 的阶段局部更新和投影可在环境 batch 与预测节点上并行执行；该并行实现保留原动力学链约束，并以有限迭代近似求解该局部 QP。

不可声明：全局最优 NMPC、任意接触切换下递归可行、Residual-RL 下的无条件闭环稳定、或“horizon 节点独立”。这些结论需要另外的混合接触规划、鲁棒管/概率约束、终端集与 residual 允许集证明。

## 9. 参考资料

1. Jeon et al., [Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control](https://arxiv.org/html/2510.12717), 2025. 该文给出 kino-dynamic 约束、一次 SQP/RTI 和 GPU 批量求解/残差控制的直接参考。
2. 本仓库 `src/themis_mpc/centroidal_mpc.py` 与 `src/themis_mpc/jax_pimpc.py`：本文所列“当前实现能力/缺口”以实际 API 和张量 shape 为准。
3. 本仓库 `docs/stochastic_contact_mpc_and_joint_mimic.md`：固定接触参数下 CD-MPC 条件凸性、随机接触与 safety claim 的既有边界。
