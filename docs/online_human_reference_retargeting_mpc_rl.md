# 将 GMR 类运动学重定向融入 MPC-RL：直接人体参考输入的可行性、理论与实施方案

## 1. 结论

可以将 GMR（General Motion Retargeting）一类运动学重定向**放入** MPC-RL 框架，从而不再预先为每段人体动作生成、保存并训练于一个机器人关节轨迹 NPZ；但不能因此说“无需重定向”。人和机器人的拓扑、杆长、关节类型、限位、质量和接触可行域不同，任何从人体 motion 到机器人控制量的映射仍然是重定向/形态投影。

有三种层次完全不同的做法：

| 做法 | 是否取消离线 NPZ | 是否真正联合动力学 | 研究价值 |
|---|---:|---:|---|
| A. 运行时调用 GMR，再喂给现有 tracker/CD-MPC | 是 | 否 | 工程便利，创新性弱 |
| B. GPU/在线 IK 产生 \(q^{\rm ik}\)，再喂给现有 CD-MPC landmark | 是 | 否；IK 与 MPC 串联 | 可作为 baseline |
| C. **人体 task-space residual 直接进入 kino-dynamic RTI-MPC**，MPC 同时求机器人 \((q,v,a,\lambda,\tau)\) | 是 | 是 | 推荐的研究主线 |

推荐 C，名称可为 **Online Human-Reference Kinodynamic Projection (OHRKP)**。它保留 GMR 的“关键身体对应、rest-pose 对齐、非均匀局部缩放”作为**静态形态标定**，但不再先求完整的 \(q^{\rm GMR}\) 轨迹；每个控制周期以当前机器人状态、未来人体任务空间参考、接触先验和机器人动力学共同求一个有限时域的可行机器人轨迹。其第一个控制量用于执行，整条预测轨迹产生 CoM、动量、接触力/力矩 landmark 来指导 residual RL。

这比“在线运行 GMR”更有实质，但创新需要谨慎定位。GMR 已表明高质量运动学重定向会显著影响 BeyondMimic 的下游可跟踪性，并使用局部非均匀缩放和两阶段 IK [GMR 论文](https://arxiv.org/abs/2510.02252)。近期 KDMR 已将重定向表述为离线、多接触、全身轨迹优化并施加动力学/接触约束，然后再用 BeyondMimic 训练 [KDMR 论文](https://arxiv.org/abs/2603.09956)。因此本工作不能把“kinodynamic retargeting”本身当作新贡献；可区分的贡献必须是：

1. **在线、闭环、状态条件化**：投影从实测 \(x_t\) 而非离线静止初值起步；外扰、初始偏差和形态可达性会即时改变最优机器人参考。
2. **训练时 MPC landmark + residual RL 协作**：MPC 的预测 \((c,l,k,\lambda,\tau)\) 不是离线标签，而是每个 rollout 随状态更新的 privileged landmark；RL 学习执行和未建模扰动修正。
3. **同一 RTI-SQP/PiMPC 计算图内的任务空间投影**：在不将人体 source 先压成固定机器人关节轨迹的前提下，将 task-space imitation、接触候选和动力学可行性放入同一局部 QP，并保留环境 batch 与 horizon 节点并行。
4. **因果模式**：离线数据可使用 \(H\) 帧 source preview；遥操作只有当前/下一帧时，通过速度外推与不确定度/权重衰减构造未来参考，而不假装拥有未来动作。

---

## 2. 为什么当前 CD-MPC 不能直接吃人体动作

当前 `MotionReferenceCommand` 的输入 contract 是已经和机器人对齐的 BeyondMimic NPZ：`joint_names`、`body_names`、`joint_pos`、`joint_vel` 和 robot link trajectory 必须匹配仿真 articulation。`mimic_mdp.py` 会在 joint/body 名称不匹配时显式报错；`reference_centroidal.py` 也以**机器人** link 的质量、质心偏移和惯量计算

\[
c=\frac{1}{m}\sum_i m_i c_i,\qquad
l=m\dot c,\qquad
k=\sum_i\left[I_i^W\omega_i+(c_i-c)\times m_i\dot c_i\right]. \tag{1}
\]

因此，不能把人类 SMPL 关节位置直接当为当前 `CentroidalMPC` 的 \((c,l,k)\) reference：人的质量分布和接触位置不是机器人的，且人类 SO(3) joint 与机器人转动关节并不一一对应。仅把 human CoM 按身高缩放也不足以得到机器人角动量、可实现 wrench 或关节限位一致的姿态。

现有 centroidal QP 可继续作为方案 A/B 的后端，但方案 C 必须依赖上一份设计文档中的 kino-dynamic RTI-MPC：`docs/residual_kino_nmpc_pimpc_extension.md`。它从机器人 \((q,v)\) 和刚体模型直接产生机器人一致的 CoM/动量/力矩，避免把人体 centroidal 量误当成机器人参考。

---

## 3. GMR 保留什么，替换什么

令源人体在时刻 \(s\) 的观测为

\[
y_s^H=\{p_i^H(s),R_i^H(s),\dot p_i^H(s),\omega_i^H(s)\}_{i\in\mathcal I_H}, \tag{2}
\]

它可来自 SMPL/SMPL-X、BVH、人体 MoCap 或视觉恢复；不要求 human joint axis 与机器人一致。离线一次性定义静态标定

\[
\mathcal M=\{(i,j,w_{p,ij},w_{R,ij},s_i,o_i)\}, \tag{3}
\]

其中 \(i\) 是人体 key body，\(j\) 是机器人 link，\(s_i\) 是非均匀局部缩放，\(o_i\) 是 rest-pose 位置/朝向偏置。这是每个“人骨架/机器人 URDF 配对”的配置，不是每条动作的 retargeted joint trajectory。GMR 的关键 body 匹配、rest-pose 对齐、局部缩放、两阶段 IK 与该静态层完全兼容 [GMR 论文](https://arxiv.org/abs/2510.02252)。

给定在线世界对齐 \(T_t^{WH}\)（由当前机器人 base yaw/位置和 source root 建立，通常仅对齐 xy/yaw），生成机器人任务空间目标

\[
\hat p_{i,k}=T_t^{WH}\!\left[p^H_i(t+k)-p^H_{\rm root}(t+k)\right]s_i+
T_t^{WH}p^H_{\rm root}(t+k)+o_i,\qquad
\hat R_{i,k}=R_{\rm align}R_i^H(t+k)R_{i,0}^{\rm offset}. \tag{4}
\]

实际应使用每个 link 的 3D scale matrix \(S_i\) 或沿骨段方向的标量，而不是含义含混的右乘标量；(4) 只是简洁记号。root translation 必须使用一致的全局比例，否则会制造脚滑。GMR 也特别强调该点。

**被替换的部分**是每帧先求

\[
q_k^{\rm GMR}=\arg\min_{q\in[q^-,q^+]}
\sum_{(i,j)\in\mathcal M}\!
w_{p,ij}\|p_j(q)-\hat p_{i,k}\|^2+
w_{R,ij}\|\log(\hat R_{i,k}^{\top}R_j(q))\|^2, \tag{5}
\]

然后将 \(q_k^{\rm GMR}\) 写入 NPZ、再训练 tracker 的流程。式 (5) 是 GMR 类逐帧运动学 IK 的抽象。它可以在线运行，但不含动力学、力矩、接触反力或实际状态；因而“在线 GMR + 原框架”只是把离线步骤移入 runtime。

---

## 4. 推荐：人体任务空间直接进入 kino-dynamic MPC

### 4.1 联合最优控制问题

以机器人状态 \(x_k=(q_k,v_k)\)、控制 \(u_k=(a_k,\lambda_k)\) 为决策，其中 \(\lambda\) 是固定接触模式下的候选接触 wrench。定义人体--机器人几何残差

\[
e_{ij}(q_k,y_{t+k}^H)=
\begin{bmatrix}
p_j(q_k)-\hat p_{i,k}\\
\log\!\left(\hat R_{i,k}^{\top}R_j(q_k)\right)
\end{bmatrix}. \tag{6}
\]

在线参考投影问题为

\[
\begin{aligned}
\min_{x,u}\quad
&\sum_{k=0}^{N-1}\Big[
 \sum_{(i,j)\in\mathcal M}\|e_{ij}(q_k,y_{t+k}^H)\|_{W_{ij,k}}^2
 +\|v_k-\hat v_k^H\|_{W_v}^2
 +\|u_k\|_R^2+\|u_k-u_{k-1}\|_{R_\Delta}^2\\
&\hspace{23mm}+\|c(q_k)-c_k^{\rm style}\|_{Q_c}^2
 +\|h(q_k,v_k)-h_k^{\rm style}\|_{Q_h}^2\Big]
+V_N(x_N,y_{t+N}^H),\\
\text{s.t.}\quad
&M(q_k)a_k+h_{\rm rb}(q_k,v_k)=S^\top\tau_k+J(q_k)^\top\lambda_k, \tag{7a}\\
&x_{k+1}=f_{\Delta t}(x_k,a_k),\quad x_0=x_t^{\rm measured}, \tag{7b}\\
&\tau^-\le \tau_k\le\tau^+,\quad q^-\le q_k\le q^+,\quad v^-\le v_k\le v^+, \tag{7c}\\
&\lambda_k\in\mathcal W(\sigma_k),\quad J_{\sigma_k}(q_k)v_k=0,\quad
\text{swing/obstacle constraints}. \tag{7d}
\end{aligned}
\]

\(h_{\rm rb}\) 是偏置项，\(h(q,v)\) 是关于系统 CoM 的角动量，二者符号不同。可先不加 \(c^{\rm style},h^{\rm style}\)；它们应来自经过质量/尺度归一化的人体 style descriptor，不能直接拿 human momentum 作为机器人硬 reference。更安全的第一版仅用 (6)、关节/控制正则和机器人动力学，MPC 自己产生 \((c,l,k,\lambda,\tau)\) landmark。

式 (7) 的可行集完全由机器人模型定义；人体 source 只以软任务代价出现。因此当人的手长、关节范围或某个接触无法被机器人复现时，优化器可以增加 style error，而不会被迫产生无物理意义的 joint reference。这正是它相对“GMR reference + policy 纠错”的核心价值。

### 4.2 与 GMR、KDMR 的数学关系

**命题 1（GMR 是联合问题的退化情形）。**若移除 (7a)--(7d) 中动力学、接触和力矩约束，仅保留每个 \(k\) 的 joint box，令 \(W_v=R=R_\Delta=Q_c=Q_h=0\)，则 (7) 按时刻可分，并退化为 (5) 的加权几何 IK。

**证明。**此时优化变量仅为相互无耦合的 \(q_k\)，目标为 \(\sum_k\sum_{ij}\|e_{ij}(q_k)\|^2\)，可逐 \(k\) 独立最小化，正是 (5)。\(\square\)

**命题 2（先 IK 后 MPC 一般不等价于联合投影）。**令 \(q^{\rm ik}\) 是 (5) 的解，\(\mathcal F\) 是包含动力学、接触和力矩约束的轨迹可行集。串联流程求

\[
z_A=\arg\min_{z\in\mathcal F}\|q(z)-q^{\rm ik}\|_Q^2+\mathcal R(z),\tag{8}
\]

联合方法求

\[
z_C=\arg\min_{z\in\mathcal F}\sum_{ij}\|e_{ij}(q(z),y^H)\|_{W_{ij}}^2+\mathcal R(z).\tag{9}
\]

一般有 \(z_A\ne z_C\)。

**证明。**(8) 与 (9) 的梯度分别含 \(J_q^\top Q(q-q^{\rm ik})\) 和 \(J_e^\top We(q,y^H)\)。只有在 \(q^{\rm ik}\in\mathcal F\)，且在 \(\mathcal F\) 上两种二次度量诱导同一目标梯度（极强条件）时二者才可能相同。常见情况下 \(q^{\rm ik}\) 不满足 torque/contact/dynamics，故先投影会改变 task-space 优先级；联合求解严格保留原人类任务空间误差。\(\square\)

KDMR 已经覆盖“离线人类 marker/contact → 动力学可行机器人轨迹”的问题，且以 GMR 为基线。因此本工作应明确对比 KDMR：式 (7) 不是离线长轨迹 NLP，而是从 \(x_t^{\rm measured}\) 出发的 receding-horizon RTI，且其 prediction 是 residual-RL 的在线 landmark。KDMR 的全局平滑/离线接触优化可能更优；本方法的优势是对扰动、初值和在线源输入的闭环适应，不应在离线 reference fidelity 上承诺超过其全局优化。

---

## 5. RTI-SQP 后仍能用 \(\pi^n\)-MPC 吗？

可以。取 warm start \(\bar q_k,\bar v_k,\bar u_k\)，在机器人局部构型坐标中令 \(q_k=\bar q_k\boxplus\delta q_k\)。对 (6) 线性化：

\[
e_{ij}(\bar q_k\boxplus\delta q_k,y^H)
=\bar e_{ij,k}+J_{ij,k}\delta q_k+O(\|\delta q_k\|^2). \tag{10}
\]

于是人体几何项的 GN QP 贡献为

\[
\tfrac12\delta q_k^\top
\underbrace{\left(2\sum_{ij}J_{ij,k}^\top W_{ij,k}J_{ij,k}\right)}_{H^{\rm human}_k\succeq0}
\delta q_k
+\underbrace{\left(2\sum_{ij}J_{ij,k}^\top W_{ij,k}\bar e_{ij,k}\right)^\top}_{(g^{\rm human}_k)^\top}
\delta q_k. \tag{11}
\]

将 (7a)--(7d) 同样在 \(\bar z\) 处一阶化，就得到

\[
\begin{aligned}
\min_{\delta x,\delta u}\;&\sum_{k=0}^{N-1}
 \tfrac12\delta z_k^\top H_k\delta z_k+g_k^\top\delta z_k,\\
\text{s.t.}\;&\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k+d_k,\\
&E_k\delta z_k=e_k,\qquad C_k\delta z_k\le r_k,\qquad \delta x_0=0. \tag{12}
\end{aligned}

若 \(W\succeq0\) 且加 \(\rho I\) regularization，\(H_k\succeq\rho I\)。接触 mode、人体映射 \(\mathcal M\)、缩放 \(S_i\)、权重和几何目标在每次 RTI/QP 前冻结时，(12) 是凸 QP。它与 `docs/residual_kino_nmpc_pimpc_extension.md` 第 4 节的 LTV staged QP 一致；ADMM/共识分裂后的局部组装、阶段多面体投影与 dual 更新可在 \((B_{\rm env},N)\) 上并行。

**关键限制：**不能把 \(S_i\)、body correspondence、binary contact 或人体--机器人时间扭曲同时作为 QP 决策变量。例如 \(S_i p_i^H\) 在 \(S_i\) 与其他可学习变量共同优化时可引入双线性；接触互补也非凸。第一版中它们必须是静态配置或每个 MPC 调用前的 detached/frozen 参数。若后续学习它们，采用慢尺度网络输出并在调用前冻结，或采用 trust-region SCP，而不能宣称单次 QP。

---

## 6. 接触、未来信息与遥操作

### 6.1 人体接触不是机器人接触

从 human foot 的高度/速度得到的 \(\hat\sigma_{k,j}^H\) 只是 source contact prior，不是机器人接触真值；经过局部缩放后仍可能因腿长、可达性、障碍物或扰动而失配。建议沿用当前研究计划的分层来源：

\[
\sigma_{k,j}^{\rm MPC}=\operatorname{Freeze}\left(
\operatorname{gate}(\hat\sigma^H_{k,j},\; a^{\rm contact}_{t,1:N,j},\; \hat x_t)
\right). \tag{13}
\]

其中：

- \(\hat\sigma^H\)：人类运动学高度/速度候选，经 hysteresis 平滑；若有鞋底/GRF 数据，则作为更可靠 prior；
- \(a^{\rm contact}\)：现有快尺度 RL 输出的未来 horizon contact plan；
- `Freeze`：在本次 QP 前固定二值/连续 mask，保持 QP 凸性；
- 实际仿真接触仅进入 reward、观测和下一次 warm start，不能倒灌为当前 QP 的未知 binary 变量。

对 heel/toe/edge 接触，应采用 point/patch contact，而非“整脚二值接触”。这是与当前 contact-patch 研究计划一致的可扩展点；每个 patch 产生独立 \(\lambda_{j,p}\) 和摩擦多面体。KDMR 已强调人类 heel-to-toe 接触的时变特性，但它依赖同步 GRF；无 GRF 的视觉/SMPL 输入不能声称得到真实的地反力。

### 6.2 离线动作与因果遥操作

离线 dataset 训练时可读取 \(y^H_{t:t+N}\)，这是合法的 privileged preview。部署/遥操作通常只有 \(y^H_t\) 或下一帧，必须切换为因果 reference predictor：

\[
\hat p^H_{t+k}=p^H_t+k\Delta t\,\dot p^H_t,\qquad
\hat R^H_{t+k}=R^H_t\exp(k\Delta t[\omega^H_t]_\times), \tag{14}
\]

并使用随预测距离衰减的权重

\[
W_{ij,k}=\gamma^k W_{ij,0},\qquad 0<\gamma<1, \tag{15}
\]

或由人体轨迹预测器提供均值/协方差 \((\mu_{i,k},\Sigma_{i,k})\)，用 \(W_{ij,k}\propto\Sigma_{i,k}^{-1}\) 降低不确定未来的强制跟踪。第一版应采用 (14)--(15)，并在论文中明确“离线 preview policy”和“因果 teleoperation policy”是不同评测设置；不能用完整未来人体动作训练后却声称在线遥操作。

---

## 7. 与 residual RL 的接口及奖励

MPC 不需要向 actor 暴露不可部署的完整未来人体序列。建议：

\[
\tau_t=\Pi_{[\tau^-,\tau^+]}\left(
\tau^{\rm kino\!\text{-}\!MPC}_t+\alpha\tau^{\rm RL}_{\rm res}(o_t,y^H_t,\hat y^H_{t+1})
\right). \tag{16}
\]

其中 `tau_kino-MPC` 来自完整 inverse dynamics/RNEA 的机器人模型结果，绝非 centroidal \(J^\top f\) 的简化前馈。MPC 输出的轨迹提供 critic/训练 landmark：

\[
z^{\rm MPC}_{t,k}=\big[c_k^{\star},\dot c_k^{\star},l_k^{\star},h_k^{\star},
\lambda_k^{\star},\tau_k^{\star},\sigma_k^{\rm MPC}\big]. \tag{17}
\]

奖励建议改为 task-space，而不是“human joint 与 robot joint”误差：

\[
\begin{aligned}
r_t={}&w_{\rm geom}\exp(-\textstyle\sum_{ij}\|e_{ij}(q_t,y_t^H)\|^2_{W_{ij}})
+w_{\rm lm}r_{\rm landmark}(x_t,z^{\rm MPC}_{t,0})\\
&+w_{\rm ct}r_{\rm contact}(\text{sim contact},\sigma^{\rm MPC}_{t,0})
-w_{\rm res}\|\tau^{\rm RL}_{\rm res}\|^2
-w_{\rm sat}\,\mathrm{clip\_fraction}
-w_{\rm viol}\,\mathrm{QPViolation}. \tag{18}
\end{aligned}

MPC 以 `torch.no_grad()`/detached landmark 方式被 PPO 使用，和当前 MPC-RL 一样；无需、也不应先要求穿过接触仿真器反传。此处“联合”指 control objective 的联合求解以及经环境回报的协作训练，并非 end-to-end differentiable MPC-to-simulator policy gradient。

### 需要证明与不能证明的内容

在固定 mode、\(C^2\) kinematics/dynamics、LICQ/SOSC、有效 warm start 和 trust region 下，(10)--(12) 给出原问题的一阶一致 RTI-QP；若投影集合是闭凸多面体，固定迭代的 PiMPC/ADMM 可以报告 QP KKT residual。由此可证明“每次优化的是机器人模型上的局部可行近似”，不能证明：

- 人体动作对任意机器人都可完整复现；
- 由 soft human task 和 residual torque 得到全局/递归可行性；
- 不带 robust terminal set 的接触切换闭环稳定性；
- actor residual 后仍满足 MPC 预测的摩擦/接触约束。

若要更强安全主张，必须加入模型误差界、约束收紧、终端不变集/backup controller，以及 residual 允许集；这些与原 `Residual kino-dynamic NMPC` 文档第 5 节的边界相同。

---

## 8. 代码落地方案

本节是规划，**本次未实现该新任务**。推荐在 kino-RTI solver 完成后实施，避免将复杂功能塞入当前 `MotionReferenceCommand`。

1. `src/themis_training/human_motion_command.py`
   - `HumanMotionCommandCfg` 读取 SMPL/SMPL-X/BVH/teleop stream；输出 key-body pose、速度、source timestamp 与 valid mask；
   - 不要求 `joint_names == robot.joint_names`，但严格检查 source skeleton mapping 和单位/fps/quaternion convention。
2. `src/themis_training/human_robot_mapping.py`
   - 数据类 `HumanRobotKeyBodyMapping`：human body、robot link、位置/旋转权重、\(S_i\)、rest-pose offsets、contact patch mapping；
   - 每种机器人（G1-29、Jingchu01-28、THEMIS）分别写显式配置，禁止按 array index 猜测对应关系。
3. `src/themis_training/online_human_reference.py`
   - 执行 (4)、(14)--(15)、source-contact heuristic 和合法 preview mask；
   - 输出 `human_task_horizon`，而不是 `joint_pos` reference；非循环 clip 尾部沿用静止/无效 mask，禁止 wrap-around。
4. 扩展 `src/themis_mpc/kino_nmpc.py`（来自 `residual_kino_nmpc_pimpc_extension.md`）
   - 新增 `HumanTaskReference` 输入以及 (6)、(11) 的 batched GPU Jacobian/Hessian blocks；
   - 接口输出机器人预测 \(q,v,c,l,h,\lambda,\tau\) 和 solver diagnostics。
5. `src/themis_training/residual_kino_action.py`
   - 实现 (16)，保留 clip/fallback 日志；不修改当前 `HybridMimicAction`。
6. 新注册任务，而不更改现有任务：
   `Mjlab-OnlineHuman-KinoMPC-Residual-Jingchu01-28DOF`，后续再添加 G1。

### 分阶段实验

| 阶段 | 目的 | 关键判据 |
|---|---|---|
| P0 | offline GMR NPZ baseline | 与当前 `MotionReferenceCommand` 完全兼容 |
| P1 | online-GMR/IK baseline | 不写 NPZ，但 task-space error 与 P0 接近；测 CPU/GPU 延迟 |
| P2 | 无 residual 的 OHRKP | 接触/torque/KKT 违反低，扰动后能从实测状态恢复 |
| P3 | OHRKP + residual RL | mimic、fall rate、energy、sim2sim 优于 P0/P1 |
| P4 | causal teleop | 只给当前/下一 source frame；与 privileged preview 分别报告 |

必须比较 GMR→MPC-RL、online IK→MPC-RL、KDMR/其可用输出→tracker（若数据与许可证允许）、以及 proposed OHRKP。除了总 reward，还应报告人体 key-body tracking、robot joint/torque limits、foot slip/penetration、接触 F1、CoM/动量误差、MPC failure/KKT residual、GPU latency 与吞吐。

---

## 9. 最终建议

短期内，不要删除现有 retargeted-NPZ 路径：它是可复现 baseline，且 GMR 的结果显示高质量 retarget reference 本身会强烈影响下游 motion-tracking success。先完成 generic kino-RTI PiMPC，再以 G1 或 Jingchu01 的一个 source clip 实现 P1/P2。

论文主张应是：**“人体参考条件化的、在线闭环 kino-dynamic 投影与 MPC-landmark-guided residual RL”**，而不是“免重定向”。前者既准确描述数学对象，也与 GMR 的离线两阶段 IK 和 KDMR 的离线 kinodynamic trajectory optimization 形成清晰区分。

## 10. 参考资料

1. Araujo et al., [Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking](https://arxiv.org/abs/2510.02252), ICRA 2026. GMR 的关键对应、非均匀局部缩放、两阶段 IK、以及 retargeting quality 对 BeyondMimic 跟踪的影响。
2. Zhang et al., [Kinodynamic Motion Retargeting for Humanoid Locomotion via Multi-Contact Whole-Body Trajectory Optimization](https://arxiv.org/abs/2603.09956), 2026. 离线多接触 kinodynamic retargeting，直接相关的现有工作。
3. Jeon et al., [Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control](https://arxiv.org/html/2510.12717), 2025. GPU RTI/SQP MPC 与 residual control 的参考。
4. 本仓库 `docs/residual_kino_nmpc_pimpc_extension.md`、`src/themis_training/mimic_mdp.py`、`src/themis_training/reference_centroidal.py`。
