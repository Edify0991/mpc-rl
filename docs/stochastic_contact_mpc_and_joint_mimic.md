# 随机接触 CD-MPC、控制 landmark 与联合 mimic 训练

本文档定义当前 hybrid-mimic 扩展的理论边界和下一步实现路径。除非本文列出的前提全部被实现并验证，不能宣称递归可行性、chance-constraint 安全性或全身闭环稳定性。

## 当前模型

当前 CD-MPC 的状态和控制为
\[
x_k=[c_k^\top,l_k^\top,k_k^\top]^\top,\qquad
u_k=[f_{L,k}^\top,\tau_{L,k}^\top,f_{R,k}^\top,\tau_{R,k}^\top]^\top.
\]
当接触日程、接触点和 CoM 线性化点固定时，
\[
x_{k+1}=Ax_k+B_k(\bar c_k,r_{L,k},r_{R,k},\sigma_k)u_k+d.
\]

MPCParameterNet 在 QP 组装之前冻结
\[
\theta=(s_\phi,\Delta d,\mu_{r,L},\mu_{r,R},
\Sigma_{r,L},\Sigma_{r,R},\Delta l,\Delta k).
\]
其中 \(s_\phi\) 是相位速率，\(\Delta d\) 是支撑时长偏置；参考落足加均值残差为
\(\bar r_i=r_i^{ref}+\mu_{r,i}\)，而
\(\Sigma_{r,i}=\mathrm{diag}(\sigma_{i,x}^2,\sigma_{i,y}^2)\) 是其 XY 不确定度。

当前实现固定数值积分步长 mpc_dt，只通过 \(s_\phi\) 改变接触时钟。这是条件于网络参数的凸 QP；不是 Gait-Net 的逐环境可变 dt。未来若输出完整 dt，必须针对每个参数值重建 \(A,B,d\)，但先冻结 dt 再求解仍可保持每个子问题为 QP。

## 方差、采样与 chance constraint

### 方差的正确语义

令实际触地点
\[
r_i=\bar r_i+\epsilon_i,\qquad\epsilon_i\sim\mathcal N(0,\Sigma_{r,i}).
\]
\(\mu_r\) 表示修正到哪里，\(\Sigma_r\) 表示对该修正有多不确定。确定但幅度很大的修正应有大 \(\lVert\mu_r\rVert\) 和小 \(\Sigma_r\)，不能把两者绑定。

### 是否应在 MPC 中采样

不推荐每个控制步只抽一个 \(r_i\) 并求唯一 QP：相同状态会得到随机 wrench/landmark，增加训练目标方差，且不构成概率安全保证。当前 sample_touchdown_candidates 仅提供镜像候选采样，未插入每步 QP。

可选的低频方案是在触地事件或 MPC 更新时生成 \(K\) 个候选：
\[
r_i^{(j)}=\bar r_i+L_i\epsilon_i^{(j)}.
\]
先通过安全检查，再批量求解 \(K\) 个 QP，以
\[
J_j=J_{MPC}^{(j)}+w_{\rm imit}e_{\rm mimic}^{(j)}
+w_{\rm risk}\mathrm{risk}(r^{(j)})
\]
选择最优可行候选。该方案可用 GPU batch，但有限采样本身不是 chance-constraint 证明。

### 接触面半空间概率约束

设可落足区域为凸多边形：
\[
\mathcal S_{i,k}=\{r\mid H_{i,k}r\le h_{i,k}\}.
\]
目标是 \(\Pr(r_i\in\mathcal S_{i,k})\ge1-\epsilon_{i,k}\)。把风险分配至各半空间行，且
\(\sum_j\epsilon_{i,k,j}\le\epsilon_{i,k}\)。由 Boole 不等式，以下保守条件足够：
\[
H_j\bar r_i+
\Phi^{-1}(1-\epsilon_{i,k,j})
\sqrt{H_j\Sigma_{r,i}H_j^\top}\le h_j. \tag{CC}
\]
等价于以
\[
b_j=\Phi^{-1}(1-\epsilon_{i,k,j})
\sqrt{H_j\Sigma_{r,i}H_j^\top}
\]
收缩区域 \(H_jr\le h_j-b_j\)，然后把网络均值投影到收缩区域。当前平地接触日程没有 \(H,h\)，所以尚未实现 (CC)。

触地点若成为 QP 决策变量，\((r-c)\times f\) 会重新双线性。第一版应让网络或候选器给出 \(\bar r\)，投影后固定它再求 QP；若以后需要 MPC 直接优化 \(r\)，采用 SCP，每轮冻结 \((\bar r,\bar f,\bar c)\) 并施加 trust region。

### 协方差如何进入 centroidal dynamics

在 \((\bar r,\bar f,\bar c)\) 处：
\[
(r-c)\times f\approx(\bar r-\bar c)\times\bar f
-[\bar f]_\times\delta r+[\bar f]_\times\delta c
+[\bar r-\bar c]_\times\delta f.
\]
单接触位置扰动的离散注入矩阵为
\[
E_{r,i,k}=
\begin{bmatrix}0_{3\times3}\\0_{3\times3}\\-\Delta t[\bar f_{i,k}]_\times\end{bmatrix}.
\]
给定模型/估计噪声 \(W_k\) 与局部反馈 \(\delta u=K_k\delta x\)，传播
\[
\Sigma_{x,k+1}=A_{cl,k}\Sigma_{x,k}A_{cl,k}^\top+
\sum_iE_{r,i,k}\Sigma_{r,i,k}E_{r,i,k}^\top+W_k,\quad
A_{cl,k}=A_k+B_kK_k.
\]
用 \(\Sigma_x\) 继续对 CoM、高度、摩擦和可达性约束做 back-off。没有 \(W,K,\Sigma_r\) 的校准，网络方差只是启发式置信度，不能称为概率保证。

### 推荐实现顺序

1. 加入 ContactSurfaceSet，保存每环境、horizon、足的 \(H,h\)、法向、摩擦和风险预算。
2. 增加 chance_constraints 模块，实现 (CC)、收缩区域和均值欧氏投影；候选先投影、再建接触日程。
3. 在 MPC 外层保存上一轮 \(\bar f\) 与 \(\Sigma_x\)，按上式传播。
4. 顺序实现：固定均值 + (CC) 投影 + nominal QP；K 候选 batch；最后才是触地点作为决策变量的 SCP。
5. 在 held-out 扰动上做 coverage calibration：标称 99% 应至少得到 99% 的经验接触面覆盖率。

## 可证明结论与边界

### 命题 1：固定参数下的条件凸性

条件是：单次 MPC 调用内 \(\theta,\sigma,\bar r,\bar c\) 固定，\(Q,Q_f\succeq0\)，\(R,R_\Delta\succ0\)，摩擦、CoP、力和 wrench 边界均线性。

结论是：当前 CD-MPC 子问题是凸 QP，且控制解唯一。

证明：此时 \(A,B_k,d\) 都为常数，动力学是仿射等式；固定接触后的零 wrench、摩擦金字塔、CoP 和饱和界线性。目标是状态、wrench 与 wrench 差分平方范数之和，Hessian 半正定，而 \(R\succ0\) 使控制块正定。严格凸二次函数在凸仿射可行域的极小点唯一。证毕。

### 命题 2：控制 landmark 不改变 QP 凸性

加入
\[
\ell_u=(u_k-u_k^{lm})^\top R_{lm}(u_k-u_k^{lm}),\qquad R_{lm}\succeq0,
\]
只会向 Hessian 加入 \(2R_{lm}\succeq0\)，并向线性项加入常数。故命题 1 保持成立。在 RL 中使用 \(\lVert f^{sim}-f^{lm}\rVert\) 的软奖励甚至不改变 QP，因此不需要单独稳定性证明。

### 定理 1：后续可使用的条件性递归可行性与 practical ISS

以下不是当前代码已经满足的结论，而是一组充分条件。

A1：网络均值满足 (CC)，协方差和模型扰动上界经过校准，且 horizon 内日程固定。

A2：每个 MPC wrench \(\lambda=[f,\tau]\) 都有受限全身实现：
\[
M(q)\ddot q+h(q)=S^\top\tau_q+J_c(q)^\top\lambda,
\]
并满足关节、力矩、不穿透和摩擦约束；realization 误差有界。

A3：存在终端集 \(X_f\)、终端反馈 \(\kappa_f\)、终端代价 \(V_f\)，对所有允许扰动保持正不变且满足约束，并有
\[
V_f(f(x,\kappa_f(x),w))-V_f(x)
\le-\ell(x,\kappa_f(x))+\gamma\lVert w\rVert^2.
\]

A4：网络输出限幅并已投影，总扰动 \(w\)（接触位置、模型、wrench realization 误差）有界。

则：若时刻 \(t\) 的鲁棒 MPC 可行，时刻 \(t+1\) 仍可行，并有
\[
V_N(x_{t+1})-V_N(x_t)
\le-\ell(x_t,u_t^\star)+\gamma\lVert w_t\rVert^2.
\]
因此闭环对 \(w\) practical ISS；\(w=0\) 时渐近稳定到终端目标或周期轨道。

证明：取时刻 \(t\) 的最优序列，移除已执行的首控制，把剩余序列移位，并在末端接 \(\kappa_f\)。A1 和 A2 将实际偏差包含在鲁棒 tube 内；A3 保证末端仍在 \(X_f\)，所以移位序列在 \(t+1\) 可行。由下一时刻的最优值不大于此候选值，前 \(N-1\) 个 stage cost 相消，余项由 A3 上界，得到上式。递推并使用正定 stage cost 下界得到 ISS；零扰动时 Lyapunov 值严格下降。证毕。

当前代码只满足“有界网络参数 + 固定参数 QP”。它还没有 chance back-off、协方差传播、终端集/终端控制器，也没有把 whole-body wrench realizability 写为硬约束。默认传感器只能测 3D 力，MPC wrench 仅作为 landmark 而非关节前馈力矩。因此当前工作只能声称条件凸 CD-MPC landmark 指导与经验鲁棒性，不能声称 chance-constrained 安全或闭环稳定已证明。

现有 MPC-RL 论文提供 PiMPC/ADMM 并行求解推导和实验，不提供上述“RL + 随机接触 + 全身实现”的递归可行性定理；Gait-Net 也不能替代 A1--A3。

## BeyondMimic 与 MPC-RL 能否同时训练

可以。对 reference 本身不可完整跟踪的动作，推荐一个 PPO actor、一个 critic、联合奖励，而非先冻结 tracker，也不必让两个 agent 竞争关节控制权。

建议单 actor 输出
\[
q^{des}=q^{ref}+\Delta q_\pi,\qquad
\tau=\tau_{PD}(q^{des}).
\]
快速策略同时输出每脚连续接触意图，慢速动作仍是有界 MPC adaptor 参数，而不是第三个直接控制 agent。总奖励可为
\[
r=w_mr_{mimic}+w_xr_{CoM/l/k}+w_fr_{force-landmark}
-w_a\lVert a\rVert^2-w_sr_{safety}.
\]

关键是按可行性调权：

1. 初期用较大 mimic 权重学习 reference 节奏。
2. 当 MPC 不可行、收缩接触面裕度小、或 landmark 误差大时，降低严格 pose imitation，增大安全和动力学 landmark 权重。
3. 对不可行片段只保留相位、上身或手部语义与平滑项；不要让关节逐帧误差强迫策略跌倒。
4. 逐步增加外扰、参考速度和落足偏差。
5. actor 必须拥有 \(\Delta q_\pi\)。当前 tracker_policy_path 是冻结 TorchScript；真正联合训练应把 tracker residual 合并到 PPO action head。

现有 hybrid 环境已把 mimic 与 MPC rewards 放在同一环境，联合奖励、单 residual PPO
仍可作为基线。`Mjlab-Hierarchical-HybridMimic-MPC-Themis` 已实现
`joint_target_residual_scale` 和慢速 held MPC parameter action；下文“当前仓库的实现”
给出了它与真正双优化器方案的边界。

### 基础 contact-aware MPC-RL 任务

`Mjlab-MPC-RL-Mimic-Contact-Themis` 是不含慢尺度 \(a_\theta\) 的 Phase-1 teacher。动作为
\([\Delta q_{29},A^\sigma_{H\times2}]\)，不使用 phase clock。motion loader 优先读取 NPZ 中显式的
`reference_contact_key` 标签；未提供时以参考候选接触点的相对最低高度和速度阈值离线生成
\(\sigma^{ref}_{i,k}\)。每次 QP 前固定
\(A^\sigma\)，并采用
\[
\sigma_{i,k}=\operatorname{clip}\bigl(\sigma^{ref}_{i,k}+
0.75\tanh(A^\sigma_{k,i}),0,1\bigr).
\]
因此参考动作决定整个 horizon 的接触序列，策略只在近端修正不可行参考或扰动；其 MPC 输入为 motion loader
重建的
\(x^{ref}=[c^{ref},l^{ref},k^{ref}]\)，控制参考固定为 \(u^{ref}=0\)：
\[
\min \sum_{j=1}^{N}\|x_j-x^{ref}_j\|_Q^2+
\|u_j\|_R^2+\|u_j-u_{j-1}\|_{R_\Delta}^2,
\]
其中 \(Q=\operatorname{diag}(Q_c,Q_l,Q_k)\)，而
\(u=[f_L,\tau_L,f_R,\tau_R]\)。因此它的奖励严格分为 motion mimic、MPC landmark
（CoM、动量、GRF 与接触一致性）以及动作/力矩/安全正则三类；相位 `foot_gait`
奖励关闭，避免覆盖策略学习的接触意图。

## 局部脚板、脚尖与边缘接触扩展

### 当前实现的实际语义

当前 feet_ground_contact 传感器的 primary 是左右 foot collision mesh，字段为 found 和 netforce。因此 mesh 的任一点（脚尖、脚跟或边缘）与 terrain 接触时，found 就可以为真；它不要求整只脚板贴地。快速 RL action 直接输出每脚连续接触意图 \(w_i\in(0,1)\)，并用该 foot-level found 作为在线奖励标签；不再有单独 GRU 或离线 BCE 训练。

在当前 `Mjlab-Hierarchical-HybridMimic-MPC-Themis` 中，动作布局是
\([\Delta q\;(29),\;a_w\;(2),\;a_\theta\;(16)]\)。每个低层步都取
\(w=\operatorname{sigmoid}(a_w)\)，并以
\[
\sigma_{i,k}=\operatorname{clip}\!\left(\sigma^{nom}_{i,k}
+0.75(2w_i-1),0,1\right)
\]
在每次 QP 组装前融合；\(w_i=0.5\) 保持名义日程，较小值减弱名义支撑，较大值提前激活 swing 足。接触奖励为
\[
r_{contact}=\exp\left(-\frac{\operatorname{mean}_i(w_i-y_i)^2}{0.25^2}\right),
\]
其中 \(y_i\) 是该步 MuJoCo 传感器的 `found`。融合后的 \(\sigma\) 在本次求解中固定，故不改变 CD-MPC 的条件凸性；`foot_gait` phase 奖励在该任务中置零，避免两个互相矛盾的接触监督信号。

这能修正第一种失配（swing 脚尖意外触地），但每只支撑脚仍使用完整矩形足底 wrench cone；只有脚尖接触时，MPC 仍可能使用整脚支撑才可实现的 CoP/moment。

### 现有 `cop_forward` 奖励：保留为行走基线，不作为 mimic 先验

当前 v2 locomotion 配置还注册了 `cop_forward_reward`。它不是 MPC landmark，也不是严格的 CoP
测量；它是为前向行走设计的 heel/toe 竖直力分布正则。XML 在每只脚的局部后部和前部放置独立的
heel/toe 微接触 geom，环境分别以 `heel_ground_contact` 和 `toe_ground_contact` 传感器读取这些
geom 与 terrain 的接触力。令每只脚的两组竖直力为 \(F_h,F_t\)，当前代码计算

\[
\rho_t=\frac{F_t}{F_h+F_t},\qquad
r_{\rm old\text{-}CoP}=\exp\left(-\frac{\tilde e^2}{s^2}\right),
\]

\[
e=\rho_t-0.5,\qquad
\tilde e=
\begin{cases}
2e,&e<0,\\
e,&e\ge0.
\end{cases}
\]

故后跟主导（\(\rho_t<0.5\)）受到更强处罚，而不接触的脚（总竖直力小于阈值）不计入该项。
它仅将 toe-load ratio 当作 sagittal CoP 前移的代理：若 heel/toe 的代表性局部前后坐标是
\(x_h<x_t\)，则近似有

\[
x_{\rm CoP}\approx\frac{x_hF_h+x_tF_t}{F_h+F_t},
\]

所以 \(\rho_t\) 增大通常意味着 CoP 前移。该实现没有使用实际接触点位置、接触面法向、每个
微点的力或完整 wrench，不能视为精确 CoP 估计。

该奖励适合普通前向行走的经验正则，但对非周期动作模仿不是普适先验：纯 heel、纯 toe、边缘、
滚动或舞蹈接触都可能与固定目标 \(\rho_t=0.5\) 冲突。当前 hybrid/mimic 环境从 v2 配置继承它；
正式的 reference-mimic 实验应将其权重置零，或只在 reference 已明确要求平足支撑的片段启用，
不能把它与 reference contact patch 的监督混为一谈。

### 后续：reference-/MPC-conditioned CoP landmark

在 patch contact 扩展中，应以每个微接触点的位置 \(p_{i,j}\)、接触激活 \(z_{i,j}\) 和沿接触
法向的力 \(F_{n,i,j}\) 计算仿真 CoP。对总法向力充分大的支撑脚：

\[
p_{\rm CoP,i}^{sim}=
\frac{\sum_j z_{i,j}F_{n,i,j}p_{i,j}}
{\sum_j z_{i,j}F_{n,i,j}},
\qquad
\sum_j z_{i,j}F_{n,i,j}\ge F_{min}.
\]

当仍使用平足 6D wrench 时，也可由 MPC 的局部 wrench 获得其隐含 CoP：

\[
p_x^{mpc}=-\frac{\tau_y^{mpc}}{f_z^{mpc}},\qquad
p_y^{mpc}=\frac{\tau_x^{mpc}}{f_z^{mpc}},
\]

该式只在 \(f_z^{mpc}\ge F_{min}\) 且 active patch 是二维足底面时有效；点接触或线接触时
不应伪造 roll/pitch CoP，而应直接跟踪活跃 patch/点力。

后续 CoP 奖励应是受接触有效 mask 门控的 landmark 跟踪：

\[
r_{\rm CoP}=
\mathbf 1_{\rm valid}
\exp\left(-\frac{\lVert\Pi_{\mathcal T}(p_{\rm CoP}^{sim}-p_{\rm CoP}^{lm})\rVert^2}
{\sigma_{\rm CoP}^2}\right),
\]

其中 \(\Pi_{\mathcal T}\) 投影到接触切平面，\(p_{\rm CoP}^{lm}\) 来自 MPC 的可实现 wrench，或来自
具有力/压力标签的 reference。仅有运动学 reference 时无法唯一推出 \(p_{\rm CoP}^{ref}\)，此时应
使用 reference patch activation / 接触点分布作为监督，而不是虚构 CoP reference。

推荐实施顺序：先记录 heel/toe/edge 的独立力和位置；再实现上述仿真 CoP estimator 与有效 mask；
随后把 MPC wrench 映射成 CoP landmark，并只在平足 patch 时启用；最后比较“无 CoP、旧
toe-ratio、MPC-CoP landmark、patch-CoP landmark”四种奖励。这样可检验 CoP 信号是否真的改善
不可行 reference 下的接触实现，而不是仅偏置机器人向脚尖承载。

### 推荐模型：接触 patch 而非每脚二值接触

每只脚定义固定的候选微接触点，例如 XML 中已有的 heel-in、heel-out、heel-center、toe-in、toe-out、toe-in2、toe-out2。对每点维护接触激活 \(z_{i,j}\)、位置 \(p_{i,j}\)、法向和摩擦；快速策略输出 \(p(z_{i,j}=1)\)，仿真中用每点 found 和法向力阈值加 hysteresis 生成在线奖励标签。

在固定预测激活下，把每点力作为 MPC 决策：

\[
\dot k=\sum_{i,j} z_{i,j}(p_{i,j}-c)\times f_{i,j}+\tau_{\rm torsion},
\]

并对每个 \(f_{i,j}\) 施加单点摩擦锥和法向界。由于 \(z,p,c\) 在线性化时固定，力仍线性进入 centroidal dynamics，子问题继续是 QP。相比“整脚 6D wrench”，它自然覆盖：单点接触时没有虚假的 roll/pitch CoP moment；多点接触时力的合成自动产生允许的力矩；线接触是两个或多个共线点的退化凸包。若硬件/仿真接触模型支持 torsional friction，才保留相应 yaw moment。

另一种等价描述是从活跃点的凸包得到 support patch，并在该凸包上建立 contact wrench cone；对点、线、面需要分别处理退化维度。第一版建议采用微接触点力，因为它避免手写退化 wrench cone，且更易与 MuJoCo 的接触点和力传感器对应。

### 实施顺序与创新边界

1. 将 feet_ground_contact 拆为 toe/heel/edge 的独立传感器，保存每点 found、力和接触位置；foot-level netforce 仅保留作总力 landmark。
2. 将快速策略的每脚 1 个接触意图扩为每脚 P 个概率或低维 patch mode（full, toe, heel, medial-edge, lateral-edge, point）。标签用法向力阈值和接触去抖生成。
3. 新增 PatchContactSchedule 和 point-force CD-MPC；当前时刻用测得/策略输出 patch，未来 horizon 用 held contact action。低置信度可接入本文前述 chance back-off。
4. 让 RL 同时跟踪总 wrench、CoP/接触点和 patch activation；在单点/线接触时显式降低不可能的接触力矩 landmark。
5. 做消融：整脚二值、已知 patch oracle、仅学习 patch、学习 patch 加 chance MPC；测试 toe-only、heel-only、edge/line、随机局部支撑和 reference-contact mismatch。

“识别脚尖接触”本身不是新问题：Atlas 的 partial-foothold 工作已经通过 CoP 探索估计线/点支撑区域并用于动量控制；已有 contact wrench cone 理论也描述了有限支撑面。可形成论文贡献的是：面向不可行动作模仿的、参考条件化的概率 contact-patch 表示，将学习到的 patch belief 直接变换为时变 centroidal wrench 可行集和 MPC/RL control landmarks，并在 patch 误判与局部支撑扰动下给出量化安全/性能界。

## CD-MPC 完整方程与端到端层级 RL

### 当前 CD-MPC 的优化问题

令 \(h=\Delta t\)，质量为 \(m\)，状态
\(x_k=[c_k^\top,l_k^\top,k_k^\top]^\top\in\mathbb R^9\)，控制
\(u_k=[f_{L,k}^\top,\tau_{L,k}^\top,f_{R,k}^\top,\tau_{R,k}^\top]^\top\in\mathbb R^{12}\)。
给定接触激活 \(\sigma_{i,k}\)、接触点 \(r_{i,k}\)、CoM 线性化点 \(\bar c_k\)，当前 QP 为：

\[
\begin{aligned}
\min_{x_{1:N},u_{0:N-1}}\quad &
\sum_{k=0}^{N-2}\left[\|x_{k+1}-x^{ref}_{k+1}\|_Q^2+
\|u_k-u^{ref}_k\|_R^2+\|u_k-u_{k-1}\|_{R_\Delta}^2\right]\\
&+\|x_N-x_N^{ref}\|_{Q_f}^2+
\|u_{N-1}-u^{ref}_{N-1}\|_R^2+\|u_{N-1}-u_{N-2}\|_{R_\Delta}^2,\\
\mathrm{s.t.}\quad &x_{k+1}=Ax_k+B_k(\bar c_k,r_k,\sigma_k)u_k+d.
\end{aligned}
\]

这里 \(Q=\operatorname{diag}(Q_c,Q_l,Q_k)\)，\(Q_f=10Q\)；当前默认
\(Q_c=(100,100,200)\)、\(Q_l=(10,10,20)\)、\(Q_k=(50,50,50)\)，
\(R_f=R_\tau=10^{-4}\)、\(R_\Delta=10^{-3}\)。当前 \(u^{ref}=0\)，
故 wrench landmark 是 MPC 求解结果，而非 CD-MPC 内部的已有 force reference。

等价的离散动力学为：

\[
\begin{aligned}
c_{k+1}&=c_k+\frac{h}{m}l_k+\frac{h^2}{2m}\sum_i\sigma_{i,k}f_{i,k}+\frac{h^2}{2}g,\\
l_{k+1}&=l_k+h\sum_i\sigma_{i,k}f_{i,k}+hm g,\\
k_{k+1}&=k_k+h\sum_i\sigma_{i,k}
\left[(r_{i,k}-\bar c_k)\times f_{i,k}+\tau_{i,k}\right].
\end{aligned}
\]

对每个 active 足的局部 wrench \(w=[f_x,f_y,f_z,\tau_x,\tau_y,\tau_z]\)，当前约束是：

\[
\begin{gathered}
|f_x|\le\mu f_z,\quad |f_y|\le\mu f_z,\quad 0\le f_z\le f_z^{max},\\
|\tau_x|\le y_h f_z,\quad -x_{toe}f_z\le\tau_y\le x_{heel}f_z,
\quad |\tau_z|\le\mu_z f_z .
\end{gathered}
\]

实现中用脚姿态把世界 wrench 旋至局部坐标后施加上式；inactive 足的完整 wrench 被置零。当前没有状态硬约束、终端集、接触 patch 或 chance constraint。

### 高层 actor 的物理动作如何进入该 QP

推荐高层在每 \(M\) 个低层控制周期输出并保持：

\[
a^H_t=[s_\phi,\Delta d,\Delta r_{L,x:y},\Delta r_{R,x:y},
\Delta l,\Delta k].
\]

其作用严格限定为：

\[
\begin{aligned}
\phi_{k+1}&=\phi_k+s_\phi\omega_0h,\\
duty&=duty_0+\Delta d,\\
r^{mpc}_{i,k}&=r^{ref}_{i,k}+\Delta r_i,\\
[l^{ref}_{k},k^{ref}_{k}]&\leftarrow[l^{ref}_{k},k^{ref}_{k}]
 +\rho_k[\Delta l,\Delta k],
\end{aligned}
\]

其中 \(\rho_k\) 是当前实现中的 horizon ramp。高层不应直接输出接触力、关节力矩或
任意 CoM 偏移；这些保留给带物理约束的 CD-MPC 和低层策略。

高层策略的探索协方差 \(\Sigma_{\pi_H}\) 与物理落足误差协方差 \(\Sigma_r\) 必须分离。
前者是 PPO 的动作分布参数；后者仅能由接触误差数据校准的 uncertainty head、ensemble
或监督 NLL 训练得到，并用于 chance back-off。若让 PPO 仅靠回报任意输出 \(\Sigma_r\)，
它会学会操纵风险尺度而非诚实表达不确定度。

### 联合训练的可行架构

这是可行的 semi-MDP/双时间尺度层级 RL，但当前 TorchScript adaptor 不是该架构。建议：

\[
\begin{aligned}
a_t^H&\sim\pi_H(o_t^H), &&\text{每 }M\text{ 个低层周期更新一次},\\
z_t^{mpc}&=\mathrm{CDMPC}(x_t,x^{ref}_t,a_t^H),\\
a_t^L&\sim\pi_L(o_t^L,z_t^{mpc},a_t^H), &&\text{每个策略周期更新},\\
\tau_t&=\tau_{PD}(q^{ref}+\Delta q_t^L).
\end{aligned}
\]

低层奖励可使用
\(r_t^L=w_mr_{mimic}+w_xr_{CoM/l/k}+w_fr_{force/wrench}-w_aa_t^{L2}-w_sr_{safety}\)。
高层使用 macro reward
\(R_t^H=\sum_{j=0}^{M-1}\gamma^jr_{t+j}^L-w_\theta\|a_t^H-a_{nom}\|^2-w_{risk}r_{risk}\)。
高层额外正则是必要的，否则它可通过大幅改写动量/落足参考来规避 mimic 任务。

训练时采用共享特征编码器和 centralized critic \(V(o^H,o^L,a^H)\)，但保留两个 actor head。
这避免两个独立 PPO 同时改变环境目标造成强非平稳性；高层 advantage 使用跨 \(M\) 步的
aggregate rollout，低层 advantage 按单步计算。初始时将高层输出固定在 nominal 并逐步
放宽其动作界、扰动和不可行 reference 比例。

这种 RL 意义下的“端到端”是两个 actor 均通过最终环境回报同步更新；它**不是**自动微分
穿过 MuJoCo、接触和当前 no-grad CD-MPC 的解析端到端。若需要后者，必须移除 inference/no-grad，
使用可微 QP 的 KKT implicit differentiation 与可微仿真/可微 surrogate；这不是 PPO 层级训练的必要条件。

### 当前仓库的实现：共享 rollout 的分层 action

任务 ID 为 `Mjlab-Hierarchical-HybridMimic-MPC-Themis`。当前安装的 rsl_rl runner
只支持一份 rollout 和一套 PPO/critic，因此实现采用**一份 factorized Gaussian actor 的
分层 action**，而不是不安全地并行注册两个彼此非平稳的 PPO 优化器：

\[
a_t=[a_t^L,a_t^H],\qquad
q_t^{des}=q_t^{ref}+0.30\,a_t^L .
\]

对当前 29-DoF THEMIS，该向量维度为 \(29+2+16=47\)：

* 低层每步输出全身关节动作 \(a^L\in\mathbb{R}^{29}\) 和连续接触意图
  \(w\in(0,1)^2\)；不输出 \(\Delta f\) 或 \(\Delta\tau\)。其执行力矩为
  \(K_p(q^{des}-q)+K_d(\dot q^{ref}-\dot q)\)。接触意图在 QP 前冻结为连续
  schedule 参数，并以仿真真实接触作为奖励监督；
* 高层 \(a^H\in\mathbb{R}^{16}\) 每五个策略周期才写入一次，并在中间周期保持；
* `LocoMPCCommand` 将 held \(a^H\) 通过 `decode_parameters` 映射到有界相位率、
  duty residual、左右足落足均值/标准差和 \([\Delta l,\Delta k]\)，再运行 detached QP；
* QP 的 CoM、速度、角动量、接触力和接触力矩轨迹是 landmark。低层的 mimic、CoM、动量、
  stance-gated GRF 和安全奖励共同给同一 rollout 提供回报。

这满足训练意义的联合优化：高层参数的因果效果经“QP landmark → 仿真轨迹 → 回报”由 PPO
估计，绝不对 QP 或仿真求导。为使低层在高层 hold 时仍满足 Markov 性，actor 和 critic 都会
接收已解码的 held MPC parameter state 以及 CoM/动量/接触力 landmark。

`touchdown_std_xy` 当前只作为接触位置不确定性的显式 metadata 保存；本实现不会在每次 QP
内随机采样落足点，也尚未添加 chance-constrained back-off。因而标准差不会伪装成已经具备
概率安全保证的约束；后续扩展必须按本文前面的协方差传播与半空间概率约束实现。

### 选择建议

对“参考动作可能不可行”的问题，推荐采用上述高层 RL + 低层 hybrid RL，而不是单独离线训练
后冻结参数网络。它的主要风险是 credit assignment、接触时序的离散跳变和参考篡改；相应地应
使用低频 hold、连续有界动作、可行性/安全 penalty、reference-deviation regularization、课程学习
和 centralized critic。接触位置/时长/动量正好是高层的合理 action space；全身 PD residual、
接触力 residual 与关节力矩 residual 正好是低层的合理 action space。

## 参考文献

1. Li et al., Gait-Net-augmented Implicit Kino-dynamic MPC for Dynamic Variable-frequency Humanoid Locomotion over Discrete Terrains, RSS 2025. https://arxiv.org/abs/2502.02934
2. Gazar et al., Multi-contact Stochastic Predictive Control for Legged Robots with Contact Locations Uncertainty, 2024. https://arxiv.org/abs/2309.04469
3. Li et al., Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation, 2026. https://arxiv.org/abs/2606.05687
4. Olkin et al., Stability of Control Lyapunov Function Guided Reinforcement Learning, 2026. https://arxiv.org/abs/2605.01978
5. Wiedebach et al., Walking on Partial Footholds Including Line Contacts with the Humanoid Robot Atlas, 2016. https://arxiv.org/abs/1607.08089
6. Caron et al., Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone for Rectangular Support Areas, 2015. https://arxiv.org/abs/1501.04719
