# 面向隐式动力学、局部运动学约束的 \(\pi\)MPC 扩展，以及 PINN 的正确角色

## 1. 结论与术语澄清

本文以 Wu et al. 的 [\(\pi\)MPC: A Parallel-in-horizon and Construction-free NMPC Solver](https://arxiv.org/html/2601.14414v2) 为准，而不是仅以本仓库的简化 JAX port 为准。

原 \(\pi\)MPC 的“NMPC”含义是：每个控制周期把非线性系统在线线性化（RTI/SQP），再用其 parallel-in-horizon ADMM 解该 LTV-QP；它不是一次 ADMM 直接全局求解非线性、带接触互补的 NLP。原文已允许逐 horizon 节点变化的 \(A_{t,k},B_{t,k},e_{t,k}\)，并通过 \(\Delta u\) 的 velocity-based 增广使更新闭式化。当前仓库 `src/themis_mpc/jax_pimpc.py` 是进一步受限的 port：公共常量 `A`、逐节点 `B_k`、公共 `e` 与 box clip。因此“当前代码限制”和“\(\pi\)MPC 原理论限制”必须分开讨论。

同样需要校正对 MPC-RL 的归因：其公开版本把 \(\pi^n\)MPC 用于**linearized centroidal-dynamics MPC**，并显式存储时变 \(\{A_k,B_k,e_k\}\)；论文的贡献是将原 \(\pi\)MPC 扩展成 massive-parallel batch solver，而不是已经求解了全身隐式非线性接触 NMPC [MPC-RL 第 III 节](https://arxiv.org/html/2606.05687v1)。截至本文检索的版本，文中没有给出“直接非线性 kino-dynamic \(\pi^n\)MPC”的具体算法、收敛证明或实验。因此下述方法是合理的研究扩展，而非对原论文现有实现的复述。

对于 humanoid kino-dynamic MPC，推荐构造一个 **implicit-RTI-\(\pi\)MPC**：

1. 外层按 RTI/SQP 在上一条 warm-start trajectory 处线性化；
2. 不强行消去浮基动力学、接触速度等隐式等式，而是令它们作为每个节点的局部线性等式；
3. 以每个 stage 的“前状态副本、控制、后状态副本”为局部变量，以 node state 为共识变量；
4. 用两块 ADMM：stage update 是彼此独立的小型 equality-constrained QP，node update 是邻居平均，局部不等式是阶段投影；
5. 所有 stage 的线性代数可同时在 \((B_{\rm env},N)\) 上执行，无需构造整个 horizon 的稠密/稀疏大 QP。

这确实保留了 \(\pi\)MPC 的 construction-free 与 horizon-parallel 精神，但不再保证原论文那样全部 closed-form 的极简矩阵公式：一般隐式等式需要每节点一个固定维度的 KKT/Schur-complement solve；一般多面体约束需要局部 projection/QP。其规模与 \(N\) 无关，因此仍适合 GPU batched factorization。

**不建议用 PINN “跳过”隐式动力学硬约束。**PINN/physics-informed world model 可以替代或修正模型预测，但一旦把真实 \(F=0\) 换成 \(\hat F=0\)，MPC 对真实系统的动态可行性保证立即丢失。更合理的第一用途是 warm start、接触/外力先验、未建模力 residual 和不确定度估计；解析刚体动力学和接触/力矩约束仍保留为 QP 硬约束。

---

## 2. \(\pi\)MPC 原始分裂：应继承的关键结构

原文从在线线性化或 RTI-NMPC 所得到的 LTV MPC 开始：

\[
\begin{aligned}
\min_{X,U}\quad&\sum_{k=0}^{N-1}
\frac12\|Cx_{k+1}-r_{y,k}\|_{W_{y,k}}^2+
\frac12\|u_k-r_{u,k}\|_{W_{u,k}}^2,\\
\text{s.t.}\quad&x_{k+1}=A_{t,k}x_k+B_{t,k}u_k+e_{t,k},\\
&u_k\in\mathcal U_k,\quad x_{k+1}\in\mathcal X_{k+1},\quad x_0=x(t). \tag{1}
\end{aligned}
\]

这里 \(A_{t,k},B_{t,k},e_{t,k}\) 本来就可以随 \(k\) 变化；原文明确将其视为 RTI-NMPC 的一般形式，并避免显式组装 horizon-wide QP。它不使用 Riccati recursion，而引入

\[
x_{k+1}=z_{k+1},\qquad B_{t,k}u_k=v_k,\qquad
z_{k+1}-A_{t,k}x_k-v_k-e_{t,k}=0. \tag{2}
\]

于是每个 stage 的 ADMM 更新只依赖该节点与上一次迭代邻居的变量，可沿 \(k\) 并行。为了让受约束 \(u_k\) 的局部更新也具有闭式形式，原文使用

\[
\bar x_k=\begin{bmatrix}x_k\\u_{k-1}\end{bmatrix},\qquad
\bar x_{k+1}=\bar A_{t,k}\bar x_k+\bar B_{t,k}\Delta u_k+\bar e_{t,k},\quad
\bar B_{t,k}=\begin{bmatrix}B_{t,k}\\I\end{bmatrix}. \tag{3}
\]

因为 \(\bar B_{t,k}^{\top}\bar B_{t,k}=B_{t,k}^{\top}B_{t,k}+I\succ0\)，\(\Delta u_k\) 的局部最小化可逆。其 \(z\) update 是对闭凸 \(\mathcal X\) 的投影。原文的并行与 construction-free 结论依赖于：线性动力学、二次凸代价、闭凸的阶段集合，以及不构造大矩阵而只存每个 stage 的小矩阵 [\(\pi\)MPC 原文第 II–IV 节](https://arxiv.org/html/2601.14414v2)。

这也说明两个重要事实：

- \(\pi\)MPC 并不要求 \(A\) 在 horizon 内恒定；本仓库的 port 才有这一实现限制。
- 它的 closed-form 更新不自动覆盖任意隐式动力学等式、通用 contact Jacobian equality 和一般多面体约束；这些必须改变 local update，但不必放弃 horizon-parallel splitting。

---

## 3. Kino-dynamic 隐式问题及其 RTI-QP

令局部构型使用 retraction \(q\boxplus\delta q\)，而非直接加 quaternion。每个 stage 定义

\[
x_k=(q_k,v_k),\qquad u_k=(a_k,\lambda_k,\tau_k),\qquad
\xi_k=(x_k^-,u_k,x_{k+1}^+). \tag{4}
\]

其中上标 \(-,+\) 仅表示 stage 的前/后状态**局部副本**。将离散积分、全身刚体动力学、接触运动学等式统一写成隐式 DAE/约束

\[
F_k(\xi_k)=0. \tag{5}
\]

典型组成是

\[
\begin{aligned}
q_{k+1}&=q_k\boxplus\left(\Delta t v_k+\tfrac12\Delta t^2a_k\right),\\
v_{k+1}&=v_k+\Delta t a_k,\\
M(q_k)a_k+h(q_k,v_k)-S^\top\tau_k-J(q_k)^\top\lambda_k&=0,\\
J_{\mathcal C_k}(q_k)v_k&=0. \tag{6}
\end{aligned}
\]

\(\mathcal C_k\) 是在一次 MPC 调用前冻结的 contact mode/patch 集。摩擦多面体、力矩/状态界、摆脚高度、collision linearization 等局部凸约束写成

\[
G_k(\xi_k)\le0,\qquad H_k^{\rm kin}(\xi_k)=0. \tag{7}
\]

对第 \(\ell\) 个 SQP 名义轨迹 \(\bar\xi_k^\ell\)，定义 \(\delta\xi_k\) 并线性化：

\[
\begin{aligned}
D_k\delta\xi_k&=d_k,
&&D_k=\nabla F_k(\bar\xi_k),\quad d_k=-F_k(\bar\xi_k),\\
E_k\delta\xi_k&=e_k,
&&E_k=\nabla H_k^{\rm kin}(\bar\xi_k),\quad e_k=-H_k^{\rm kin}(\bar\xi_k),\\
C_k\delta\xi_k&\le r_k,
&&C_k=\nabla G_k(\bar\xi_k),\quad r_k=-G_k(\bar\xi_k). \tag{8}
\end{aligned}
\]

令 \(\mathcal D_k=\{\delta\xi:D_k\delta\xi=d_k,E_k\delta\xi=e_k\}\)，
\(\mathcal Z_k=\{\delta\xi:C_k\delta\xi\le r_k\}\)。对 tracking/kinematic residual 作 Gauss--Newton：

\[
\ell_k(\bar\xi_k\boxplus\delta\xi_k)
\simeq\tfrac12\delta\xi_k^\top Q_k\delta\xi_k+q_k^\top\delta\xi_k,
\qquad Q_k=J_{\ell,k}^\top W_kJ_{\ell,k}+\rho_{\rm reg}I\succ0. \tag{9}
\]

注意 \(D_k\) 不要求能消成 \(\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k+d_k\)。若离散 DAE 是 index-1 且 \(\partial F/\partial x_{k+1}\) 满列秩，可用隐函数定理消元并退化为原 \(\pi\)MPC LTV 形式；但浮基动力学和接触方程中常有只依赖 \((x_k,u_k)\) 的行，故一般 \(\partial F/\partial x_{k+1}\) 不可逆。保留 (8) 的局部隐式等式更一般且数值上更诚实。

得到的固定 SQP 子问题为

\[
\begin{aligned}
\min_{\delta\xi}\quad&\sum_{k=0}^{N-1}
\left(\tfrac12\delta\xi_k^\top Q_k\delta\xi_k+q_k^\top\delta\xi_k\right),\\
\text{s.t.}\quad&\delta\xi_k\in\mathcal D_k\cap\mathcal Z_k,\\
&P_+\delta\xi_k=P_-\delta\xi_{k+1},\quad k=0,\ldots,N-2,\\
&P_-\delta\xi_0=0. \tag{QP-I}
\end{aligned}
\]

\(P_-\)、\(P_+\) 分别抽取 stage 前、后 state increment。固定接触 mode 下，\(Q_k\succ0\)、\(\mathcal D_k\) affine、\(\mathcal Z_k\) 闭凸时，(QP-I) 是凸 QP。

---

## 4. 用 node-consensus 构造 horizon-parallel ADMM

### 4.1 分裂

为每个物理节点引入唯一的 consensus state \(s_k\)（\(s_0=0\)），并为每个 stage 引入约束副本 \(z_k\)。将 (QP-I) 等价改写为

\[
\begin{aligned}
\min_{\delta\xi,s,z}\quad&\sum_{k=0}^{N-1}
\left(\tfrac12\delta\xi_k^\top Q_k\delta\xi_k+q_k^\top\delta\xi_k
+I_{\mathcal D_k}(\delta\xi_k)+I_{\mathcal Z_k}(z_k)\right),\\
\text{s.t.}\quad&P_-\delta\xi_k=s_k,\qquad P_+\delta\xi_k=s_{k+1},\\
&\delta\xi_k=z_k,\qquad s_0=0. \tag{10}
\end{aligned}
\]

这与原 \(\pi\)MPC 的 \((x,z,v)\) copy 思路相同，但不再依赖 \(B_k u_k=v_k\) 的特殊结构。将所有 \(\delta\xi\) 作为 ADMM block \(p\)，将 \((s,z)\) 合并为第二 block \(w\)。尽管 \(w\) 有两个张量，给定 \(p\) 后的最小化完全可分，故仍是标准**两块** ADMM，而不是没有收敛保证的任意三块 ADMM。

取 scaled dual 为 \(\alpha_k^-\)、\(\alpha_k^+\)、\(\eta_k\)，penalty 为 \(\rho_s,\rho_z>0\)。其 augmented Lagrangian 是

\[
\begin{aligned}
\mathcal L={}&\sum_k\Big[
\tfrac12\delta\xi_k^\top Q_k\delta\xi_k+q_k^\top\delta\xi_k+I_{\mathcal D_k}(\delta\xi_k)+I_{\mathcal Z_k}(z_k)\\
&+\tfrac{\rho_s}{2}\|P_-\delta\xi_k-s_k+\alpha_k^-\|^2
+\tfrac{\rho_s}{2}\|P_+\delta\xi_k-s_{k+1}+\alpha_k^+\|^2\\
&+\tfrac{\rho_z}{2}\|\delta\xi_k-z_k+\eta_k\|^2\Big]. \tag{11}
\end{aligned}
\]

### 4.2 全部 stage 并行的 \(\delta\xi\) update

定义

\[
R=\begin{bmatrix}P_-\\P_+\\I\end{bmatrix},\qquad
b_k^i=\begin{bmatrix}s_k^i-\alpha_k^{-i}\\s_{k+1}^i-\alpha_k^{+i}\\z_k^i-\eta_k^i\end{bmatrix},\qquad
K_k=Q_k+R^\top\operatorname{diag}(\rho_sI,\rho_sI,\rho_zI)R. \tag{12}
\]

每个 stage 的 primal update 为

\[
\delta\xi_k^{i+1}=
\arg\min_{\delta\xi\in\mathcal D_k}
\tfrac12\delta\xi^\top K_k\delta\xi-
\left(R^\top\operatorname{diag}(\rho_sI,\rho_sI,\rho_zI)b_k^i-q_k\right)^\top\delta\xi. \tag{13}
\]

令

\[
\tilde\xi_k=K_k^{-1}\left(R^\top\operatorname{diag}(\rho_sI,\rho_sI,\rho_zI)b_k^i-q_k\right),\quad
\tilde D_k=\begin{bmatrix}D_k\\E_k\end{bmatrix},\quad
\tilde d_k=\begin{bmatrix}d_k\\e_k\end{bmatrix}. \tag{14}
\]

若 \(\tilde D_k\) 满行秩，则 equality-constrained QP 的闭式 Schur complement 解为

\[
\boxed{\;
\delta\xi_k^{i+1}=
\tilde\xi_k-K_k^{-1}\tilde D_k^\top
\left(\tilde D_kK_k^{-1}\tilde D_k^\top\right)^{-1}
\left(\tilde D_k\tilde\xi_k-\tilde d_k\right).
\;} \tag{15}
\]

该计算在每个 \((b,k)\) 上独立：使用 `vmap`/batched Cholesky 或 LDL\(^\top\) 即可。它的矩阵维度是 \(n_\xi\) 与等式数 \(n_e\)，不随 \(N\) 增长。若接触 Jacobian 使 \(\tilde D\) 行秩亏，必须做 contact-rank 检查、删除冗余行或使用带阈值的 QR/SVD；盲目加逆会制造数值不稳定。

### 4.3 仍可并行的 consensus/projection update

对于中间 node \(1\le k\le N-1\)，\(s_k\) update 为两个相邻 stage 值的加权平均：

\[
s_k^{i+1}=\frac12\left[
P_+\delta\xi_{k-1}^{i+1}+\alpha_{k-1}^{+i}
+P_-\delta\xi_k^{i+1}+\alpha_k^{-i}
\right],\qquad s_0=0, \tag{16}
\]

末端 \(s_N=P_+\delta\xi_{N-1}^{i+1}+\alpha_{N-1}^{+i}\)。所有 \(k\) 可 simultanously gather/scatter，因而仍是 horizon-parallel tensor operation。

局部不等式副本为

\[
z_k^{i+1}=\operatorname{Proj}_{\mathcal Z_k}
\left(\delta\xi_k^{i+1}+\eta_k^i\right), \tag{17}
\]

随后按残差更新 dual：

\[
\begin{aligned}
\alpha_k^{-i+1}&=\alpha_k^{-i}+P_-\delta\xi_k^{i+1}-s_k^{i+1},\\
\alpha_k^{+i+1}&=\alpha_k^{+i}+P_+\delta\xi_k^{i+1}-s_{k+1}^{i+1},\\
\eta_k^{i+1}&=\eta_k^i+\delta\xi_k^{i+1}-z_k^{i+1}. \tag{18}
\end{aligned}

这正是所需的“每个节点子问题 + 一致性约束合并为大问题”，但没有显式组装大问题。原 \(\pi\)MPC 的原始三类 residual 也应扩展为：dynamics/kinematic equality residual、edge-consensus residual、stage-set projection residual及相应 dual residual；仅监控相邻 state 差不够。

### 4.4 不等式 projection 的真实代价

\(\mathcal Z_k\) 为 box、单 halfspace 或 SOC 时可使用解析投影，和 \(\pi\)MPC 原文一致。对于 humanoid 中的“摩擦 pyramid + torque box + joint box + 线性化 self-collision + swing-height”等交集，多面体投影通常没有一个统一的 elementwise closed form。可选路径：

1. **多副本 consensus**：每一种简单集合一个 \(z_k^{(j)}\)，各自解析投影，再在 stage 内共识；保持 GPU 并行，但增加 dual/迭代次数。
2. **固定迭代的 stage-local Dykstra/dual projected gradient**：每个 \((b,k)\) 独立；适合统一行数 padding 后的 GPU kernel。
3. **小型 batched QP**：投影 \(\min_z\|z-y\|^2/2\;\mathrm{s.t.}\;C_kz\le r_k\)。最通用，但只有在内层精度受控时外层 ADMM 的理论才近似成立。

不要为追求“closed form”删除接触速度或力矩等关键约束。工程上首先选择 1：box、摩擦 pyramid、接触 mask 分别复制；复杂自碰撞留为软 GN cost 或单独安全层。

---

## 5. 理论保证与其边界

### 定理 1：固定 RTI 子问题的等价性与 ADMM 收敛

假设：

1. 接触 mode、contact patch、所有线性化点在本次 solve 内固定；
2. \(Q_k\succeq0\)，\(\mathcal D_k\) 为非空 affine set，\(\mathcal Z_k\) 为非空闭凸集；
3. (QP-I) 有鞍点；每个 (13)/(17) 子问题精确求解；
4. \(\rho_s,\rho_z>0\)。

则 (10) 与 (QP-I) 等价，且两块 ADMM 的 primal residual 和 dual residual 收敛到零；任一极限点是 (QP-I) 的 KKT 点。

**证明概略。**(10) 仅添加等式副本，投影回 \(P_-\delta\xi_k=s_k=P_+\delta\xi_{k-1}\)、\(z_k=\delta\xi_k\) 后恢复 (QP-I)，反向令副本取原变量即得可行性，故等价。把所有 \(\delta\xi\) 记为 block \(p\)、所有 \((s,z)\) 记为 block \(w\)，目标是凸闭真函数之和，耦合是仿射等式。标准两块 ADMM 定理即给出收敛。式 (15)--(17) 恰为两个 block minimization 的可分实现，不改变其解。\(\square\)

### 定理 2：原 \(\pi\)MPC 是该构造的特例

令 \(D_k\delta\xi_k=d_k\) 可消成 \(\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k+e_k\)，令 \(\mathcal Z_k=\mathcal X\times\mathcal U\)，并采取原文的 \(v_k=B_ku_k\)、velocity-based state (3)，则上述 node-consensus split 可重参数化为原文 (2) 的 \((x,z,v)\) split。原文额外利用 \(\bar B_k^\top\bar B_k\succ0\) 得到不含局部 KKT 的 closed form；本扩展放宽了这一特殊可消结构。

### NMPC 层的准确结论

若 \(F,G,H^{\rm kin}\) 二阶连续、局部 LICQ/SOSC/严格互补成立且 warm start 足够近，则 (8)--(9) 是原非线性问题的一阶一致 SQP 模型；RTI 一次 solve 是局部 tracking step。上述 ADMM 定理只针对**固定线性化 QP**，不证明外层非凸 SQP 的全局收敛，也不证明 contact mode 切换、模型误差或 residual-RL 叠加后的递归可行性。

---

## 6. PINN/physics-informed world model：可做什么，不能做什么

### 6.1 为什么不能直接“跳过”硬动力学

若网络直接预测显式离散模型

\[
x_{k+1}=\hat f_\phi(x_k,u_k), \tag{19}
\]

并在线线性化 \(\hat f_\phi\)，则 PiMPC 当然仍能解一个 LTV-QP。但其可行性是对 \(\hat f_\phi\) 而非真实刚体/接触系统的可行性。令真实误差为 \(w_k=f(x_k,u_k)-\hat f_\phi(x_k,u_k)\)，则

\[
x_{k+1}=\hat A_kx_k+\hat B_ku_k+\hat e_k+w_k. \tag{20}
\]

除非可验证 \(w_k\in\mathcal W_k\) 并据此做 robust tube/constraint tightening，(19) 没有硬动力学、摩擦和力矩可行性保证。对接触丰富 humanoid，网络最容易在 OOD 接触切换时误差最大，恰是 MPC 最需要可信模型的时候。

QuietWalk 的 PINN 不应被误读为 NMPC solver：它以 inverse-dynamics consistency 从 proprioception 估计每脚**竖直 GRF**，之后将冻结 predictor 作为 RL 中的 impact-force reward 信号；它并未将 PINN 作为在线带约束 MPC 的动力学等式求解器 [QuietWalk](https://arxiv.org/abs/2604.23702)。

LIFT 也不是此类替换：它保留已知 Lagrangian dynamics，网络仅预测外接触/耗散等不确定 torque 与方差；该 physics-informed world model 用于模型内 stochastic exploration 和 policy finetuning，而非作为一个满足 contact/torque hard constraints 的在线 QP solver [LIFT](https://arxiv.org/html/2601.21363v3)。

### 6.2 推荐的混合残差模型

保留解析模型，学习未建模广义力：

\[
M(q)a+h(q,v)=S^\top\tau+J(q)^\top\lambda+r_\phi(q,v,u),\qquad
\Sigma_\phi(q,v,u)\succeq0. \tag{21}
\]

在 RTI 名义点可选两种安全级别：

- **冻结 residual（第一版）**：\(r_\phi\) 被 `stop_gradient` 后作为已知仿射项；(8) 仍为 QP。网络只改变 \(d_k\)，不改变在线凸性。
- **一阶 residual**：\(r_\phi\simeq\bar r+R_x\delta x+R_u\delta u\)。其 Jacobian 并入 \(D_k\)；依然是 QP，但更敏感于网络导数，必须以 trust region 限制 \(\|\delta\xi\|\)。

若有高置信误差界

\[
\Pr\{\|r_{\rm true}-r_\phi\|_{W^{-1}}\le\beta\}\ge1-\varepsilon, \tag{22}
\]

可将力矩、摩擦、接触速度等约束收紧为鲁棒/机会约束的内近似。例如对线性约束 \(a^\top y\le b\)，若预测不确定度传播为 \(\Sigma_y\)，使用

\[
a^\top\mu_y+\Phi^{-1}(1-\varepsilon)\sqrt{a^\top\Sigma_ya}\le b. \tag{23}
\]

但网络输出一个“方差”本身不构成 (22) 的校准证明；至少需要 held-out calibration、coverage test 与失配情况下的 fallback。当前框架还不具备此鲁棒/机会约束实现。

### 6.3 最值得先做的三个网络接口

| 网络输出 | 放入求解器的位置 | 是否损害硬动力学 |
|---|---|---|
| warm-start \((\bar x,\bar u)\)、ADMM dual、active-contact prior | 外层初值/迭代预算 | 否 |
| \(r_\phi,\Sigma_\phi\) 未建模广义力 | (21) 的冻结仿射项与约束收紧 | 仅受模型误差界限制 |
| 参考 contact probability / GRF prior | 作为 \(\|\lambda-\lambda_\phi\|^2\) 软 cost 或固定 schedule prior | 否；不要直接硬设为真实力 |
| 完整 \(\hat f_\phi\) 替代刚体/接触方程 | (19) | **是**，只对 surrogate 可行 |

因此推荐顺序是：先网络 warm start/接触先验，后 residual generalized-force 与不确定度，最后才考虑 learned world model 作为训练时 planning surrogate；不要让 PINN 取代 MPC 的硬约束内核。

---

## 7. 建议的软件实现

本节为设计，不在本次修改中实现。

1. 保留 `src/themis_mpc/jax_pimpc.py` 原接口，作为 CD-MPC baseline。
2. 新增 `src/themis_mpc/jax_implicit_rti_pimpc.py`：
   - static shapes `D,E,C,Q,q:[B,N,...]`，所有 stage 的最大接触/约束行数 padding；
   - `solve_qp_implicit_consensus()` 实现 (15)--(18)；`lax.fori_loop` 固定 inner iteration，AOT cache key 包含 \((B,N,n_\xi,n_e,n_{ineq})\)；
   - 输出 primal/dual/consensus/dynamics/inequality residual，而不仅是一个总 scalar。
3. 新增 GPU `kino_dynamics_backend`：输出 (8) 所需 residual/Jacobian，先与 Pinocchio finite difference oracle 比较；禁止 rollout 热路径中的逐环境 CPU Pinocchio。
4. `stage_projection` 首版优先拆为 box、friction pyramid、contact mask 的解析 projection；一般 polytope projection 以固定内迭代实现并记录残差。
5. 仅在 solver diagnostics 合格时把 \((c,l,h,\lambda,\tau)\) 写为 MPC landmark；失败时用上次可行 trajectory/安全 PD fallback，且将 failure 传入 critic/reward。
6. `PhysicsResidualNet`（可选）输出 \((r_\phi,\log\Sigma_\phi)\)，初期只 `detach` 到 (21)。它不能命名为“PINN solver”；训练损失应至少包括监督/仿真 transition loss、inverse-dynamics residual、接触 mask 与时序平滑。

### 验收顺序

1. 线性 LTV、无局部等式时，扩展 solver 应与原 \(\pi\)MPC/当前 baseline 在容差内一致。
2. 人工小 DAE 与有限差分验证 \(D,E,C\)，比较 CPU 高精度 QP 的 objective/KKT。
3. 固定双足/patch 接触，测 equality rank、摩擦/torque violation、ADMM residual 与 \((B,N)\) 扩展。
4. 加 Kino dynamics、RTI warm start、trust region、fallback。
5. 最后才做 PINN residual 消融：无网络、warm start-only、冻结 residual、带 uncertainty tightening；不要把“完整 learned dynamics replacement”作为主安全方案。

## 8. 可严谨写入论文的表述

> We formulate each RTI subproblem as a convex implicit-dynamics QP with stage-local DAE and kinematic equalities. By duplicating predecessor/successor states at each stage and enforcing node consensus through two-block ADMM, all stage KKT solves and set projections are batched over both environments and horizon nodes without assembling a horizon-wide QP. The method is construction-free with respect to the horizon, while preserving the original implicit equalities within each local subproblem.

不要写成“PINN 消除了非线性动力学”或“horizon 节点独立”。准确说法是：非线性通过 RTI 在外层局部近似；隐式线性化等式被保留为 stage-local hard constraints；并行只发生在 ADMM 的局部更新，时间一致性通过 consensus residual 迭代恢复。

## 9. 参考资料

1. Wu et al., [\(\pi\)MPC: A Parallel-in-horizon and Construction-free NMPC Solver](https://arxiv.org/html/2601.14414v2), 2026.
2. Gros et al., [From linear to nonlinear MPC: Bridging the gap via the real-time iteration](https://doi.org/10.1080/00207179.2019.1692091), 2020.
3. Hu et al., [QuietWalk: Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion Under Diverse Footwear](https://arxiv.org/abs/2604.23702), 2026.
4. Huang et al., [Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](https://arxiv.org/html/2601.21363v3), ICLR 2026.
5. Li et al., [Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation](https://arxiv.org/html/2606.05687v1), 2026.
6. 本仓库 `docs/residual_kino_nmpc_pimpc_extension.md`、`docs/online_human_reference_retargeting_mpc_rl.md` 与 `src/themis_mpc/jax_pimpc.py`。
