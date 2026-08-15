# 从质心动力学到固定接触计划 LTV-QP

## 1. 结论

给定固定数值步长、固定质量，并在一次 MPC 求解前冻结接触模式、接触位置和质心线性化轨迹，质心动力学可写成

\[
\xi_{k+1}=A_k\xi_k+B_ku_k+e_k,\qquad
\xi_k=\begin{bmatrix}c_k\\l_k\\\kappa_k\end{bmatrix}.
\tag{1}
\]

这不是将完整人形机器人非线性动力学变成线性系统；它是对低阶 centroidal dynamics 的关键双线性项

\[
(p_{i,k}-c_k)\times f_{i,k}
\tag{2}
\]

在预测轨迹处进行线性化。接触位置、时序和名义 CoM 是 QP 外先确定的参数，不是 QP 决策变量，因此最终仍为凸 QP。

当前仓库的 CentroidalMPC 实际实现为

\[
A_k=A,\qquad e_k=e,\qquad B_k=B(\sigma_k,p_k,\bar c_k).
\tag{3}
\]

即：**\(B_k\) 在 horizon 内时变，\(A,e\) 保持常量**。对于固定 \(\Delta t\) 的足地 centroidal model，这一简化是自然的；参考动作仍会通过接触、力臂、约束和代价使每个 stage 的问题变化。

MPC-RL 与原始 \(\pi\)MPC 的理论形式均允许 \(\{A_k,B_k,e_k\}\) 全部随 horizon 变化；当前仓库的 PyTorch/JAX port 只是更受限的实现接口。[MPC-RL](https://arxiv.org/html/2606.05687v1) [\(\pi\)MPC](https://arxiv.org/html/2601.14414v2)

## 2. 状态、控制与建模假设

所有量在同一世界/惯性坐标系表达：

\[
c\in\mathbb R^3,\qquad l=m\dot c\in\mathbb R^3,\qquad
\kappa=k_G\in\mathbb R^3.
\tag{4}
\]

\(\kappa\) 是关于系统 CoM 的 centroidal angular momentum。第 \(i\) 个候选接触的 wrench 为

\[
u_{i,k}=\begin{bmatrix}f_{i,k}\\\tau_{i,k}\end{bmatrix},
\qquad
u_k=\operatorname{col}(u_{1,k},\ldots,u_{n_c,k}).
\tag{5}
\]

一次 QP 中必须冻结

\[
\{\sigma_{i,k},p_{i,k},R_{i,k},\bar c_k\}_{i,k},
\tag{6}
\]

其中 \(\sigma_{i,k}\) 是接触计划，\(p_{i,k}\) 是接触位置，\(R_{i,k}\) 为接触局部坐标系，\(\bar c_k\) 为 CoM 线性化点。冻结不代表它们在 horizon 内恒定，而是代表它们不随本次 QP 的决策变量变化。

## 3. 连续时间 centroidal dynamics

忽略空气阻力和未建模外力：

\[
\dot c=\frac{1}{m}l,
\tag{7a}
\]

\[
\dot l=mg+\sum_{i=1}^{n_c}\sigma_i f_i,
\tag{7b}
\]

\[
\dot\kappa=
\sum_{i=1}^{n_c}\sigma_i\left((p_i-c)\times f_i+\tau_i\right).
\tag{7c}
\]

式 (7a)--(7b) 对状态和力是仿射的。非线性来自

\[
-c\times f_i,
\tag{8}
\]

即 CoM 与接触力的双线性积。若 \(p_i\) 也作为优化变量，\(p_i\times f_i\) 同样是非凸双线性项。因此落足位置不能与力同时在当前 QP 内优化。

## 4. 固定接触计划下的 LTV 线性化

### 4.1 冻结力臂

对给定名义 CoM 轨迹 \(\bar c_k\)，当前 CD-MPC 使用 frozen-moment-arm 近似：

\[
(p_{i,k}-c_k)\times f_{i,k}
\approx
(p_{i,k}-\bar c_k)\times f_{i,k}.
\tag{9}
\]

它忽略了

\[
-(c_k-\bar c_k)\times f_{i,k}.
\tag{10}
\]

完整一阶 Taylor 展开可说明这个近似。令
\(c_k=\bar c_k+\delta c_k\)、\(f_{i,k}=\bar f_{i,k}+\delta f_{i,k}\)，则

\[
\begin{aligned}
(p_i-c)\times f_i \approx{}&
(p_i-\bar c)\times\bar f_i\\
&-\delta c\times\bar f_i
+(p_i-\bar c)\times\delta f_i.
\end{aligned}
\tag{11}
\]

当前实现保留对全量 \(f_i\) 的线性项，但没有将
\(-\delta c\times\bar f_i\) 加入状态矩阵 \(A_k\)。这是低成本 landmark planner 的近似，而非完整 RTI 的一阶线性化。

定义叉乘矩阵 \([a]_\times b=a\times b\)。令

\[
F_k=
\begin{bmatrix}
\sigma_{1,k}I_3&0&\cdots&\sigma_{n_c,k}I_3&0
\end{bmatrix},
\tag{12}
\]

\[
M_k=
\begin{bmatrix}
\sigma_{1,k}[p_{1,k}-\bar c_k]_\times&\sigma_{1,k}I_3&\cdots&
\sigma_{n_c,k}[p_{n_c,k}-\bar c_k]_\times&\sigma_{n_c,k}I_3
\end{bmatrix}.
\tag{13}
\]

则

\[
\dot l=mg+F_ku_k,\qquad \dot\kappa=M_ku_k.
\tag{14}
\]

接触计划、接触位置和 \(\bar c_k\) 都可随 \(k\) 改变，因此 \(F_k,M_k\) 通常时变。

### 4.2 零阶保持离散化

令 stage 长度 \(h=\Delta t\)，控制在 stage 内零阶保持。对 \(c,l\) 使用常加速度积分、对 \(\kappa\) 使用一阶积分：

\[
c_{k+1}=c_k+\frac{h}{m}l_k+
\frac{h^2}{2m}F_ku_k+\frac{h^2}{2}g,
\tag{15a}
\]

\[
l_{k+1}=l_k+hF_ku_k+hmg,
\tag{15b}
\]

\[
\kappa_{k+1}=\kappa_k+hM_ku_k.
\tag{15c}
\]

将三式堆叠：

\[
\xi_{k+1}=A\xi_k+B_ku_k+e,
\tag{16}
\]

\[
A=
\begin{bmatrix}
I_3&\frac{h}{m}I_3&0\\
0&I_3&0\\
0&0&I_3
\end{bmatrix},\qquad
e=
\begin{bmatrix}
\frac{h^2}{2}g\\hmg\\0
\end{bmatrix},
\tag{17}
\]

\[
B_k=
\begin{bmatrix}
\frac{h^2}{2m}F_k\\
hF_k\\
hM_k
\end{bmatrix}.
\tag{18}
\]

固定 \(m,g,h\) 时，\(A,e\) 与 \(k\) 无关，但 \(B_k\) 是时变的，所以这是 LTV 而不是 LTI。

### 4.3 当前双足代码

当前控制变量为

\[
u_k=
\begin{bmatrix}
f_{L,k}\\\tau_{L,k}\\f_{R,k}\\\tau_{R,k}
\end{bmatrix}\in\mathbb R^{12}.
\tag{19}
\]

对应地：

\[
F_k=\begin{bmatrix}
\sigma_{L,k}I_3&0&\sigma_{R,k}I_3&0
\end{bmatrix},
\tag{20}
\]

\[
M_k=\begin{bmatrix}
\sigma_{L,k}[p_{L,k}-\bar c_k]_\times&\sigma_{L,k}I_3&
\sigma_{R,k}[p_{R,k}-\bar c_k]_\times&\sigma_{R,k}I_3
\end{bmatrix}.
\tag{21}
\]

这正是 [_build_Bk()](../src/themis_mpc/centroidal_mpc.py) 的含义：它对每个 horizon stage 使用 schedule.sigma、左右接触位置和 c_bar 构造 \([9,12]\) 的 \(B_k\)。

飞行相若 \(\sigma_{L,k}=\sigma_{R,k}=0\)，则 \(B_k=0\)，系统退化为：

\[
c_{k+1}=c_k+\frac{h}{m}l_k+\frac12h^2g,\quad
l_{k+1}=l_k+hmg,\quad
\kappa_{k+1}=\kappa_k.
\tag{22}
\]

## 5. LTV dynamics 如何形成凸 QP

对 horizon \(N\)，典型问题为

\[
\begin{aligned}
\min_{\xi_{1:N},u_{0:N-1}}\quad&
\sum_{k=0}^{N-1}
\|\xi_{k+1}-\xi_{k+1}^{ref}\|_{Q_k}^2+
\|u_k-u_k^{ref}\|_{R_k}^2\\
&+\sum_{k=0}^{N-1}\|u_k-u_{k-1}\|_{R_{\Delta,k}}^2+
\|\xi_N-\xi_N^{ref}\|_{Q_N}^2,\\
\text{s.t.}\quad&
\xi_{k+1}=A_k\xi_k+B_ku_k+e_k,\\
&u_k\in\mathcal U_k,\qquad \xi_{k+1}\in\mathcal X_k,\qquad
\xi_0=\phi(q_t,\dot q_t).
\end{aligned}
\tag{23}
\]

固定动力学矩阵后，状态等式是仿射的。接触 wrench cone 可使用多面体近似：

\[
\begin{aligned}
&f_z\ge0,\quad |f_x|\le\mu f_z,\quad |f_y|\le\mu f_z,\\
&|\tau_x|\le y_{\max}f_z,\quad
|\tau_y|\le x_{\max}f_z,\quad
|\tau_z|\le\mu_zf_z,\\
&\sigma_{i,k}=0\Rightarrow u_{i,k}=0.
\end{aligned}
\tag{24}
\]

这些都是线性不等式或 box bound。若
\(Q_k\succeq0,R_k\succ0,R_{\Delta,k}\succeq0\)，则 (23) 是凸 QP。原始 \(\pi\)MPC 的 parallel-in-horizon ADMM 正是针对这种逐节点 LTV 矩阵、二次代价和凸 stage constraint 的结构设计的。

若把 \(p_{i,k}\)、\(\sigma_{i,k}\)、\(h_k\) 或完整构型 \(q_k\) 同时当作 QP 内决策变量，则 (18)、(24) 会产生双线性或互补约束，问题不再是该凸 QP。

## 6. 当前参考动作驱动 CD-MPC 中的时变量

MotionReferenceCommand.reference_centroidal_horizon() 提供

\[
\{c_k^{ref},\dot c_k^{ref},l_k^{ref},\kappa_k^{ref},
p_{L,k}^{ref},p_{R,k}^{ref},
\sigma_{L,k}^{ref},\sigma_{R,k}^{ref}\}_{k=0}^{N}.
\tag{25}
\]

它们进入 LocoMPCCommand._update_command() 中的 x_ref 和 make_reference_contact_schedule()。逐项如下。

| 量 | 当前来源 | horizon 内时变 | 进入位置 | 是否破坏 QP |
|---|---|---:|---|---:|
| \(c_k^{ref},l_k^{ref},\kappa_k^{ref}\) | 重定向后 reference 的 centroidal 计算 | 是 | 代价 target；当前 \(c_k^{ref}\) 默认也是 \(\bar c_k\) | 否 |
| \(p_{L/R,k}^{ref}\) | reference 足端/接触点位置 | 是 | \(M_k\)，即 \(B_k\) | 否，若预先固定 |
| \(\sigma_{L/R,k}^{ref}\) | reference contact schedule | 是 | \(F_k,M_k,\mathcal U_k\) | 否，若预先固定 |
| \(R_{i,k}\) | 足端接触朝向 | 可时变 | 摩擦锥/CoP 集合 \(\mathcal U_k\) | 否，若预先固定 |
| \(\bar c_k\) | 当前默认为 x_ref 的 CoM | 是 | \(M_k\)，即 \(B_k\) | 否 |
| policy contact residual | 策略 action | 可时变 | 先修改 schedule，再冻结 | 否；若 QP 内优化则非凸 |
| momentum residual | 高层/参数策略 | 当前为 horizon ramp | \(x_k^{ref}\) | 否 |
| 实测 \(\xi_0\) | 仿真器当前状态 | 每次 MPC 调用变化 | 初始条件 | 否 |
| \(m,g,h,\mu\)、足底尺寸 | 当前 MPC config | 不变 | \(A,e,B_k,\mathcal U_k\) | 否 |

因此，对当前 mimic CD-MPC，**动力学中最关键且已经时变的对象是 \(B_k\)**。reference CoM、动量并不直接使动力学变化；它们主要是 cost target。reference 的接触状态、接触位置和 CoM 线性化点才通过 (13)、(18) 使 \(B_k\) 时变。

### 6.1 支撑脚锚点

若某脚在 reference schedule 中处于支撑相，但 GMR 后该脚的位置逐帧漂移，直接使用

\[
p_{i,k}=p_{i,k}^{ref}
\tag{26}
\]

会让 MPC 错误认为支撑点可无滑动地移动。更合理的做法是对每个连续支撑段 \(n\) 在 touchdown 时冻结接触锚点：

\[
\bar p_{i,n}=\Pi_{\rm terrain}(p_{i,k_{\rm TD}}^{ref}),\qquad
p_{i,k}=\bar p_{i,n},\quad k\in n.
\tag{27}
\]

它仍在 touchdown/离地时变，但不会在单个支撑相中虚构地面滑动。当前 make_reference_contact_schedule() 直接传入每帧 reference.contact_pos_w，尚未系统实现该支撑段锚定；这是后续 reference-contact preprocessing 的优先改进。

## 7. 何时 \(A_k,e_k\) 也应时变

对当前固定 \(h\)、固定质量、无显式交互外力的 frozen-arm CD-MPC，\(A,e\) 保持常量是合理的。下面情况会要求完整 LTV 接口。

### 7.1 可变积分时间

若策略先输出并冻结每一 stage 的 \(h_k\)：

\[
A_k=
\begin{bmatrix}
I_3&\frac{h_k}{m}I_3&0\\0&I_3&0\\0&0&I_3
\end{bmatrix},\qquad
e_k=
\begin{bmatrix}
\frac12h_k^2g\\h_kmg\\0
\end{bmatrix},
\tag{28}
\]

\[
B_k=
\begin{bmatrix}
\frac{h_k^2}{2m}F_k\\h_kF_k\\h_kM_k
\end{bmatrix}.
\tag{29}
\]

只要 \(h_k\) 在 QP 前冻结，仍是 QP；若同时优化 \(h_k\) 与 \(f_k\)，就有 \(h_kf_k\)、\(h_k^2f_k\) 双线性项。

### 7.2 已知时变外力或对象交互

若将预测/测量得到的物体作用 wrench 作为已知外力：

\[
e_k=e+
\begin{bmatrix}
\frac{h^2}{2m}f_k^{obj}\\
hf_k^{obj}\\
h((p_k^{obj}-\bar c_k)\times f_k^{obj}+\tau_k^{obj})
\end{bmatrix}.
\tag{30}
\]

搬运和推拉中的时变交互载荷会使 \(e_k\) 时变。若手部 wrench 是优化变量，它应扩展到 \(u_k\) 和 \(B_k\)，而非放入 \(e_k\)。

### 7.3 完整 RTI centroidal 线性化

若保留式 (11) 的 CoM 偏差项：

\[
-(\delta c\times\bar f)=[\bar f]_\times\delta c,
\tag{31}
\]

则角动量的状态 Jacobian 有

\[
A_k^{\kappa c}=h\sum_i\sigma_{i,k}[\bar f_{i,k}]_\times.
\tag{32}
\]

因为名义力 \(\bar f_{i,k}\) 沿 horizon 变化，\(A_k\) 必须时变。这是完整 RTI 比当前 frozen-arm 近似更精确的地方。

### 7.4 冲量 touchdown

普通飞行相只令 \(B_k=0\)，不要求改变 \(A,e\)。若显式建模着陆冲量，应使用不同的跳变节点：

\[
\xi_k^+=A_k^{imp}\xi_k^-+B_k^{imp}j_k+e_k^{imp},
\tag{33}
\]

其中 \(j_k\) 为冲量 wrench；它需要 stage-wise transition 或独立 impact node。

## 8. 当前实现与完整 \(\pi\)MPC LTV 接口

当前 PyTorch/JAX port 的实际接口是：

    A      : [nx, nx]                 # 全 horizon 共享
    B_s    : [B_env, N, nx, nu]       # 每个 environment、stage 可变
    e      : [B_env, nx]              # 对 horizon 广播

对应文件为：

- [centroidal_mpc.py](../src/themis_mpc/centroidal_mpc.py)
- [pimpc.py](../src/themis_mpc/pimpc.py)
- [jax_pimpc.py](../src/themis_mpc/jax_pimpc.py)

原始 \(\pi\)MPC/MPC-RL 的一般形式则是

\[
A_{t,k},\quad B_{t,k},\quad e_{t,k},
\tag{34}
\]

三者对每个 environment 和 stage 都可以不同。当前 port 已经拥有逐节点 \(B_k\) 的 parallel ADMM 结构；推广时应将接口改为

    A_s : [B_env, N, nx, nx]
    B_s : [B_env, N, nx, nu]
    e_s : [B_env, N, nx]

并让每个 stage 的局部 ADMM update 使用自身 \(A_{t,k},B_{t,k},e_{t,k}\)。

## 9. 对当前项目的建议

当前 Phase-1 mimic CD-MPC 建议先维持

\[
A_k=A,\qquad e_k=e,\qquad B_k=B(\sigma_k,p_k,\bar c_k),
\tag{35}
\]

优先完成：

1. 融合人体接触时序先验和重定向后机器人运动学，离线生成 \(\sigma_{i,k}^{ref}\)；
2. 为连续支撑段生成 touchdown anchor，实施式 (27)；
3. 飞行段强制 \(B_k=0\)，禁止 policy residual 虚构支撑；
4. 使用 reference CoM 或上一次 MPC 预测作为 \(\bar c_k\)；
5. touchdown 后短窗口降低瞬时 GRF landmark 权重，或改为比较力的窗口积分。

只有在研究可变 \(h_k\)、对象外力预测、冲量节点、完整 RTI Jacobian 或 kino-dynamic RTI 时，才应新建独立的完整 LTV solver：

    src/themis_mpc/centroidal_ltv_mpc.py
    src/themis_mpc/pimpc_ltv.py
    src/themis_mpc/jax_pimpc_ltv.py

保留原 CentroidalMPC 作为 baseline，并复用 motion loader、reference centroidal、reward 和 PPO 基础设施。这样才能干净比较

\[
\text{constant }(A,e)+B_k
\quad\text{vs.}\quad
\{A_k,B_k,e_k\}\text{ fully LTV}.
\tag{36}
\]

当前方法应准确表述为：接触计划和 reference-conditioned moment arm 在每次 MPC 调用前冻结的 centroidal LTV-QP；它为全身 imitation RL 提供训练期的预测性动力学 landmark，而不是精确求解完整人形全身动力学。
