# 参考 centroidal 量、精确在线状态与坐标系契约

本文记录 motion-imitation CD-MPC 中每一个 centroidal 变量的来源、坐标系和变换。它对应
`src/themis_training/reference_centroidal.py`、`mimic_mdp.py` 与
`mpc_grf_mdp.py` 的现行实现。

## 1. 坐标系与状态约定

- \(\{M\}\)：参考 NPZ (`body_*_w`) 所属的原始 motion-world 坐标系。它是离线重定向/导出时的
  世界系。
- \(\{R\}\)：通过初始 anchor 预对齐后得到的 canonical reference world。
- \(\{W_e\}\)：第 \(e\) 个 MJLab/MuJoCo environment 的仿真世界系。
- \(\{B\}\)：机器人 root-link/body frame；MuJoCo XML 中的复合惯量近似
  \(I_{\rm approx}^{B}\) 在此表达。
- \(\{F_i\}\)：第 \(i\) 个足端 site 的接触局部坐标系。

CD-MPC 一律在 \(\{W_e\}\) 中求解：

\[
x_k=\begin{bmatrix}c_k\\l_k\\kappa_k\end{bmatrix},\qquad
c_k,l_k,\kappa_k\in\mathbb R^3_{W_e},
\]

其中 \(l=m\dot c\)，\(\kappa=k_G\) 是**关于系统 CoM**的角动量。接触位置
\(r_{i,k}\)、接触力 \(f_{i,k}\)、接触 wrench \(u_{i,k}=[f_{i,k};\tau_{i,k}]\) 也先在
\(\{W_e\}\) 中表达。

## 2. 离线参考的质心与动量

由 BeyondMimic 或 `csv_to_npz_mjlab` 生成的标准动作文件提供每个刚体原点的
\(p_j^M,q_{WB,j}^M\)，以及该刚体**惯性 CoM**的 \(v_{C,j}^M\)、角速度 \(\omega_j^M\)。由 MuJoCo
模型给出质量 \(m_j\)、惯性帧偏移 \(d_j^B\)、惯性主轴 \(I_j^I\) 和惯性帧朝向。实现首先计算

\[
p_{C,j}^{M}=p_j^M+R_{WB,j}^M d_j^B,\qquad
v_{C,j}^{M}=\texttt{body\_lin\_vel\_w}_j.
\]

只有来自直接 MuJoCo origin-Jacobian 重建的数据才有 \(v_{j,\mathrm{link}}^M\)，此时必须显式指定
`body_linear_velocity_point="link_origin"`，再使用
\(v_{C,j}^{M}=v_{j,\mathrm{link}}^M+\omega_j^M\times(R_{WB,j}^Md_j^B)\)。两种速度语义不能混用，
否则会重复叠加刚体偏置处的切向速度。

\[
c^M=\frac{1}{m}\sum_jm_jp_{C,j}^M,\qquad
l^M=\sum_jm_jv_{C,j}^M=m\dot c^M,
\]

\[
k_G^M=\sum_j\left[
R_{WI,j}^M I_j^I(R_{WI,j}^M)^\top\omega_j^M+
(p_{C,j}^M-c^M)\times(m_jv_{C,j}^M)
\right].
\]

接触点由候选 body origin 与配置的 body-frame 偏移导出：

\[
r_i^M=p_{b(i)}^M+R_{WB,b(i)}^M d_i^B,
\qquad r_{i/C}^M=r_i^M-c^M.
\]

所以这里得到的是完整刚体质量分布上的参考 \(k_G\)，不依赖地反力或接触标签；接触标签仅决定
MPC 在该 stage 是否允许某个 wrench 非零。

## 3. BeyondMimic 风格的预对齐，再计算 centroidal 量

BeyondMimic 的 `MotionCommand` 将 motion position 加到每个 environment 的 `env_origins`，而其
body-pose tracker 在运行时以 anchor 计算相对目标：保持参考的 \(z\)，用当前机器人 anchor 的
\(x,y\) 与 yaw 对齐全身 body target。它没有一个现成的“预计算 centroidal trajectory”模块。

本实现将该约定中的**初始 anchor 规范化**作为 `MotionReferenceCommand` 加载 clip 后执行的一次固定
motion preprocessing：先对完整刚体运动学预对齐，再调用 `compute_reference_centroidal()`，而不是先计算
\(k_G\) 后再旋转。原始 CSV/PKL 到 NPZ 的转换不改变坐标系，因此同一 NPZ 仍可原样 replay。令初始
anchor 的位置为 \(a_0^M\)，只取其水平部分
\(a_{0,xy}^M=[a_{0,x}^M,a_{0,y}^M,0]^\top\)，并令

\[
R_{RM}=R_z\!\left(-\operatorname{yaw}(q_{\rm anchor,0}^{M})\right).
\]

对每个刚体的完整轨迹执行：

\[
p_j^{R}=R_{RM}(p_j^M-a_{0,xy}^M),\qquad
q_{WB,j}^{R}=q_{RM}\otimes q_{WB,j}^{M},
\]

\[
v_j^{R}=R_{RM}v_j^M,\qquad
\omega_j^{R}=R_{RM}\omega_j^M.
\]

因此初始 anchor 的 \(x,y\) 为零、初始 yaw 为零，而其高度、完整 roll/pitch、关节运动、后续根部平移
都被保留。之后由第 2 节公式从 \(\{R\}\) 的各 link pose/velocity 计算
\(c^R,l^R,k_G^R,r_i^R\)。这比“先算 \(k_G^M\)，再临时与机器人误差一起旋转”更适合 MPC：物理量与
接触位置从同一套预对齐刚体运动学得到。

`prealign_reference_kinematics_to_initial_anchor()` 实现这个纯函数。新的默认接口是
`reference_frame_alignment="initial_anchor"`，它让 tracker body target、anchor reward、contact position 和
centroidal reference 共享同一 canonical clip；`prealign_centroidal_to_initial_anchor=True` 仅保留为
`reference_frame_alignment="none"` 时的旧 centroidal-only compatibility fallback。

预对齐后，将位置平移到第 \(e\) 个仿真环境：

\[
p^{W_e}=p^{R}+o_e,\qquad
v^{W_e}=v^{R},\qquad l^{W_e}=l^{R},\qquad k_G^{W_e}=k_G^{R},
\]

其中 \(o_e=\texttt{env\_origins}[e]\)。\(k_G\) 是关于 CoM 的量，平移不改变其值；接触点随位置一同
平移，并重算 \(r_{i/C}^{W_e}=r_i^{W_e}-c^{W_e}\)。MPC reference 不随机器人当前 tracking error 重锚定，
否则规划目标会跟着误差漂移并失去全局 reference 的意义。

## 4. 精确在线 centroidal 状态：mimic MPC 的当前状态与 landmark

Mimic 任务现在通过 `LocoMPCCommand.current_centroidal_state()` 读取 MJLab 的全部 articulated-body
world-frame link pose/velocity，并调用 `compute_centroidal_state()`。对每一个当前仿真刚体，仍按第 2 节
的原点到惯性 CoM 平移、spin 加 orbital 的公式计算，因此

\[
(c_0,\dot c_0,l_0,k_{G,0})=\operatorname{Centroidal}(
\{p_j^{W_e},q_j^{W_e},v_j^{W_e},\omega_j^{W_e}\}_{j=1}^{n_b}).
\]

这不是从 reward 反传穿过 simulator；它只是每个 environment step 的确定性状态测量。MPC 的初值使用
\(x_0=[c_0;l_0;k_{G,0}]\)，而新的、独立注册的
`MpcExactCentroidalLandmarkTracking` 使用

\[
r_{\rm landmark}=\exp\!\left[-w_c\|c-c^{mpc}\|^2-w_v\|\dot c-\dot c^{mpc}\|^2
-w_l\|l-l^{mpc}\|^2-w_k\|k_G-k_G^{mpc}\|^2\right].
\]

其中 landmark 是已 `detach` 的 MPC rollout 在当前训练时刻插值得到的目标。由于对固定质量系统
\(l=m\dot c\)，\(\dot c\) 与 \(l\) 两项在物理上冗余；默认保留较小的 \(w_l\)，便于直接记录和约束
动量，但实验中也可以把 \(w_l=0\) 以避免重复加权。

测点必须特别区分：`body_link_pos_w` 是 link/actor 原点，而 MJLab 的
`body_com_lin_vel_w` 是该 rigid body's inertial-CoM 线速度。因此实现使用
\(p_{C,j}=p_{link,j}+R_jd_j\) 与直接读取的 \(v_{C,j}\)。不能把
`body_link_lin_vel_w` 当作 link-origin 速度后再次加入 \(\omega\times R_jd_j\)，否则会把 CoM 偏移
重复计入并错误放大 \(l\) 和 \(k_G\)。离线 NPZ 的 `body_lin_vel_w` 则由 origin 处 Jacobian 导出，故
离线公式仍需要该平移项；二者接口含义不同，但最终的 \(c,l,k_G\) 定义相同。

## 5. 在线 \(k_G\) 根部近似：legacy fallback，不进入新的 mimic 路径

实时仿真接口提供：

```text
root_link_pos_w, root_link_lin_vel_w, root_link_ang_vel_w, root_link_quat_w
site_pos_w, site_quat_w
```

它们均按命名约定在 \(\{W_e\}\) 表达；`root_link_quat_w`、`site_quat_w` 给出 body/site 到世界的
旋转。原有的低成本在线状态近似被保留为：

\[
c_0^{\rm approx}=p_{\rm root}^{W_e},\qquad
l_0^{\rm approx}=m v_{\rm root}^{W_e},
\]

\[
I_{\rm approx}^{W_e}=R_{WB}I_{\rm approx}^{B}R_{WB}^{\top},\qquad
k_{G,0}^{\rm approx}=I_{\rm approx}^{W_e}\omega_{\rm root}^{W_e}.
\]

此前 `omega_w @ I_body` 把世界系角速度直接乘以机身系惯量；在 robot 有 yaw/roll/pitch 时坐标系不
一致。现已在 `_approx_centroidal_angular_momentum_w()` 中加入上述惯量旋转，并在
`MpcAngMomTracking` 和 `mpc_ang_vel_tracking` 中使用同一约定。后者的反变换为

\[
\omega^{W_e}=(I_{\rm approx}^{W_e})^{-1}k_G^{W_e}.
\]

这仍然不是当前真实全身 centroidal momentum：它忽略关节速度、自转惯量随构型的变化和 root origin
到实际 CoM 的偏移。因此它仅为旧 locomotion reward/API 保留；新的 mimic MPC 初值与 exact landmark
均不调用它，也不能用它替代参考动作计算出的 \(k_G^{ref}\)。

## 6. 当前 imitation MPC 的数据流

当 `motion_command_name` 指向 `MotionReferenceCommand` 时：

\[
x_k^{ref}=
\begin{bmatrix}
c_k^{ref,W_e}\\l_k^{ref,W_e}\\k_{G,k}^{ref,W_e}
\end{bmatrix}
\]

来自 `reference_centroidal_horizon()` 的“预对齐后质量一致计算 + environment-origin 放置”，并直接作为
MPC cost target。`apply_mimic_centroidal_reference_to_mpc_target()` 是专用的、无状态的接入函数：

\[
J_{\rm state}=\sum_{k=1}^{N-1}\|x_k-x_k^{ref}\|_Q^2+
\|x_N-x_N^{ref}\|_{Q_f}^2,
\]

故完整的 \(k_{G,k}^{ref}\) 直接进入 \(Q_k\) 对应的动量跟踪项；它既不是 online \(k_G\) 的替换值，也
不需要穿过 MPC 反向传播。

在线状态使用第 4 节的 exact \(x_0=[c_0,l_0,k_{G,0}]\)。因此不能用 reference momentum 覆盖实际状态，
也不应再用 `anchor angular velocity × I_approx` 覆盖 full-reference \(k_G\)。若 motion command 不提供
`reference_centroidal_horizon()`，旧 `centroidal_horizon()` 分支仍保留，作为 legacy anchor-based
reference fallback；它不具有完整质量一致性，也无法保证参考惯量坐标系严格一致，不应用于新的 mimic
实验。

## 7. 接触 wrench 与坐标一致性

动力学中的力臂与 wrench 都在 \(\{W_e\}\)：

\[
\dot k_G=\sum_i (r_i^{W_e}-c^{W_e})\times f_i^{W_e}+\tau_i^{W_e}.
\]

`site_pos_w` 与经对齐的 `reference.contact_pos_w` 都在该系，所以 `_build_Bk()` 中的
`skew(r - c)` 坐标一致。摩擦锥和足底 CoP 边界首先定义于脚局部系 \(\{F_i\}\)。实现使用

\[
u_i^{F_i}=
\operatorname{blkdiag}((R_{W_eF_i})^\top,(R_{W_eF_i})^\top)u_i^{W_e},
\]

再施加 \(G_{foot}u_i^{F_i}\le b_{foot}\)。`site_quat_w` 被转换为
\(R_{W_eF_i}\)，因此倾斜足端时不是把世界竖直分量误当成足底法向分量。当前 `ContactSchedule` 只
保存求解时刻的足端朝向并沿 horizon 保持；这是现有接触锥实现的时不变朝向近似，而不是坐标系混用。

## 8. 已知近似及实验解释边界

1. reference 与 online state 都是全身质量一致的，但两者不同是合理的：前者来自重定向运动，后者来自
   接触、扰动和控制误差后的真实仿真状态。这正是 MPC 需要优化纠正的状态偏差。
3. reference contact position 在 swing phase 仍保存运动学位置，但当 \(\sigma_i=0\) 时该 contact
   wrench 被约束为零，因此不进入动力学作用项。
4. 参考 NPZ 必须确实满足 `body_*_w` 的 SI 单位、四元数 `[w,x,y,z]`、线/角速度均在同一 motion
   world frame。否则任何坐标变换都不能修复源数据定义错误。
5. `aligned_body_pos_w()` 已按 BeyondMimic 的**运行时** anchor-relative 规则产生 body tracking target；
   它和本文的固定 centroidal preprocessing 是两个不同接口。直接使用 raw `body_pos_w` 的旧 anchor
   position metric 未自动加 `env_origins`，因此多环境训练时不应把该 raw 值与 simulator-world 位置直接
   比较；应改用已对齐 target 或显式加 environment origin。
