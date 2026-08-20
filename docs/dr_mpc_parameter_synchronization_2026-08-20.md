# G1 / Jingchu01 DR–MPC 参数同步审计

## 结论

在本次修改前，THEMIS 已在 `themis_training/env_cfgs.py` 启用三类启动期
randomization：`pd_gains`（0.8--1.2）、`joint_armature`（0.5--1.5）和
`body_inertia`（pseudo-inertia，约 0.8--1.2）。基础 velocity 环境还包含
`foot_friction`。然而 THEMIS 的 `LocoMPCCommand` 在构造时把
`cfg.mass`、`cfg.mu_foot` 写进一个单一 `MPCConfig` 并预构建矩阵；它没有在
reset 后读取 `env.sim.model`，所以这些 DR **没有同步到 THEMIS MPC**。

修改前 G1/Jingchu01 仅将 velocity 环境已有的 `foot_friction` 的几何名映射到
各自足端；它们同样没有重新读取模型，也没有 batched MPC 参数。因此“仿真被随机化、
MPC 仍是 nominal”的不一致确实存在。

## 当前 G1 / Jingchu01 通路

两个机器人现在与 THEMIS 使用相同的 `pd_gains`、`joint_armature`、
`body_inertia` event 配置。每次 command reset，`LocoMPCCommand` 从
`env.sim.model` 获取该环境实际的：

- robot-body masses；
- left/right foot geom 的 tangential friction（同一足端多个 collision geom 取最小值）；
- body-inertia trace 对应的惯量近似；
- `dof_armature`、`actuator_gainprm`、`actuator_biasprm`、`actuator_forcerange` 快照。

这些值存储在 `MPCModelParameters[B]`，随每次 `MPCInput` 传入 `g1_mpc` 或
`jingchu01_mpc`。质量实际重建每环境的

\[
A_i[0{:}3,3{:}6]=\Delta t/m_i I,quad
B_{i,k}^{c}=\Delta t^2/(2m_i)E_{i,k},quad
d_i^l=m_i\Delta t g.
\]

足端摩擦和法向力上界实际重建每环境、每只脚的 wrench cone `G_i u <= h_i`；PiMPC
后端对应地使用每环境的 force/moment box bound。ADMM、PyTorch PiMPC 与 JAX PiMPC
均接受 batched `A,B,d`。因 batch dynamics 不再共享同一个 `A`，PyTorch/JAX PiMPC 的
静态 Ruiz preconditioner 在该路径被关闭；这避免把 nominal scaling 错用于随机化模型。

Mimic command 的在线 articulated centroidal state 也在 reset 后读取同一套实际
`body_mass/body_inertia`；reference clip 仍采用 matched nominal robot 重建，因而 MPC
纠偏量明确反映 DR 下的模型差异，而不是误将 reference 重建为随机化机器人。

## 重要边界

纯 centroidal wrench QP 的状态是 `(c,l,k)`，其动力学不含 joint actuator gain、armature
或 torque limit。故这些 actuator 数组已经被**同步记录**，但不能诚实地塞进当前
`A,B,d,G,h`。把 torque limit 直接解释成 foot-force cap 没有 Jacobian / whole-body
dynamics 的依据。

它们进入 MPC 硬约束的正确后续实现是研究计划中的线性化：在 reset/linearization point
构造 `tau = a + B_tau u`、`qdot+ = qdot + dt(a_qdd + B_qdd u)`，再把 joint torque、speed
和 position 边界加入 QP。届时 `MPCModelParameters` 中已保存的 armature/gain/force-range
就是该映射所需的 simulator-realized actuator 参数。

本次 batched CD-MPC 修改覆盖 G1/Jingchu01 的 locomotion 与 mimic teacher 路径。独立的
`LocoManipMPC`（18-D wrench，额外手接触/box model）仍使用其旧的专用矩阵装配，不能将它
称为已完成 DR 同步；在把同一 batched parameter contract 扩展到该独立 solver 前，应避免
用它做“DR-consistent MPC”实验。

## 线动量 reward 与缓存修正

`MpcExactCentroidalLandmarkTracking` 已移除
`w_linear_momentum * ||l-l^{mpc}||^2`。在固定质量下 `l=m\dot c`，它与 CoM velocity
tracking 重复；线动量仍保留为 MPC state/reference，以保证 centroidal dynamics 正确，
但不再作为独立 reward landmark。`MimicLocoMPCCommand.current_centroidal_state()` 现在每个
环境步只计算一次，critic 的 CoM/angular-momentum observation 与 reward 共用缓存。
