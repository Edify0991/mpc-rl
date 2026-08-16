# 机器人特定 MPC 包布局

## 目的

`themis_mpc` 现在只承载 THEMIS 的论文复现基线：固定 phase-walking / double-support schedule、
centroidal QP、PiMPC/JAX-PiMPC 与 THEMIS loco-manipulation。它不再包含 reference-motion contact
schedule，也不为 mimic 任务提供落足/时序参数化入口。

G1 和 Jingchu01 的 CD-MPC solver 使用各自独立包：

```text
src/themis_mpc/       THEMIS paper baseline only
src/g1_mpc/           Unitree G1 QP/PiMPC copy + G1 schedule
src/jingchu01_mpc/    Jingchu01 QP/PiMPC copy + Jingchu01 schedule
```

三个包均是独立 Python package；因此后续机器人特有的质量、足底参数、接触几何、约束与 solver 改动
不会反向改变 THEMIS 基线或另一种机器人的实验。

## 每个 G1/Jingchu01 包的文件

| 文件 | 默认是否被训练任务导入 | 用途 |
|---|---:|---|
| `admm_qp.py`, `pimpc.py`, `jax_pimpc.py`, `centroidal_mpc.py`, `loco_manip_mpc.py` | 是 | 从 THEMIS 基线复制的 solver/MPC 实现；包内 import 已改为本机器人包名。 |
| `contact_schedule.py` | 是 | 已实现的 mimic schedule：reference 的 (H) 帧接触状态/位置，加冻结的 full-horizon **activation** residual。 |
| `experimental_contact_schedule.py` | 否 | 以前加入的 phase-rate、duty-factor、落足均值/方差、Raibert/reference touchdown residual、fast contact-intention 参数化。 |

训练 MDP 不属于 solver package。每个 `*_training/` 包中分别拥有：

| 文件 | 用途 |
|---|---|
| `mpc_grf_mdp.py` | 本机器人 `LocoMPCCommand`、原论文 MPC landmark/GRF reward 与训练特征。 |
| `mimic_mdp.py` | MotionReference command、mimic 观测与全身跟踪奖励。 |
| `mpc_grf_mimic_mdp.py` | mimic 专用 CD-MPC command、精确 articulated centroidal \(x_0\)、reference \(x^{ref}\)、contact-plan 与 exact landmark reward。 |

## 默认 mimic contact schedule

默认路径只执行

\[
\sigma_{t,k}^{MPC}=
\operatorname{clip}\left(
\sigma_{t,k}^{ref}+s_\sigma\tanh(a_{t,k}^{plan}),0,1
\right),
\]

其中 reference 的接触位置 (r_{t,k}^{ref}) 直接固定为 QP 参数；reference stage 原本有支撑而残差把
全部接触删除时，`preserve_nominal_support` 恢复其中一个接触到 (sigma=0.5)。每次 QP 内
(sigma,r) 都不是决策变量，因此 wrench 约束仍线性、centroidal QP 保持凸。

这条路径是当前 G1/Jingchu01 Phase-1 teacher 所使用的实现；它不采样落足点，也不把接触位置、接触时长
或方差作为训练 action。

## 实验性 schedule 的隔离

`experimental_contact_schedule.py` 保留了原先的探索性设计，但没有被 `centroidal_mpc.py` 或任何
task config 导入。它包括：

- `phase_rate_scale` 与 `duty_factor_offset`；
- touchdown mean residual 与 (xy) standard deviation metadata；
- phase-walking 场景中的 policy contact-state 调制；
- reference touchdown trajectory 覆盖与 Raibert-style landing update。

该文件仅作为后续实验起点。若要启用，必须同时完成并验证：参数 action 的语义、固定/采样策略、可行性
guard、reward/landmark 定义，以及随机落足或 chance constraint 的理论与实验；不能仅靠修改 import 就用于
主训练结果。

## 任务接线

- `g1_training/env_cfgs.py` 导入同包的 `mpc_grf_mdp.py` 与 `mpc_grf_mimic_mdp.py`；
- `jingchu01_training/env_cfgs.py` 导入同包的 `mpc_grf_mdp.py` 与 `mpc_grf_mimic_mdp.py`；
- 两者将从 THEMIS base config 得到的 `LocoMPCCommandCfg` 浅复制为各自
  `MimicLocoMPCCommandCfg`，再写入 robot-specific mass/inertia/site/contact 参数；
- `themis_training.__init__` 不再注册 THEMIS MPC-mimic/hierarchical/student task。THEMIS 保留
  `Mjlab-MPC-Guided-Locomotion-Themis`、`Mjlab-MPC-Guided-Loco-manipulation-Themis` 和纯
  `Mjlab-MotionTracker-Themis`。

因此已有 G1/Jingchu01 mimic task ID 不变，训练入口不需要改变；但其内部 MPC 实现已与 THEMIS 完全分包。
