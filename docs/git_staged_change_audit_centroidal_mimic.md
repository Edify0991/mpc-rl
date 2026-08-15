# 当前暂存区审计：MJLab 动作输入、精确质心动力学与 Mimic MPC 扩展

## 审计范围

本文只描述 Git **暂存区（index）**相对于 `HEAD` 的内容，不包含任何尚未暂存的工作区修改。
审计时暂存区共 21 个路径，统计为 **3279 行新增、124 行删除**：10 个新增文件、11 个修改文件。

它是 [git_change_audit_mpc_rl_mimic.md](git_change_audit_mpc_rl_mimic.md) 的后续审计文件；前一份
文档描述较早的 MPC-RL mimic/contact 方案，而本文件重点记录随后加入的：

1. 与 BeyondMimic 转换流程对齐的 MJLab motion I/O；
2. reference 与 simulator 的质量一致 centroidal quantity；
3. 将 mimic 扩展从原始论文基线 `mpc_grf_mdp.py` 拆分出去；
4. G1、Jingchu01 与 THEMIS 任务对新模块的显式接线；
5. 对 LTV-QP、PiMPC/RTI-NMPC、在线人类参考重定向的研究设计文档。

## 总体代码边界

```text
retargeted CSV / PKL
        │
        ▼
csv_to_npz_mjlab + mjlab_motion_io
        │  (MJLab forward capture; named NPZ)
        ▼
MJLabMotionLoader / MotionReferenceCommand
        ├── tracker q_ref, dq_ref, body targets
        └── reference centroidal/contact horizon
                       │
                       ▼
              mpc_grf_mimic_mdp
              (exact x0, x_ref, landmark reward)
                       │
                       ▼
              G1 / Jingchu01 / THEMIS mimic configs
```

`src/themis_training/mpc_grf_mdp.py` **不在当前暂存区**。它保持原始 MPC-RL 基线的命令、QP、奖励与
接口，因而可以继续用于对应论文意图的基线训练。新的 mimic 功能由
`src/themis_training/mpc_grf_mimic_mdp.py` 以子类方式实现；非 mimic 的 velocity/MPC 任务不会自动
选择该子类。

## 暂存文件逐项摘要

| 路径 | 状态 | 内容与运行时影响 |
|---|---|---|
| `pyproject.toml` | 修改 | 新增 `csv-to-npz-mjlab`、`replay-npz-mjlab` 两个 CLI entry point。 |
| `src/themis_training/mjlab_motion_io.py` | 新增 | CSV/PKL 读取、SLERP/重采样、速度计算、MJLab forward capture、CPU 重建工具及机器人 profile 映射。 |
| `src/themis_training/csv_to_npz_mjlab.py` | 新增 | 将重定向结果转为训练所需命名 NPZ。 |
| `src/themis_training/replay_npz_mjlab.py` | 新增 | MuJoCo 回放与自洽裁剪，裁剪后重建运动学/速度。 |
| `src/themis_training/mimic_mdp.py` | 修改 | 引入命名 motion loader、一次性初始 anchor 对齐、环境原点平移与线速度语义检查。 |
| `src/themis_training/reference_centroidal.py` | 修改 | 支持惯性 CoM 或 link-origin 速度输入，并加入参考的静态坐标对齐函数。 |
| `src/themis_training/export_reference_centroidal.py` | 修改 | 离线导出时验证并保留线速度定义元数据。 |
| `src/themis_training/process_reference_centroidal.py` | 修改 | 转发线速度点语义。 |
| `src/themis_training/process_sim2sim_centroidal.py` | 修改 | 用 MCAP 中 velocity-point 标签选择正确公式。 |
| `src/themis_training/mpc_grf_mimic_mdp.py` | 新增 | 独立的 mimic CD-MPC command、精确在线 centroidal state、reference `x_ref` 和 exact landmark reward。 |
| `src/themis_training/env_cfgs.py` | 修改 | THEMIS mimic 任务显式选择扩展 command；非 mimic 基线仍使用原 MDP。 |
| `src/themis_training/g1/g1_constants.py` | 修改 | 增加 G1 完整 centroidal body set。 |
| `src/themis_training/g1_env_cfgs.py` | 修改 | G1 motion/mimic/hierarchical 配置接入扩展 command、完整 inertial body set 和 exact reward。 |
| `src/themis_training/jingchu01_env_cfgs.py` | 修改 | Jingchu01 相同接入，并配置其足端 site/contact offset。 |
| `src/themis_training/themis/themis_constants.py` | 修改 | 增加 THEMIS 精确总质量及完整 centroidal body set。 |
| `docs/centroidal_dynamics_to_ltv_qp.md` | 新增 | 固定 contact plan 下 centroidal dynamics 到 LTV-QP 的完整推导。 |
| `docs/reference_centroidal_coordinate_contract.md` | 新增 | reference、simulator、MPC 与 reward 的坐标/速度语义契约。 |
| `docs/mjlab_motion_conversion_and_replay.md` | 新增 | 转换、回放、裁剪的输入/输出契约与命令。 |
| `docs/residual_kino_nmpc_pimpc_extension.md` | 新增 | Residual-MPC 到 GPU horizon-parallel kino-dynamic RTI-NMPC 的扩展方案。 |
| `docs/implicit_dynamics_pimpc_pinn_extension.md` | 新增 | 隐式动力学/局部运动学约束的 PiMPC splitting 与 PINN 合理边界。 |
| `docs/online_human_reference_retargeting_mpc_rl.md` | 新增 | 将 GMR 类动作重定向融入 MPC-RL 的设计和研究定位。 |

## 1. 动作数据转换、回放与数据契约

### `mjlab_motion_io.py`

新增 `RetargetedMotion` 与完整的 I/O/运动学工具链。

- **输入读取**：CSV 采用 `[root xyz, root quat_xyzw, joints...]`；PKL 读取 `root_pos`、`root_rot`、
  `dof_pos` 和可选 `fps`。PKL 四元数支持 `auto/wxyz/xyzw`，自动模式选择使初始 upright tilt 更小的
  顺序。
- **重采样**：root 平移、关节角线性插值，root 四元数 shortest-path SLERP；时间采样采用
  `arange(0, duration, 1/fps)`，避免 loop 场景的重复末帧。
- **速度**：root/joint 用有限差分，root angular velocity 由相对四元数对数映射得到。
- **MJLab 捕获**：`capture_motion_through_mjlab_simulator` 将每帧 root/joint 写入 effort articulation，
  调用无积分 `forward()`，读取 entity body pose/velocity。该步骤复现 BeyondMimic 的“写 state → forward
  → 读取 simulator state”语义，而不是进行受重力/接触影响的 rollout。
- **profile**：支持 `themis`、`g1`、`jingchu01`，并从各自 constants 获取实际 joint order 与模型。

转换器保存 `body_linear_velocity_point="inertial_com"`。这很关键：`body_pos_w` 为 link origin，
但 `body_lin_vel_w` 为刚体惯性 CoM 的世界线速度。

### `csv_to_npz_mjlab.py`、`replay_npz_mjlab.py` 与 `pyproject.toml`

`csv_to_npz_mjlab.py` 将上述流程封装为 CLI，额外保存 source 文件、输入格式、四元数顺序、capture
backend、robot profile、joint/body layout 和 temporal processing 元数据。它所输出的 NPZ 可直接被
`MotionReferenceCommandCfg.motion_file` 加载。

`replay_npz_mjlab.py` 提供：

- 以相同 profile 的 MuJoCo viewer 回放 reference；
- 通过 `--trim-frame-range` 裁剪，并重新计算 root/joint/body 运动学，而不是直接截取端点速度。

裁剪路径在 CPU Jacobian 下获得 **link-origin** 线速度，故显式保存
`body_linear_velocity_point="link_origin"`，避免将两种速度定义混用。

注册后的命令为：

```bash
python -m themis_training.csv_to_npz_mjlab INPUT.{csv,pkl} OUTPUT.npz \
  --robot {themis,g1,jingchu01} --output-fps 50 --device cuda:0

python -m themis_training.replay_npz_mjlab OUTPUT.npz --robot jingchu01 --loop
```

### `mimic_mdp.py`

原来的裸 NPZ 加载逻辑被组织成：

- `MJLabMotionClip`：不可变、具名、device-resident trajectory；
- `MJLabMotionLoader`：检查字段、维度、唯一 name、有限数值、四元数范数，并提供严格的 joint/body
  名称索引；
- `MotionReferenceCommand`：在此 clip 上推进 frame，提供 tracker observation/reward，以及 MPC 的
  centroidal/contact horizon。

新增 `reference_frame_alignment`：默认 `initial_anchor`。在加载时一次性移除初始 anchor 的 (x/y)
平移和 yaw，保留高度、roll/pitch、后续 root 运动及完整 body kinematics。它不同于运行时的
`aligned_body_pos_w()`：后者仍是 BeyondMimic 风格的“当前机器人 anchor error”对齐。

对 multi-env，canonical reference 的位置量只在输出时加上 `env_origins`；速度、线动量和关于 CoM 的
角动量不平移。reference centroidal horizon 的 position/contact position 也使用同样规则，保证
reference、simulator 与 CD-MPC 都在同一个世界坐标表达。

## 2. 参考与在线 centroidal quantity

### `reference_centroidal.py`

`compute_reference_centroidal` 现显式接收 `body_linear_velocity_point`。给定 body origin (p_j)、
world orientation (R_j)、body-frame inertial CoM offset (d_j)，有

\[
p_{C,j}=p_j+R_jd_j.
\]

若 NPZ 记录 inertial CoM 速度，直接采用 (v_{C,j}=v_j^{stored})；若记录 origin 速度，则采用

\[
v_{C,j}=v_j^{origin}+\omega_j\times(R_jd_j).
\]

随后质量一致的 quantities 为

\[
c=\frac{1}{m}\sum_jm_jp_{C,j},\qquad
l=\sum_jm_jv_{C,j}=m\dot c,
\]

\[
k_G=\sum_j\left(I_j^W\omega_j+(p_{C,j}-c)\times m_jv_{C,j}\right).
\]

其中 (I_j^W=R_{WB,j}R_{BI,j}I_j^B R_{BI,j}^{\top}R_{WB,j}^{\top})。接触点仍为

\[
r_i=p_{body(i)}+R_{body(i)}d_i^B,\qquad r_i^{rel}=r_i-c.
\]

新增 `compute_centroidal_state` 供 simulator online state 使用；新增
`prealign_reference_kinematics_to_initial_anchor` 对 reference body pose/velocity 做静态 world-frame
canonicalization。其位置/速度/角速度变换保持同一刚体运动学语义。

`export_reference_centroidal.py`、`process_reference_centroidal.py` 和
`process_sim2sim_centroidal.py` 随之传递/验证该元数据。前者把 source velocity-point 写入导出 NPZ；
后者读取 MCAP label，避免误把 origin velocity 当 CoM velocity。

## 3. 独立的 mimic CD-MPC 扩展

### `mpc_grf_mimic_mdp.py`

新增以下三个层次，且均以 `mpc_grf_mdp` 为基类或转发对象：

1. `MimicLocoMPCCommandCfg`：仅当任务显式使用该 cfg 时，manager 才构造 mimic command；
2. `MimicLocoMPCCommand`：继承基线 command 的 contact-plan action、QP 构造、solver、rollout
   storage 和非 mimic API；
3. `MpcExactCentroidalLandmarkTracking`：使用完整 articulated state，而非 root-inertia 近似。

`as_mimic_loco_mpc_cfg` 将基线 cfg 浅复制为扩展 cfg，因此可复用原始 MPC 参数默认值，而不会改变
原始任务实例。

每次 MPC update，扩展 command 从 MuJoCo model 读取 `body_mass`、`body_ipos`、`body_inertia`、
`body_iquat`，结合 entity world body state 计算

\[
x_0=[c^\top,l^\top,k_G^\top]^\top.
\]

若 motion command 提供 `reference_centroidal_horizon(N+1,dt)`，则写入

\[
x_k^{ref}=[(c_k^{ref})^\top,(l_k^{ref})^\top,(k_{G,k}^{ref})^\top]^\top.
\]

Reference contact state 和 contact point position 被传入基线 `make_reference_contact_schedule`；策略的
full-horizon contact residual、QP、(dt) 与基线求解器保持不变。MPC 输出还保存线动量轨迹，用于
`l=m\dot c` 一致的 landmark interpolation。

精确 landmark reward 是

\[
r=\exp\left[-w_c\|c-c^{mpc}\|^2-w_v\|\dot c-\dot c^{mpc}\|^2
-w_l\|l-l^{mpc}\|^2-w_k\|k_G-k_G^{mpc}\|^2\right].
\]

模块覆盖了 `mpc_com_ref` 与 `mpc_ang_mom_ref`；其它未修改 observation/reward helper 通过模块级
`__getattr__` 继续使用基线实现。

## 4. 机器人配置接线

### THEMIS：`themis_constants.py`、`env_cfgs.py`

新增 `THEMIS_TOTAL_MASS=38.33729752907104` 和完整 `THEMIS_CENTROIDAL_BODY_NAMES`。用途是让 CD-MPC
mass、reference centroidal reconstruction 和 online reconstruction 一致；它不同于 pose reward 所需的
稀疏 tracking body set。

`themis_hybrid_mimic_env_cfg` 在建立 effort robot 后，将 `loco_mpc` 显式转换为
`MimicLocoMPCCommandCfg`。其 hierarchical/contact mimic 衍生任务的 critic reference、contact plan、
high-level parameter regularizer 与 future-contact reward 也均改为引用 `mpc_grf_mimic_mdp`。

普通 velocity、MPC-GRF、V2 和 loco-manip 配置仍引用 `mpc_grf_mdp`。同一文件中将两处写死的
`mass=37.0` 改为 `THEMIS_TOTAL_MASS`，消除动力学模型与 XML 质量之间的误差。

### G1：`g1_constants.py`、`g1_env_cfgs.py`

新增 `G1_CENTROIDAL_BODY_NAMES`，覆盖 pelvis、下肢、腰、上肢和 wrist 等全部惯性 body；不再复用仅供
body pose reward 的 `G1_TRACKING_BODY_NAMES`。

G1 Phase-1 contact mimic 配置将基线 `loco_mpc` 转换为 `MimicLocoMPCCommandCfg`，并使用：

- `G1_TOTAL_MASS`、`G1_CENTROIDAL_INERTIA_BODY`；
- initial-anchor reference alignment；
- reference contact schedule 和 full-horizon policy contact-plan residual；
- exact centroidal landmark reward，权重为
  (w_c=4,w_v=1,w_l=0.02,w_k=0.10)。

旧 root-approximation 的 CoM/角动量奖励权重归零，避免与 exact reward 重复。hierarchical G1 任务也
使用 extension module 的 critic landmark/parameter regularizer；student 则按既有逻辑移除 MPC command
与 MPC observation/reward。

### Jingchu01：`jingchu01_env_cfgs.py`

与 G1 使用同一扩展接线。额外保留 Jingchu01 特有：

- `SceneEntityCfg("robot", site_names=JINGCHU01_FEET_SITE_NAMES)`；
- `left_foot_site/right_foot_site`；
- 足端 body origin 到接触点的 (d^B=(0,0,-0.04)) m offset；
- `JINGCHU01_TOTAL_MASS` 和 `JINGCHU01_CENTROIDAL_INERTIA_BODY`。

这确保 reference contact point、MPC contact location 与 foot sensor/site 使用同一机器人配置。

## 5. 新增研究与理论文档（不改变运行时）

下列六份 Markdown 都是设计/理论材料，不会被训练程序自动加载：

| 文档 | 内容 |
|---|---|
| `centroidal_dynamics_to_ltv_qp.md` | 推导冻结 (σ,p,R,\bar c) 时的 (\xi_{k+1}=A\xi_k+B_ku_k+e)，说明 (B_k) 可随 horizon 时变而一次 QP 仍凸。 |
| `reference_centroidal_coordinate_contract.md` | 定义 reference/online/MPC/landmark 的 world axes、CoM definition、线速度语义及环境原点处理。 |
| `mjlab_motion_conversion_and_replay.md` | 解释 CSV/PKL 转换、MJLab capture、回放/裁剪和可复制命令。 |
| `residual_kino_nmpc_pimpc_extension.md` | 从 RTI/SQP 线性化出发，将 kino-dynamic NMPC 转为 GPU batched、parallel-in-horizon local QP split 的方案。 |
| `implicit_dynamics_pimpc_pinn_extension.md` | 对隐式 DAE/kinematic equality 的 stage-local KKT split；明确 PINN 应用于 warm start/residual/uncertainty，而不替换硬动力学约束。 |
| `online_human_reference_retargeting_mpc_rl.md` | 分析 GMR 与 MPC-RL 的三种结合层次，推荐在线、状态条件化 task-space kinodynamic projection，而非把人体 centroidal 数据直接当机器人 reference。 |

## 6. 使用与验证边界

已可在代码层验证的内容：

- motion NPZ schema/name/shape/velocity semantic 的检查；
- 新旧 command cfg 的显式选择；
- G1、Jingchu01、THEMIS mimic config 的构造；
- 固定 schedule/contact position/线性化 CoM 下 CD-MPC 仍是基线 QP。

尚未由当前暂存区完成的内容：

- 全训练实验、sim2sim/real-world 成功率与 ablation；
- chance constraint、接触位置高斯采样闭环与协方差传播；
- kino-dynamic RTI-NMPC/PiMPC 或 PINN world-model 的运行时代码；
- online human task-space projection 的运行时代码。

因此，六份理论文档应被视为下一阶段实现/论文设计依据，而不是当前训练入口已经具备的功能声明。
