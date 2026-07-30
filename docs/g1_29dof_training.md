# Unitree G1-29DOF 训练接入

本仓库现在提供与 THEMIS 完全分离的 Unitree G1-29DOF 配置。G1 任务不会复用
THEMIS 的关节名、足底碰撞几何、PD 参数、质量或近似转动惯量。

## 新增文件与任务

- `src/themis_training/g1/xmls/g1.xml` 及同目录 `assets/`：G1 29-DOF MJCF
  和网格资产。
- `src/themis_training/g1/g1_constants.py`：29 关节顺序、Unitree 电机等效
  armature / PD、位置与 effort 两套 articulation、足底碰撞、初始姿态，以及
  质心 MPC 参数。
- `src/themis_training/g1_env_cfgs.py`：G1 平地速度、motion tracker、第一阶段
  MPC-RL mimic-contact teacher、第二阶段 causal student 环境。

已注册的 task id：

| Task | 用途 | 动作 |
|---|---|---|
| `Mjlab-Velocity-G1-29DOF` | 不依赖参考动作的 G1 速度基线 | 29D 位置目标 |
| `Mjlab-MotionTracker-G1-29DOF` | BeyondMimic 风格 tracker | 29D 位置目标 |
| `Mjlab-MPC-RL-Mimic-Contact-G1-29DOF` | 阶段一 teacher | 29D `q_des` residual + `2H` 接触计划 residual |
| `Mjlab-Hierarchical-HybridMimic-MPC-G1-29DOF` | 阶段一的分层扩展 | 上述动作 + 慢尺度 16D MPC 参数 |
| `Mjlab-MPC-RL-Mimic-Student-G1-29DOF` | 阶段二 DAgger/PPO student | 29D `q_des` residual |

## 机器人与 MPC 一致性

`G1_JOINT_NAMES` 按 MJCF 的关节顺序为左右腿各 6、腰部 3、左右臂各 7，共
29 个。地面接触传感器聚合 `left_ankle_roll_link`、`right_ankle_roll_link`
子树；两只脚的 MPC site 是 `left_foot`、`right_foot`。

G1 MJCF 内所有 `<inertial mass>` 的和为

\[
m_{G1}=33.341142\;\mathrm{kg}.
\]

该值写入 G1 的 `LocoMPCCommandCfg.mass`，不再沿用 THEMIS 的 37 kg。
`LocoMPCCommandCfg` 新增 `inertia_body`，G1 使用

\[
I_{B,0}=\operatorname{diag}(1.20, 1.45, 0.75)\;\mathrm{kg\,m^2}.
\]

它只用于把测得的 base angular velocity 转换成 MPC 初始角动量近似
\(k_0\approx R I_{B,0}R^T\omega\)。这不是完整的构型相关 centroidal inertia；
进行定量实验前，建议通过 MuJoCo/Pinocchio 在代表性姿态上计算复合刚体惯量并
替换该常量。参考轨迹的 \(c,l,k\) 仍由 `reference_centroidal.py` 的各刚体质量、
速度和惯量公式计算。

## 参考动作要求

`MotionReferenceCommand` 严格要求 NPZ 的关节通道对应 G1 关节名，身体通道至少
含 `G1_TRACKING_BODY_NAMES`。它包括 `pelvis`、左右 hip-roll/knee/ankle-roll、
`torso_link`、左右 shoulder-roll/elbow/wrist-yaw。不能直接把 THEMIS retarget
动作送入 G1 task；应先依据该映射得到 G1 retarget 的 NPZ。

对于 MPC-RL task，reference 中还应带由本仓库离线工具得到的 centroidal 与参考
接触计划字段。接触位置取参考的 `left_foot` / `right_foot`，而 QP 中的 \(dt\)
始终固定为 `0.07 s`，故接触计划在 QP 装配前冻结，保持凸 QP。

## 启动方式

使用仓库原有的训练入口，只把 task 名替换为上表的值。例如先运行脚本的帮助以核
对当前参数名：

```bash
cd /home/user/wmd/mpc-rl
uv run train --help
```

注册表不硬编码个人动作路径。对于命令行训练，可用 `THEMIS_G1_MOTION_FILE` 指向
G1 retarget 的 NPZ：

```bash
cd /home/user/wmd/mpc-rl
THEMIS_G1_MOTION_FILE=/absolute/path/to/g1_retargeted_motion.npz \
  CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MotionTracker-G1-29DOF \
  --env.scene.num-envs 4096 --agent.max-iterations 15000

THEMIS_G1_MOTION_FILE=/absolute/path/to/g1_retargeted_motion.npz \
  CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MPC-RL-Mimic-Contact-G1-29DOF \
  --env.scene.num-envs 4096 --agent.max-iterations 15000
```

也可以在 Python 中直接调用工厂并传入 `motion_file=`。若环境变量设置了不存在的
路径，配置构造会立即报错，避免训练中途才发现参考动作错误。

## 已知边界

G1 的 ankle/waist 是并联机构；当前采用上游 G1 配置的双 5020 电机 1:1 名义等效
惯量与增益。它适合仿真起步，但不是 transmission 随构型变化的精确模型。真实机
或高保真 sim2real 前，应以标定后的传动比、摩擦、力矩-速度曲线与 composite
inertia 替换这些近似。
