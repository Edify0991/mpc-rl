# G1 / Jingchu01 训练任务的包级隔离

## 目标与边界

`themis_training` 现在只注册 THEMIS 原论文对应的三个任务：速度命令
locomotion、motion tracker，以及带箱子推拉的 loco-manipulation。G1 与
Jingchu01 不再从这里注册；它们各自是独立的 `mjlab.tasks` entry point，
并且每一种机器人在同一包内将**原论文任务**和后续的**参考动作 mimic
任务**分开。

| 机器人 | 训练包 | 纸面/原始任务 | 后续 mimic 任务 |
| --- | --- | --- | --- |
| THEMIS | `themis_training` | `Mjlab-MPC-Guided-Locomotion-Themis`、`Mjlab-MPC-Guided-Loco-manipulation-Themis` | 无：mimic 验证仅在 G1/Jingchu01 |
| G1-29DOF | `g1_training` | `Mjlab-Velocity-G1-29DOF`、`Mjlab-MPC-Guided-Locomotion-G1-29DOF`、`Mjlab-MPC-Guided-Loco-manipulation-G1-29DOF` | `MotionTracker`、`MPC-RL-Mimic-Contact`、`Student`、`Hierarchical-HybridMimic` 的 G1 ID |
| Jingchu01-28DOF | `jingchu01_training` | `Mjlab-Velocity-Jingchu01-28DOF`、`Mjlab-MPC-Guided-Locomotion-Jingchu01-28DOF`、`Mjlab-MPC-Guided-Loco-manipulation-Jingchu01-28DOF` | 同名 Jingchu01 的四个 mimic ID |

`pyproject.toml` 为三者声明了单独的 `mjlab.tasks` entry point。安装或
执行 `uv sync` 后，mjlab 会发现这些包。旧的
`themis_training.g1_env_cfgs`、`themis_training.jingchu01_env_cfgs` 及其
机器人资产目录已经删除。

## 文件职责

```text
src/
  themis_training/             # 仅 THEMIS 私有任务、MPC MDP 与 runner cfg
  g1_training/
    g1/g1_constants.py         # G1 MJCF、关节、执行器和 centroidal 参数
    g1/xmls/                   # G1 专属资产（与 themis/xmls 同一层级惯例）
    env_cfgs.py                # G1 原始任务与 mimic 工厂
    mpc_grf_mdp.py             # G1 训练 command、MPC landmark/GRF MDP
    mpc_grf_mimic_mdp.py       # G1 mimic 专用 MPC command/landmark MDP
    rl_cfg.py                  # G1 私有 PPO 网络与优化参数
    __init__.py                # 注册 G1 IDs（entry point）
  jingchu01_training/
    jingchu01/jingchu01_constants.py # JC01 专属模型与动力学参数
    jingchu01/xmls/            # JC01 专属资产（与 themis/xmls 同一层级惯例）
    env_cfgs.py                # JC01 原始任务与 mimic 工厂
    mpc_grf_mdp.py             # JC01 训练 command、MPC landmark/GRF MDP
    mpc_grf_mimic_mdp.py       # JC01 mimic 专用 MPC command/landmark MDP
    rl_cfg.py                  # JC01 私有 PPO 网络与优化参数
    __init__.py                # 注册 Jingchu01 IDs（entry point）
  training_common/             # 无机器人名称的纯数学/reference 与配置装配
  mjlab_tools/                 # CSV/PKL 转换、回放、离线 centroidal 分析 CLI
```

参考 centroidal 数学与 MPC landmark 配置装配位于 `training_common`；离线转换/
回放脚本位于 `mjlab_tools`。Mimic、phase、push-box、PPO runner 与 finetune 均在
对应机器人包中，G1/JC01 不会调用 THEMIS 的训练 MDP 或 runner 参数。

## 原论文 locomotion 的机器人化移植

`g1_mpc_locomotion_env_cfg` 与
`jingchu01_mpc_locomotion_env_cfg` 复用原始的 velocity-command、固定步长
CD-MPC 和 MPC landmark reward 结构，但会将 `loco_mpc` 替换成对应机器人
包中的 `LocoMPCCommandCfg`：

\[
  m \leftarrow m_{robot},\qquad
  I_B \leftarrow I_{B,robot},\qquad
  (p_L,p_R) \leftarrow (p_{L,robot},p_{R,robot}).
\]

因此 QP 的状态、代价、固定时域步长和原论文的 landmark 接口不变，而接触
几何、质心质量和初始角动量近似不再错用 THEMIS 数值。参考动作 command、
reference-contact schedule、teacher contact action 仅在 mimic 工厂中打开，
不会污染上述 paper-compatible locomotion 工厂。

## Loco-manipulation 接触语义

两个机器人都注册了原始 THEMIS 推箱任务的机器人化版本。它们保留箱子、
50/50 walk–push reset、手部接触奖励、箱子 CoM landmark、手力 centroidal
MPC 及跌倒/箱子倾倒终止，但显式替换手接触点：

| 机器人 | MPC 手接触点 | 接触几何 |
| --- | --- | --- |
| G1 | `left_palm`, `right_palm` | `left_hand_collision`, `right_hand_collision` |
| Jingchu01 | `left_wrist_contact`, `right_wrist_contact` | `left_wrist_roll_collision_0`, `right_wrist_roll_collision_0` |

G1 的 palm site 已存在于原模型。Jingchu01 没有手掌/末端执行器链接；因此
其 XML 中增加的两个 `*_wrist_contact` site **严格位于 wrist-roll 原点**。
这使当前任务可运行且模型假设可见，但它不是“虚构的手掌”。若后续加入真实
夹爪、手掌或工具，应移动 site 到标定后的接触面并重新检查 `r_{LH},r_{RH}`：

\[
  \dot k = \sum_{i\in\{L,R\}} (p_i-c)\times f_i
          + \sum_{h\in\{LH,RH\}}(p_h-c)\times f_h .
\]

手接触位置误差会直接改变该力矩臂；这不是可由 reward 权重补偿的误差，故不
应把 wrist-origin 模型用于需要精确手掌动力学的实验结论。

## 运行

完成安装/同步后可按 ID 训练。例如：

```bash
uv run mjlab-train Mjlab-MPC-Guided-Locomotion-G1-29DOF
uv run mjlab-train Mjlab-MPC-Guided-Loco-manipulation-Jingchu01-28DOF
uv run mjlab-train Mjlab-MPC-RL-Mimic-Contact-G1-29DOF
```

命令名请以本仓库当前 mjlab 安装实际提供的训练 CLI 为准；这里的重点是
任务 ID 和 entry point，不要求通过 THEMIS 包加载 G1/JC01 配置。
