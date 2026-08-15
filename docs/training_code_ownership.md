# 训练代码归属与机器人隔离

## 结论

训练代码按**物理语义**而不是按文件数量划分：凡是会改变某个机器人的状态、
动作、奖励尺度或优化超参数的内容，必须由该机器人训练包拥有；不依赖机器人
模型的纯数学工具可保留一份公共实现。MDP、action、奖励和训练流程即使当前
实现相同，也按你的实验边界保存在对应机器人包中。

## 当前目录

```text
src/
  themis_training/       # THEMIS 私有 env / MPC MDP / PPO cfg / task entry point
  g1_training/           # G1 私有 env / constants / PPO cfg / task entry point
  jingchu01_training/    # JC01 私有 env / constants / PPO cfg / task entry point
  themis_mpc/            # THEMIS MPC 求解器
  g1_mpc/                # G1 MPC 求解器及 mimic 扩展
  jingchu01_mpc/         # JC01 MPC 求解器及 mimic 扩展
  training_common/       # 纯数学 reference/无状态配置装配
  mjlab_tools/           # 可独立执行的离线工具
```

每个训练包直接在 `__init__.py` 注册任务，这与原仓库的注册风格一致，并且
`pyproject.toml` 的 `mjlab.tasks` entry point 指向各自包根。

## 必须机器人私有的内容

- `constants.py` 与机器人命名子目录下的 `xmls/`：关节顺序、执行器增益/限幅、质量、惯量、site、geom、
  初始姿态。
- `env_cfgs.py`：观测中使用的关节/刚体、动作维度、接触传感器、奖励参数及
  任务注册引用。
- `rl_cfg.py`：actor/critic 宽度、PPO 熵系数、rollout 长度、迭代数、实验名。
  G1 和 JC01 现在各有独立文件，即使当前数值相同，也可无耦合地调参。
- `*_mpc/`：接触 schedule、MPC command、质量/惯量与模型特有的 mimic 适配。
- `phase_mdp.py`、`push_box_mdp.py`、`finetune.py`：THEMIS 保留原基线，G1 与
  Jingchu01 都有各自副本。
- `mimic_mdp.py`、`hybrid_mimic.py`、`dagger_distillation.py`、
  `mpc_parameter_net.py`：只位于 G1/Jingchu01，因为 mimic 实验不再使用 THEMIS。

## 应共享而不复制的内容

`training_common` 只保留不构成 RL task 的内容：

- `reference_centroidal.py`：由质量、惯量、位姿和速度计算 CoM/动量的公式。
- `mpc_locomotion_features.py`：原论文的 MPC landmark reward 结构；robot package
  显式传入自己的 `LocoMPCCommandCfg`、MPC MDP 和足端 site。

这保证 THEMIS 原任务可独立复现，而 G1/Jingchu01 的改动不会写回它的 MDP
或训练配置；仍只共享动量计算等纯数学定义。

## 离线工具

`mjlab_tools` 包含 CSV/PKL 转 NPZ、NPZ 回放、reference centroidal 处理、MCAP
sim2sim 重建与作图。其可执行入口保持原名称：

```bash
uv run csv-to-npz-mjlab --help
uv run replay-npz-mjlab --help
uv run process-reference-centroidal --help
uv run process-sim2sim-centroidal --help
```

因此工具不再假定 “THEMIS training package”，但 `--robot` 仍明确选择
`themis`、`g1` 或 `jingchu01`，由相应的模型配置解析。
