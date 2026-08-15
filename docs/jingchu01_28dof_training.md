# Jingchu01 28-DOF 训练接入

本仓库新增了与 THEMIS/G1 隔离的 Jingchu01 28-DOF 训练配置。资产来自
`/home/user/wmd/jc01-model`，并复制到
`src/jingchu01_training/jingchu01/xmls/`，因此训练不依赖该工作区外的资产路径。

## 关节、根链接和接触约定

Jingchu01 有 28 个受控关节：左右腿各 6、腰部 2、左右臂各 7。参考动作
`assets/ref_motion/smpl_jc01_taichi1.npz` 使用根名称 `Robotbase`，而源 XML 使用
`Body`。本仓库的 XML 副本将根 body 重命名为 `Robotbase`，同时更新 contact exclude
与 subtree-angular-momentum sensor，因此 motion tracker 不需要隐式的名称别名。

接触顺序固定为 **left, right**：

| MPC contact | Site | body-frame point |
|---|---|---|
| left | `left_foot_site` | `left_ankle_roll + (0,0,-0.04)` m |
| right | `right_foot_site` | `right_ankle_roll + (0,0,-0.04)` m |

`LocoMPCCommandCfg` 新增 `left_foot_site_name` / `right_foot_site_name`，默认仍为
THEMIS/G1 的 `left_foot/right_foot`；Jingchu01 显式设置为自己的 site 名。因此 QP
内部接触向量始终是左脚后右脚，不受 MJCF 命名影响。

MJCF 的所有 inertial body 质量和为

\[
m_{JC01}=57.00294\;\mathrm{kg}.
\]

该值用于 CD-MPC。当前初始角动量转换使用 MuJoCo 在 `STANDING_KEYFRAME` 计算的
复合刚体惯量（关于系统 CoM）：

\[
I_{B,0}=\begin{bmatrix}
8.686559 & -0.000549 & 0.285403\\
-0.000549 & 7.616278 & 0.000288\\
0.285403 & 0.000288 & 1.642220
\end{bmatrix}\;\mathrm{kg\,m^2}.
\]

它显著优于任意手工对角近似，但仍不是构型相关的完整 centroidal inertia；高保真或
sim2real 实验前仍应使用 CRBA/MuJoCo 在线或分段标定替换。

## 注册任务

| Task ID | 动作空间 |
|---|---|
| `Mjlab-Velocity-Jingchu01-28DOF` | 28D position target |
| `Mjlab-MPC-Guided-Locomotion-Jingchu01-28DOF` | 原 THEMIS velocity-command CD-MPC 的 JC01 端口 |
| `Mjlab-MPC-Guided-Loco-manipulation-Jingchu01-28DOF` | 原 THEMIS 推箱 loco-manipulation 的 JC01 端口（wrist-origin 接触） |
| `Mjlab-MotionTracker-Jingchu01-28DOF` | 28D position target |
| `Mjlab-MPC-RL-Mimic-Contact-Jingchu01-28DOF` | 28D joint residual + `2H` contact-plan residual |
| `Mjlab-Hierarchical-HybridMimic-MPC-Jingchu01-28DOF` | 上述动作 + 慢尺度 16D MPC 参数 |
| `Mjlab-MPC-RL-Mimic-Student-Jingchu01-28DOF` | 28D causal joint residual |

## 启动

`THEMIS_JINGCHU01_MOTION_FILE` 必须指向具备 named `joint_names/body_names` 的
BeyondMimic NPZ。仓库中已有可用例子：

```bash
cd /home/user/wmd/mpc-rl
THEMIS_JINGCHU01_MOTION_FILE=$PWD/assets/ref_motion/smpl_jc01_taichi1.npz \
  CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MotionTracker-Jingchu01-28DOF \
  --env.scene.num-envs 4096 --agent.max-iterations 15000

THEMIS_JINGCHU01_MOTION_FILE=$PWD/assets/ref_motion/smpl_jc01_taichi1.npz \
  CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-MPC-RL-Mimic-Contact-Jingchu01-28DOF \
  --env.scene.num-envs 4096 --agent.max-iterations 15000
```

对于第二条命令，建议先用已有离线脚本生成同一动作的 centroidal/contact 诊断结果；训练
环境会依据同一 MJCF 与命名动作在线重建 \(c,l,k\) 和参考接触计划。
