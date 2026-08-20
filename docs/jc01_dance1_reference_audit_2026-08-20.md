# Jingchu01 `dance1_subject2` 训练前参考审计

## 结论

**不要采用 50 Hz 重定向运动学推断的接触日程。** 现已按“source BVH 接触、retargeted
robot centroidal reference”的方案导出 `reference_contact_state`；它可作为固定 nominal MPC
schedule 进行实验，但其与机器人足端运动学明显不一致，必须作为显式的 contact-transfer
假设记录和消融。仍缺少 Jingchu01 MuJoCo XML 资产，故尚无法计算可用于论文/训练审计的
精确参考 CoM、线动量和角动量。

这不是 MPC 频率的问题。mimic 仍必须保留 50 Hz 的 joint reference；MPC 已在
`MotionReferenceCommand.horizon_frame_progress()` 中按

\[
s_i=s_0+i\,f_{ref}\Delta t_{MPC},\qquad
f_{ref}=50\ \mathrm{Hz},\quad\Delta t_{MPC}=0.07\ \mathrm{s}
\]

连续采样，所以每个 MPC node 跨 3.5 个 reference frame。不要把训练用 clip 替换为
14.29 Hz 文件；那会损失 mimic 所需的关节时序。离线的 14.29 Hz grid 用于审计、接触
标签验证和复现实验。

## 已审计文件

| 文件 | 角色 | 时间轴 |
|---|---|---|
| `ref/dance1_subject2.bvh` | LaFAN1 原始人类动作；由 `LeftFoot`/`RightFoot` FK 推断接触 | 3944 frames，30.0003 Hz |
| `ref/dance1_subject2.pkl` | Jingchu01 重定向后的 root/28 DOF 轨迹 | 3944 frames，30 Hz |
| `ref/jingchu01_fullbody_dance1_subject2_full2.npz` | 50 Hz whole-body mimic 输入 | 6572 frames，50 Hz |

PKL 的 root trajectory 与 50 Hz NPZ 中 `Robotbase` 的位置范围一致，说明 NPZ 确实来自
该重定向片段的时间重采样/whole-body export；两个动作长度也都约为 131.4 s。

## 接触结果（MPC grid，0.07 s）

规则与 `MimicLocoMPCCommand._load_or_infer_reference_contact_state()` 一致：每只脚相对
自身整段最低高度不超过 0.03 m、中心差分速度不超过 0.35 m/s，随后填补不超过 2 帧的
swing gap、删除不超过 3 帧的 stance run。

| 指标 | 左脚 | 右脚 |
|---|---:|---:|
| 重定向 JC01 stance 占比 | 0.00% | 0.43% |
| 原始 BVH stance 占比 | 42.39% | 0.96% |
| 两者逐 node 一致率 | 57.61% | 98.62% |
| 两脚同时一致率 | 56.71% | |

图已输出：

- [接触序列比较图](/home/edify/Code/mpc-rl/ref/audit_dance1_subject2/contact_sequence_comparison.png)
- [重定向 centroidal 代理图（不可用于 MPC）](/home/edify/Code/mpc-rl/ref/audit_dance1_subject2/retargeted_centroidal_proxy_not_for_mpc.png)
- [可复现摘要](/home/edify/Code/mpc-rl/ref/audit_dance1_subject2/audit_summary.json)

从图和数值看，重定向后的左脚在其最低高度附近的速度仍大于约 1.29 m/s；它不满足
“地面固定支撑”假设。右脚也几乎没有稳定支撑。这可能来自 contact-unaware retargeting、
ankle-roll 到真实 sole 的偏移不正确、reference 坐标/地面未对齐，或该片段本身含有较长
飞行段。因此本实验选择以 source BVH 的动作语义接触作为 nominal schedule，而不是将
retargeted kinematic heuristic 作为接触真值；二者差异本身需要在论文中报告。

## 已添加的工具与产物

新增 `audit-mimic-reference`（实现位于
`src/mjlab_tools/audit_mimic_reference.py`）。它创建而不覆盖原始 clip：

```bash
PYTHONPATH=src audit-mimic-reference \
  ref/jingchu01_fullbody_dance1_subject2_full2.npz \
  ref/dance1_subject2.bvh \
  ref/audit_dance1_subject2 --mpc-dt 0.07
```

新文件 `jingchu01_fullbody_dance1_subject2_full2_with_reference_contacts.npz` 保留全部 50 Hz
mimic 通道，并额外包含：

- `retargeted_contact_state`：由重定向运动学得到的 50 Hz 标签；
- `source_bvh_contact_state`：原 BVH 标签重采样至相同 50 Hz 时间轴；
- `mpc_retargeted_contact_state`、`mpc_source_bvh_contact_state`：0.07 s 审计 grid；
- `mpc_reference_frame_progress`：证明每个 MPC node 对应的连续 50 Hz frame 坐标。

`reference_contact_state` 现在默认写入 `source_bvh_contact_state`：即从重定向前的人体 BVH
提取、重采样至同一 50 Hz 时间轴的 nominal MPC schedule。`retargeted_contact_state` 仍完整
保留，仅用于诊断其与 source schedule 的不一致；CoM、线动量和角动量仍必须由重定向后的
机器人运动和匹配机器人模型计算。

## 精确 centroidal 的硬性前置条件

`src/jingchu01_training/jingchu01/xmls/jingchu01.xml` 及 meshes 当前不在工作区（被
`.gitignore` 排除），而 Python 环境也没有 MuJoCo。没有以下模型资产，就不能获得

\[
c=\frac{1}{m}\sum m_jp_{C,j},\quad l=\sum m_jv_{C,j},\quad
k_G=\sum\left(I_j^W\omega_j+(p_{C,j}-c)\times m_jv_{C,j}\right).
\]

现有代理图仅采用 uniform-body CoM 和 uniform-mass linear-momentum proxy；它用于发现
时间轴或数值异常，**不能**作为 MPC reference，且不提供角动量。

恢复与 NPZ 相匹配的 MuJoCo XML/mesh 后，必须运行：

```bash
PYTHONPATH=src process-reference-centroidal \
  ref/audit_dance1_subject2/jingchu01_fullbody_dance1_subject2_full2_with_reference_contacts.npz \
  <matched-jingchu01.xml> \
  ref/audit_dance1_subject2/exact_centroidal.npz \
  --contact left_sole_center=left_ankle_roll:0,0,-0.04 \
  --contact right_sole_center=right_ankle_roll:0,0,-0.04
PYTHONPATH=src plot-reference-centroidal \
  ref/audit_dance1_subject2/exact_centroidal.npz \
  ref/audit_dance1_subject2/exact_centroidal_figures
```

运行前还须确认 XML body names、inertial frame、sole-site offset 与 NPZ 的 body names 完全匹配。

## 训练配置与当前研究计划的一致性

1. Jingchu01 multi-critic 现在采用固定 actor-advantage 系数
   `(mpc_landmark=1.5, mimic=1.0, task=1.0, regularization=1.0)`；这只是在当前阶段强调
   MPC landmark，未声称自适应融合。
2. `THEMIS_JINGCHU01_REFERENCE_CONTACT_KEY` 是显式 opt-in。若采用本文的 source-motion
   方案，设置为 `reference_contact_state`（或等价的 `source_bvh_contact_state`）；运行时仍在
   50 Hz robot clip 上采样，并以 `mpc_dt` 形成 horizon。source schedule 是语义先验而非
   真值接触，因此必须报告其与机器人足端运动学/真实接触的偏差。
3. 若你要验证“固定 motion-derived schedule”的核心研究假设，应使用
   `Mjlab-MPC-RL-Mimic-Reference-MultiCritic-Jingchu01-28DOF`，而不是带 policy
   contact-plan residual 的 `...Mimic-Contact-MultiCritic...`。
4. XML 恢复并完成精确 centroidal 图审计后，才可开始本方案的 MPC mimic training。训练中
   应固定 source-BVH schedule，并以 retargeted schedule/实际仿真接触作为诊断与消融；不要
   将两者不一致隐藏为“接触真值”。
