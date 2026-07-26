# JC01 BeyondMimic sim2sim 与参考动作的 centroidal 对比

本流程对同一段太极动作的两类数据采用同一套质量与惯量模型：

- 参考动作：`assets/ref_motion/smpl_jc01_taichi1.npz`（3999 帧、50 Hz）；
- BeyondMimic sim2sim 实际机器人状态：
  `assets/ref_motion/processed/Jul21_14-14-00_beyondmimic_jc01_dance_wo_state_estimation_policy_running_frames.mcap`；
- MuJoCo 模型：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/scene_jingchu01.xml`；
- URDF provenance：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF.urdf`。

这里的 sim2sim 文件不是 policy observation，而是每个控制 tick 的实际根状态、实际关节状态及足端力
传感器标签。因此它用于评估策略的实际动力学轨迹，而不是再次计算参考轨迹。

## 计算方法

每个 MCAP frame 给出 root pose、root velocity、`joint_q`、`joint_dq`。其中该记录的
`base_state.tags.velocity_source` 是 `body_object_velocity_root_local`，所以根的线速度和角速度先由

\[
v_B^W=R_{WB}v_B^B,\qquad \omega_B^W=R_{WB}\omega_B^B
\]

转换至世界系。28 维 `joint_q`/`joint_dq` 使用 NPZ 的 `source_joint_names`（即 MuJoCo/deployment
顺序），**不能**使用 policy 的重排 `joint_names`。随后把 \((q,\dot q)\) 写入 XML 的
`root_freejoint` 与各关节地址，调用 `mj_forward`。对每个 body 原点 `xpos`，脚本以
\(v_j=J_{p,j}(q)\dot q\)、\(\omega_j=J_{\omega,j}(q)\dot q\) 取刚体速度。这里不能直接使用
MuJoCo 的 `data.cvel`：它对应惯性中心，而后续公式已经显式从 body 原点平移到惯性 CoM。这保证
sim2sim 与参考处理采用相同的刚体质量、惯性偏置和惯量，并避免重复平移速度。

对质量 \(m_j\)、body-frame inertia-CoM 偏移 \(d_j^B\)、世界惯量 \(I_j^W\)，计算

\[
p_j^{com}=p_j+R_jd_j^B,\qquad
v_j^{com}=v_j+\omega_j\times(R_jd_j^B),
\]

\[
c=\frac{1}{M}\sum_jm_jp_j^{com},\qquad
l=\sum_jm_jv_j^{com},
\]

\[
k=\sum_j\left(I_j^W\omega_j+
(p_j^{com}-c)\times m_jv_j^{com}\right),\qquad M=\sum_jm_j.
\]

左右接触点仍是 ankle-roll body 上的脚板中心：

\[
r_i=p_{body(i)}+R_{body(i)}(0,0,-0.04)^T,
\qquad r_i^{rel}=r_i-c.
\]

sim2sim 的 `contact_state` 由 MCAP 中连续的 `contact_force_norm_n` 重算，而不使用记录器的
`foot_contact.{left,right}.in_contact` 布尔量。默认阈值为 \(\lVert f_{foot}\rVert\ge100\) N；原始
20 N logger label 仍以 `logged_contact_state` 保存，用于溯源或改变阈值后复核。

## 坐标和时间对齐

reference NPZ 与 sim2sim 的世界原点、初始偏航是任意的，直接比较世界位置没有物理意义。因此只对
sim2sim 施加**一次、时间不变**的初始根 SE(2) 配准：

\[
R_a=R_z(\psi^{ref}_0-\psi^{sim}_0),\qquad
p^a=R_a(p^{sim}-p^{sim}_{root,0})+p^{ref}_{root,0}.
\]

速度、线动量与角动量仅旋转：\(v^a=R_av^{sim}\)、\(l^a=R_al^{sim}\)、
\(k^a=R_ak^{sim}\)。这不是逐帧拟合，不会隐藏 tracking error；它只消除了 global-frame gauge。

MCAP 有 15,989 帧、79.9798 s（约 199.90 Hz），reference 有 79.96 s 但只有 3,999 帧。绘图和指标
都将两者在线性 normalized motion progress \([0,1]\) 上重采样到共同的 1,600 个点；接触布尔标签用
最近样本而不是线性插值。

## 复现命令

先生成 reference centroidal 文件（若它还不存在），命令见
[参考处理说明](jc01_reference_centroidal_processing.md)。然后在仓库根目录运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /home/user/anaconda3/envs/wmd_ampmjlab/bin/python \
  -m themis_training.process_sim2sim_centroidal \
  assets/ref_motion/processed/Jul21_14-14-00_beyondmimic_jc01_dance_wo_state_estimation_policy_running_frames.mcap \
  assets/ref_motion/smpl_jc01_taichi1.npz \
  /home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/scene_jingchu01.xml \
  assets/ref_motion/processed/jc01_taichi1_beyondmimic_sim2sim_centroidal.npz \
  --urdf /home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF.urdf \
  --body-map Robotbase=Body \
  --contact left_sole_center=left_ankle_roll:0,0,-0.04 \
  --contact right_sole_center=right_ankle_roll:0,0,-0.04

MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src \
  /home/user/anaconda3/envs/wmd_ampmjlab/bin/python \
  -m themis_training.plot_sim2sim_centroidal_comparison \
  assets/ref_motion/processed/smpl_jc01_taichi1_centroidal.npz \
  assets/ref_motion/processed/jc01_taichi1_beyondmimic_sim2sim_centroidal.npz \
  assets/ref_motion/processed/jc01_taichi1_beyondmimic_comparison_figures \
  --samples 1600
```

`MPLCONFIGDIR=/tmp/matplotlib` 只为当前受限运行环境提供 Matplotlib cache；在用户本机有可写的
Matplotlib cache 时可省略。当前 mjlab 实验环境已经包含 `zstandard`（MCAP chunk 使用 zstd 压缩）；若在
一个精简环境中重现，需额外安装 `zstandard==0.25.0`。

## 本次生成的文件与结果

- `assets/ref_motion/processed/jc01_taichi1_beyondmimic_sim2sim_centroidal.npz`：对齐后的
  \(c,\dot c,l,k,r,r-c\)、原始 CoM/接触位置、时间、frame id、接触标签、足力范数和完整 provenance；
- 同名 `.json`：帧数、时长、质量、坐标配准参数；
- `assets/ref_motion/processed/jc01_taichi1_beyondmimic_comparison_figures/`：PNG/PDF 图和
  `centroidal_comparison_metrics.json`。

共同质量为 57.00294 kg。此次结果的向量 RMSE 为：CoM position 0.13782 m、CoM velocity
0.04079 m/s、linear momentum 2.32525 kg m/s、angular momentum 0.73941 kg m²/s。以连续足端力范数
\(\ge100\) N 重算后，左右足 contact 相对 reference 运动学 schedule 的 agreement 分别为 91.25% 与
76.44%，stance 占比分别为 95.69% 与 92.31%。虽然 100 N 已明显优于记录器原始 20 N 标签，但该段
sim2sim 动作仍以双足载荷为主；不能将阈值标签直接解释为 reference schedule 的高精度预测。

图包括：

- `centroidal_reference_sim2sim_components`：四个 centroidal quantity 的 x/y/z 分量；黑虚线为
  reference，彩色实线为 sim2sim；
- `centroidal_error_and_contact`：四个向量范数误差，以及 reference nominal schedule 与 MCAP
  真值接触标签；
- `com_reference_sim2sim_spatial`：CoM 的水平及三维空间轨迹。

## MCAP 兼容性说明

该 runtime 文件的 Chunk record 缺少标准 MCAP 的 `records` 长度字段，所以严格 `mcap` Python reader
会拒绝它。`mcap_policy_frames.py` 明确支持这种 legacy chunk 和标准 chunk 两种布局，但只读取本项目
所需的 zstd 压缩 JSON `policy_running_frame` topic；它不是通用 MCAP parser。后续 recorder 应修正
Chunk writer 以输出标准长度字段，届时该脚本仍可直接读取。
