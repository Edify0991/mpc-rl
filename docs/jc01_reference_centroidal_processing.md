# JC01 `smpl_jc01_taichi1` 离线 centroidal 处理与可视化

本流程针对：

- reference：`assets/ref_motion/smpl_jc01_taichi1.npz`；
- MuJoCo runtime model：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/scene_jingchu01.xml`；
- URDF provenance：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF.urdf`。

数值惯性参数以 XML 为准，因为 MuJoCo 编译后实际使用的是 XML 的 `body_mass`、`body_ipos`、
`body_inertia`、`body_iquat`。URDF 路径会写入输出 metadata 以保留模型来源，但不单独参与数值
计算。

## 已核对的名称和接触点

reference 有 29 个 body。除根 body 外，28 个 body 名称和 XML 完全一致；唯一映射为：

```text
Robotbase -> Body
```

左右候选接触点分别定义在 reference 的 `left_ankle_roll` 和 `right_ankle_roll` 上，局部偏移为

```text
left_sole_center  = left_ankle_roll:  (0, 0, -0.04) m
right_sole_center = right_ankle_roll: (0, 0, -0.04) m
```

该偏移与 XML 中 `left_foot_site` / `right_foot_site` 的局部位置一致。它是每脚一个中心候选点；
不能代表 toe/heel/edge patch contact。

## 一次性处理命令

在仓库根目录运行：

```bash
PYTHONPATH=src /home/user/anaconda3/envs/wmd_ampmjlab/bin/python \
  -m themis_training.process_reference_centroidal \
  assets/ref_motion/smpl_jc01_taichi1.npz \
  /home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/scene_jingchu01.xml \
  assets/ref_motion/processed/smpl_jc01_taichi1_centroidal.npz \
  --urdf /home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/JC01-URDF.urdf \
  --body-map Robotbase=Body \
  --contact left_sole_center=left_ankle_roll:0,0,-0.04 \
  --contact right_sole_center=right_ankle_roll:0,0,-0.04 \
  --height-threshold 0.03 \
  --speed-threshold 0.35 \
  --min-stance-frames 3 \
  --min-swing-frames 2
```

输出：

- `smpl_jc01_taichi1_centroidal.npz`：CoM、CoM velocity、(l)、(k)、接触点、相对 CoM
  向量、接触高度/速度以及平滑接触标签；
- `smpl_jc01_taichi1_centroidal.json`：质量、名称映射、阈值、来源模型等可复现 metadata。

接触候选规则为：接触点接近该脚在整段 clip 内的最低高度，且中心差分速度小于阈值，随后填补
短 swing gap 并删除短 stance run。它给出 reference nominal schedule，不等同于力传感器真值。

## 可视化命令

```bash
PYTHONPATH=src /home/user/anaconda3/envs/wmd_ampmjlab/bin/python \
  -m themis_training.plot_reference_centroidal \
  assets/ref_motion/processed/smpl_jc01_taichi1_centroidal.npz \
  assets/ref_motion/processed/smpl_jc01_taichi1_figures
```

绘图输出 PNG 和 PDF：

- `centroidal_time_series`：CoM、CoM velocity、线动量、角动量的三轴时间序列；
- `contact_kinematics`：候选接触点高度、速度和推断 contact schedule；
- `centroidal_spatial_trajectory`：世界系 CoM 与左右候选接触点的 3D 轨迹；散点表示推断为 stance
  的样本。

## 本次已生成结果

上述命令已在该仓库中实际运行，输出路径为：

- `assets/ref_motion/processed/smpl_jc01_taichi1_centroidal.npz`；
- `assets/ref_motion/processed/smpl_jc01_taichi1_centroidal.json`；
- `assets/ref_motion/processed/smpl_jc01_taichi1_figures/`。

reference 共 3999 帧、50 Hz、79.96 s；XML 选中全部 reference body 后总质量为 57.00294 kg。
计算得到的 CoM 范围为

\[
c_x\in[-1.03171,-0.65644],\quad
c_y\in[-0.47124,-0.03710],\quad
c_z\in[0.72482,0.84653]\ \mathrm{m}.
\]

使用中心脚板点、(h_{thr}=0.03\) m、(v_{thr}=0.35\) m/s 的运动学 contact rule 后，左/右脚
推断 stance 占比分别为 87.12% / 72.47%，contact-mode 切换数分别为 8 / 14。该数字仅用于检查
reference schedule；不能将其解释为真实接触力测量结果。

## 物理量定义

对刚体 (j)，body-frame 到 CoM 的惯性偏移为 (d_j^B)，则

\[
p_j^{com}=p_j+R_jd_j^B,\qquad
v_j^{com}=v_j+\omega_j\times(R_jd_j^B).
\]

全身量为

\[
c=\frac{1}{m}\sum_jm_jp_j^{com},\qquad
l=\sum_jm_jv_j^{com},
\]

\[
k=\sum_j\left(I_j^W\omega_j+(p_j^{com}-c)\times m_jv_j^{com}\right).
\]

候选接触点是

\[
r_i=p_{body(i)}+R_{body(i)}d_i^B,\qquad r_i^{rel}=r_i-c.
\]

这些量可直接作为后续 CD-MPC 的 (x^{ref}=[c,l,k])、固定接触位置 (r_i) 和接触计划的离线
输入。
