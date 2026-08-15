# BeyondMimic 重定向动作到 MJLab reference NPZ

`src/themis_training/csv_to_npz_mjlab.py` 是 BeyondMimic `csv_to_npz.py` 在本项目中的 **MJLab**
版本。它保留输入格式、四元数、插值和有限差分语义；随后像 BeyondMimic 一样，将每帧状态写入运行时的
articulation、执行无积分的 forward、从 articulation data 读取状态，再保存 NPZ。因此生成的 NPZ 可直接作为
`MotionReferenceCommandCfg.motion_file` 使用。

## 输入契约

- **CSV**：每帧为 `[root_pos_x, root_pos_y, root_pos_z, root_quat_x, root_quat_y,
  root_quat_z, root_quat_w, q_1, ..., q_J]`。四元数从 CSV 的 `xyzw` 转成保存格式 `wxyz`。
- **PKL**：字典至少包含 `root_pos[T,3]`、`root_rot[T,4]`、`dof_pos[T,J]`；可选 `fps` 覆盖
  `--input-fps`。PKL 的四元数可由 `--pkl-root-rot-order` 指定；`auto` 与 BeyondMimic 一样，以
  upright tilt 较小者选择 `wxyz` 或 `xyzw`。
- `--frame-range START END` 是 1-indexed 且两端包含，语义与 BeyondMimic 相同。

转采样以 root 平移和关节角的线性插值、root 姿态的 shortest-path SLERP 完成。采样时间是
`arange(0, duration, 1/output_fps)`，故与原脚本一样不额外附加重复的终止帧。速度为有限差分：

\[
v_0(t)\simeq \frac{d p_0}{dt},\qquad \dot q(t)\simeq\frac{dq}{dt},\qquad
\omega_0(t)\simeq\frac{\log(q(t+\Delta t)q(t-\Delta t)^{-1})}{2\Delta t}.
\]

其中 \(\omega_0\) 在 world frame 表达，符合 MuJoCo free-joint 的 `qvel` 约定。

## 生成的 body 运动学与坐标系

转换器为 `themis`、`g1` 或 `jingchu01` 构建对应的 effort `Entity` 与 `Simulation`。每帧先写入
root state、joint state，再调用 `Simulation.forward()`，**不调用 `step()`**，所以不会把一个运动学参考误变成
受接触/重力影响的物理 rollout。随后直接读取 `EntityData`；这与 BeyondMimic 写入 Articulation 后经
render/update 读取状态的流程一一对应。

输出 `body_pos_w` / `body_quat_w` 是 link/body-frame origin 的世界位姿；而 `body_lin_vel_w` 是每个刚体
惯性 CoM 的世界线速度：

\[
p_{C,j}^W=p_{j,\mathrm{link}}^W+R_{WB,j}^W d_j^B,qquad
v_{C,j}^W=\texttt{body\_lin\_vel\_w}_j.
\]

这与 `reference_centroidal.compute_reference_centroidal` 的默认输入契约完全匹配；不要再对该速度加
\(\omega\times Rd\)。后者据此计算

参考 \(c,\dot c,l,k_G\)。因此请在生成 NPZ 后按需运行现有的
`python -m themis_training.process_reference_centroidal`，或让 `MotionReferenceCommand` 在训练初始化时
根据相同模型参数自动计算。接触计划可以由该 processor 输出 `contact_state`，也可由
`MotionReferenceCommand` 的运动学高度/速度规则在线加载为固定 reference schedule。

为了保证名称和数组布局严格一致，保存的 `joint_names` 和 `body_names` 也直接来自 Entity；
`source_joint_names` 仅保留原始重定向输入的关节次序。旧的 `reconstruct_body_kinematics()` 仍保留给
CPU 诊断/裁剪，它用 Jacobian 在 link origin 计算线速度，不能与本转换器生成的 CoM 速度混用。

## 使用方法

在仓库根目录、并使用安装了 `mujoco`、`mjlab`、`torch` 且可用 CUDA/MJWarp 的训练环境执行：

```bash
python -m themis_training.csv_to_npz_mjlab \
  /path/to/retargeted_motion.pkl \
  assets/ref_motion/processed/jingchu01_motion.npz \
  --robot jingchu01 --output-fps 50 --device cuda:0 --sim-dt 0.02
```

CSV 示例：

```bash
python -m themis_training.csv_to_npz_mjlab \
  /path/to/retargeted.csv assets/ref_motion/processed/g1_motion.npz \
  --robot g1 --input-format csv --input-fps 30 --output-fps 50 \
  --frame-range 1 900
```

默认 joint order 是对应 robot profile 的训练 joint order。仅当输入确实采用另一种已知顺序时，才通过
`--joint-names name1,name2,...` 覆盖；DOF 数和每个 joint 名会在 MuJoCo 模型中验证。

## MJLab/MuJoCo 回放与裁剪

```bash
python -m themis_training.replay_npz_mjlab \
  assets/ref_motion/processed/jingchu01_motion.npz --robot jingchu01 --loop
```

这会打开 MuJoCo viewer，以相同 MJLab profile 的前向运动学回放动作；它不启动 IsaacLab。服务器无图形
display 时请只使用转换/裁剪命令。裁剪会重新计算 root/joint/body 速度和 body 运动学，而不是简单截取
端点速度：

```bash
python -m themis_training.replay_npz_mjlab \
  assets/ref_motion/processed/jingchu01_motion.npz --robot jingchu01 \
  --trim-frame-range 120 520 \
  --trim-output assets/ref_motion/processed/jingchu01_motion_segment.npz
```

回放/裁剪是轻量级的 CPU MuJoCo 工具；裁剪重建的 body 线速度位于 **link origin**，因此输出会明确标记
`body_linear_velocity_point=link_origin`。若要得到与 BeyondMimic 完全相同的 EntityData/CoM-velocity
契约，请对原 CSV/PKL 重新运行转换器，而不是把裁剪结果当作同一类原始捕获数据。

## 与 centroidal-MPC mimic 配置的关系

转换器刻意不做坐标对齐，而是保存可回放、可审计的原始重定向世界系；其 NPZ metadata 会标注
`reference_frame_alignment=raw_source`。训练加载端的 `MJLabMotionLoader` 只读取、验证名称/形状，不会再次
重采样。随后 `MotionReferenceCommand` 默认以 `reference_frame_alignment="initial_anchor"` 对**全部**
body pose/velocity 做一次固定 canonicalization：移除初始 anchor 的 `xy+yaw`，保留高度、roll/pitch、动作内
后续平移和关节运动。该同一个变换同时服务 body imitation、参考 centroidal 计算和 MPC 接触位置；最后才加入
每个 parallel environment 的 `env_origin`。

这种分层优于在 `csv_to_npz` 中默认对齐：同一个 NPZ 可以用于原始轨迹可视化、不同训练实验的坐标约定或
不同 spawn origin，而不会不可逆地混入某次训练的环境语义。若需要一个永久 canonical 数据集，可在转换后
另存显式版本，但必须将 metadata 改为已对齐，并把训练 cfg 设为
`reference_frame_alignment="none"`，以避免二次旋转。BeyondMimic 风格的**运行时** robot-error yaw/xy 对齐
仍仅在 `aligned_body_pos_w()` 中生成 tracker body target；绝不能用它重锚定 CD-MPC reference，否则规划目标
会跟随机器人误差漂移。
