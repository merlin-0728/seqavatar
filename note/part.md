# SeqAvatar Part-MoE 消融流程笔记

更新时间：2026-07-12

## 本轮重点

- 2026-07-12 02:24 CST 解释 I3D/ZJU `part0` 不如 `part_moe_leg`：同序列对齐后，I3D 4 序列 `part0` novelview 平均 PSNR `32.362646131807856`、SSIM `0.96719044205402175`、LPIPS*1000 `29.2224481107328725000`，低于 `part_moe_leg/20260622_145118` 同 4 序列 novelview 平均 PSNR `32.39407307725735675`、SSIM `0.96744398220732645`、LPIPS*1000 `28.4920528156118955000`；novelpose 也低于 `part_moe_leg`，PSNR 少 `0.16812400572034275`、SSIM 少 `0.00062418078442`、LPIPS*1000 高 `1.12102899108268825`。ZJU `part0` test 平均 PSNR `31.0948642357904671666666666666666666666666666666666666666667`、SSIM `0.9620049190700678`、LPIPS*1000 `28.7554153190084646666666666666666666666666666666666666666667`，低于 `part_moe_leg/20260622_185140` 的 PSNR `31.173660165097125333333333333333333333333333333333`、SSIM `0.96237724571242931666666666666666666666666666666667`、LPIPS*1000 `28.0125608609677885000`。主要原因：I3D/ZJU 使用 SMPL vertex segmentation 作为粗标签，`part_moe_leg` 先用 shared non-rigid MLP 训练到 I3D `4000`/ZJU `1000`，等几何和非刚性场稳定后再复制到 7 个 expert；`part0` 从第 0 步把随机/未收敛 shared MLP 直接拆成 7 个 expert，并在早期增密阶段频繁动态重算标签，导致每个 expert 数据更少、路由边界更早固定，短训练预算尤其吃亏。DNA 上 `part0` 更好不等价于 I3D/ZJU 也更好，因为 DNA 是 SMPL-X/prior_only、25000 步且已有结果显示 early split 能受益；I3D/ZJU 的 SMPL 粗分割和短训练更依赖 warmup。
- 2026-07-12 02:16 CST I3D-Human 与 ZJU-MoCap `part0` 消融已全部跑完，GPU 0/1/3 空闲。I3D 默认四序列 `ID1_1/ID1_2/ID2_1/ID3_1` 均完成 `iteration_15000` 训练和 `render_part0.py`，novelview 平均 PSNR `32.362646131807856`、SSIM `0.96719044205402175`、LPIPS*1000 `29.2224481107328725000`；novelpose 平均 PSNR `30.28636955741382125`、SSIM `0.9583724801355553`、LPIPS*1000 `35.75321242658101375000`。ZJU 默认六序列 `CoreView_377/386/387/392/393/394` 均完成 `iteration_3000` 训练和 `render_part0.py`，test 平均 PSNR `31.0948642357904671666666666666666666666666666666666666666667`、SSIM `0.9620049190700678`、LPIPS*1000 `28.7554153190084646666666666666666666666666666666666666666667`。
- 2026-07-12 00:43:54 CST 已用 GPU 0/1/3 启动 I3D-Human 与 ZJU-MoCap 的 `part0` 消融，`RUN_TIME=20260712_004354`。GPU0 顺序跑 `I3D: ID1_1 ID3_1` 后接 `ZJU: CoreView_377 CoreView_392`；GPU1 跑 `I3D: ID1_2` 后接 `ZJU: CoreView_386 CoreView_393`；GPU3 跑 `I3D: ID2_1` 后接 `ZJU: CoreView_387 CoreView_394`。driver 日志分别为 `logs/part/20260712_004354_part0_i3d_zju_gpu0_driver.log`、`gpu1_driver.log`、`gpu3_driver.log`。启动后复查三路日志，均进入 `I3D-Human part0`，`TRAIN_ENTRYPOINT=train_part0.py`、`RENDER_ENTRYPOINT=render_part0.py`、`PART_MOE_START_ITER=0`、`PART_MOE_WARMUP=0`、`PART_LABEL_SCHEMA=part_moe_leg`、`NUM_PARTS=7`。
- 2026-07-12 00:44 CST 按 DNA `part0` 同思路给另外两个数据集脚本新增 `part0` mode：`scripts/exps_i3dhuman.sh part0`、`scripts/exps_zjumocap.sh part0`。新增逻辑只在 `part0_enabled=1` 时生效：`experiment_name=part0`、`part_label_schema=part_moe_leg`、`num_parts=7`、`part_moe_start_iter=0`、`part_moe_warmup=0`、入口切到 `train_part0.py/render_part0.py`；常规 `orginal/part_moe/part_moe_leg/part_moe_foot/part_moe_arm` 仍走原 `train.py/render.py`。I3D/ZJU 继续使用 `smpl_type=smpl`、`part_grouping_mode=smpl_vertex_seg` 和 `SMPL_VERTEX_SEG_PATH=/media/image/mxz/human/SeqAvatar/smpl_model/smpl_vert_segmentation.json`。ZJU 的静态 Part-MoE 仍保留 `densify_until_iter <= part_moe_start_iter` 限制；`part0` 因训练中动态刷新 label，使用基线 `DENSIFY_UNTIL_ITER=1200` 并绕过该静态 label 限制。验证：`bash -n scripts/exps_i3dhuman.sh scripts/exps_zjumocap.sh scripts/exps_dnarendering.sh` 通过。
- 2026-07-12 00:47 CST 复核 `part_moe_pair`：这是历史 DNA-Rendering 消融，完整六序列输出在 `output/DNA-Rendering/<seq>/part_moe_pair/20260702_004310/`，总日志是 `logs/part/20260702_004310_DNA-Rendering_part_moe_pair.log`，启动时间 `Thu Jul 2 00:43:21 CST 2026`。实际 `cfg_args` 六序列一致：`use_part_moe=True`、`part_moe_start_iter=10000`、`part_moe_warmup=1000`、`part_moe_global_keep=0.1`、`num_parts=5`、`part_label_schema='part_moe_pair'`、`part_grouping_mode='prior_only'`、`part_max_smpl_dist=0.08`、`smpl_type='smplx'`、`non_rigid_mlp_depth=3`、`non_rigid_mlp_width=512`。它的标签是 bilateral-pair 五类：`0 unknown`、`1 body`、`2 hands`、`3 leg_foot`、`4 face`，即左右手合成一个 expert、左右腿脚合成一个 expert；与 `part_moe_leg` 的 7 类左右拆分不同。当前代码 `part_label/common.py` 和 `scripts/exps_dnarendering.sh` 已无 `part_moe_pair` schema/mode，因此它是历史结果，不是现脚本可直接重启的分支。
- 2026-07-12 00:36 CST 复核 `part0` 两个 RUN_TIME：用户判断是对的，`RUN_TIME` 本身只是输出目录/启动批次，不是实验变量；只要代码入口、参数、数据划分和评价口径一致，两个启动批次混合成六序列平均没有问题。本轮检查六个 `cfg_args`，关键设置一致：`use_part_moe=True`、`part_moe_start_iter=0`、`part_moe_warmup=0`、`part_moe_global_keep=0.1`、`num_parts=7`、`part_label_schema='part_moe_leg'`、`part_grouping_mode='prior_only'`、`part_max_smpl_dist=0.08`、`smpl_type='smplx'`，以及 `non_rigid_mlp_depth=3`、`non_rigid_mlp_width=512`、`seq_xyz_knn=8`、`time_step_num=3`、`seq_len=8`、`max_time_step=3`、`minimal_time_step=1`。因此当前可表述为：`part0` 是同设置下当前最好结果；混合 RUN_TIME 只需作为崩溃后补跑 provenance 记录，若论文排版/复现实验想更干净，可以选择整批重跑，但不是接受当前比较的必要条件。
- 2026-07-12 00:25 CST 当前结论：按已有完整六序列平均指标看，`part0` 是当前最好的一组结果，PSNR `32.4067`、SSIM `0.974630`、LPIPS `0.027753`，三项都优于当前记录中的 `part_moe_pair/part_moe_arm/part_moe_leg`。优势很小，正式表格中应保留完整单序列指标和补跑 provenance，避免只看均值过度解读。
- 2026-07-12 00:10 CST 梳理 `part0` 代码流程：`scripts/exps_dnarendering.sh part0` 只负责统一参数和入口选择；训练走 `train_part0.py`，在 optimizer 创建前生成 `iteration_0` label 并初始化 7 个 expert；label 由 `ablations/part0_moe_controller.py` 根据当前 canonical Gaussian 到 SMPL-X canonical 顶点最近邻计算；训练中每次增密/剪枝后刷新内存 label，最终保存 `iteration_25000` label；渲染走 `render_part0.py`，优先加载最终迭代 label；真正的 expert 路由在 `nets/mlp_delta_non_rigid.py`，按 part label 调不同 expert，并用 `part_moe_alpha=0.9` 与 global expert 混合。
- 2026-07-12 00:04 CST 已把 `part0` 启动入口合并进 `scripts/exps_dnarendering.sh`，现在可用 `bash scripts/exps_dnarendering.sh part0` 启动；该 mode 内部仍调用 `train_part0.py/render_part0.py` 和 `part0_moe_controller.py` 保持动态 label 刷新逻辑。已删除冗余启动脚本 `scripts/exps_dnarendering_part0.sh`、`scripts/launch_dnarendering_part0_4gpu.sh`。验证：`bash -n scripts/exps_dnarendering.sh` 通过；`RUN_TIME=codex_part0_skipcheck SKIP_COMPLETED=1 SEQUENCES_OVERRIDE="0044_11" GPU_id=0 bash scripts/exps_dnarendering.sh part0` 正确识别并跳过已有完整结果。
- 2026-07-11 23:44 CST 按原始精度汇总 `part0` 六序列评价指标，表格口径改为 `PSNR`、`SSIM`、`LPIPS*1000`；数值直接来自 JSON 并用 Decimal 做乘 1000，不四舍五入。平均值按不四舍五入口径截断到小数点后 30 位：PSNR `32.406713173124525666666666666666`、SSIM `0.974629526088635116666666666666`、LPIPS*1000 `27.753426172097938833333333333333`。
- 2026-07-11 23:37 CST 复查 `part0`：首轮 `20260711_164830` 未跑全，但补跑 `20260711_212950` 的四个缺失序列已全部完成训练和 `render_part0.py` novelview 渲染；当前没有 `train_part0.py/render_part0.py` 进程，`nvidia-smi` 显示四张卡空闲。六序列可用结果由 `0051_09/0813_05@20260711_164830` 加 `0044_11/0206_04/0007_04/0019_10@20260711_212950` 组成，平均 PSNR `32.4067`、SSIM `0.974630`、LPIPS `0.027753`。
- 2026-07-11 16:48:30 CST 已用 4 张卡启动 DNA 六序列 `part0` 消融，`RUN_TIME=20260711_164830`。当前第一批四个序列训练中，未发现 traceback/OOM/error；主线文件 `train.py/render.py/scene/gaussian_model.py/ablations/part_moe_controller.py/scripts/exps_dnarendering.sh` 保持无 diff。
- 2026-07-11 追加 `part0` 消融：在训练开始前的 canonical 初始高斯上生成 `part_moe_leg` 七类 label，并从第 1 个训练 iteration 开始直接使用 7 个 Part-MoE expert。
- `part0` 现在保留原 DNA 增密：`densify_until_iter=1500`。每次 `densify_and_prune()` 改变高斯拓扑后，会按当前 canonical Gaussian 到 SMPL-X canonical 顶点的最近邻规则刷新内存 part label。
- `part0` 仍保留独立训练/渲染实现路径：`train_part0.py`、`render_part0.py`、`ablations/part0_moe_controller.py`；启动入口已并入 `scripts/exps_dnarendering.sh part0`。常规 `train.py/render.py/scene/gaussian_model.py/ablations/part_moe_controller.py` 不承载 part0 动态刷新改动，避免影响已有 `orginal/part_moe_leg/part_moe_arm` 等实验。
- 本轮已阅读 `/media/image/mxz/human/SeqAvatar` 的训练、渲染、Part-MoE 分层和脚本流程。
- 代码里的正确命令/schema 拼写是 `part_moe_leg`；用户口头写的 `paert_moe_leg` 应按 `part_moe_leg` 执行。
- `orginal` 是脚本和输出目录沿用的基线名称，脚本也兼容 `original` 输入，但输出仍写到 `orginal`。
- 当前主线建议按用户结论使用 `part_moe_leg`：在原 `orginal` 基线的共享 non-rigid MLP 上，于中途冻结共享分支并复制出按部位路由的专家。

## DNA-Rendering 一键使用

```bash
cd /media/image/mxz/human/SeqAvatar

# 原始基线
GPU_id=3 bash scripts/exps_dnarendering.sh orginal

# 当前重点消融：Part-MoE leg
GPU_id=3 bash scripts/exps_dnarendering.sh part_moe_leg

# 新增消融：第 0 步前分组，从训练一开始使用 7 个 expert
GPU_id=3 bash scripts/exps_dnarendering.sh part0

# 只跑指定序列
SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh part_moe_leg

SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=3 bash scripts/exps_dnarendering.sh part0

# part0 四卡手动启动六个 DNA 序列，四个 shell 共用同一个 RUN_TIME
RUN_TIME=$(date +%Y%m%d_%H%M%S)
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0044_11 0007_04" GPU_id=0 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0051_09 0019_10" GPU_id=1 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0813_05" GPU_id=3 bash scripts/exps_dnarendering.sh part0

# 跳过已有完整结果
SKIP_COMPLETED=1 GPU_id=3 bash scripts/exps_dnarendering.sh part_moe_leg
```

默认 DNA 序列：`0044_11 0051_09 0206_04 0813_05 0007_04 0019_10`。

## `part_moe_leg` 默认参数

- 训练总步数：`25000`
- 增密截止：`densify_until_iter=1500`
- Part-MoE 激活/分层步：`part_moe_start_iter=10000`
- warmup：`part_moe_warmup=1000`
- global/shared keep：`part_moe_global_keep=0.1`
- schema：`part_label_schema=part_moe_leg`
- expert 数量：`num_parts=7`
- DNA 使用 `smpl_type=smplx`、`part_grouping_mode=prior_only`
- `part_moe_leg/foot/arm` 在 DNA 脚本里启用 `final_eval_only=1`，只做最终 25000 步全量 eval/save，避免中间 eval 显存紧张。

`part_moe_leg` 标签定义：

- `0 unknown/global`
- `1 body`
- `2 left_hand`
- `3 right_hand`
- `4 face`
- `5 left_leg_foot`
- `6 right_leg_foot`

## `part0` 消融

目的：验证“不经过 10000 步 shared non-rigid warmup，直接从 canonical 初始空间分组并训练 7 个 MLP expert”是否更好。

默认 DNA 参数：

- mode/输出名：`part0`
- schema：`part_label_schema=part_moe_leg`
- expert 数量：`num_parts=7`
- 分层步：`part_moe_start_iter=0`
- warmup：`part_moe_warmup=0`
- global/shared keep：`part_moe_global_keep=0.1`，所以第 1 个训练 iteration 起 `part_moe_alpha=0.9`
- 增密：`densify_until_iter=1500`
- 初始 label 输出：`part_labels/iteration_0/gaussian_part_label.npy`
- 最终渲染 label 输出：`part_labels/iteration_25000/gaussian_part_label.npy`

实现路径：

- `scripts/exps_dnarendering.sh part0` 统一启动；`part0` mode 内部调用 `train_part0.py` 和 `render_part0.py`，避免改动常规 `train.py/render.py` 的静态 label 路径。
- `train_part0.py` 在 `Scene` 初始化完、`training_setup()` 建 optimizer 之前调用 `PartMoeController.before_training()`。
- `before_training()` 保存 `point_cloud/iteration_0/point_cloud.ply` 和 `mlp_ckpt/iteration_0/ckpt.pth`，再生成并加载 `part_labels/iteration_0/*`。
- `GaussianModel.init_part_moe_from_shared()` 在 optimizer 尚未创建时也会冻结原 shared non-rigid 分支；随后训练时 non-rigid 的有效梯度主要走 Part-MoE expert。
- 每次增密/剪枝后，`PartMoeController.refresh_labels_after_topology_change()` 会对当前所有高斯重算 label，不是简单复制父点 label。
- 保存迭代时，`PartMoeController.save_current_labels()` 会写出当前迭代的 label；`render_part0.py` 优先加载 `part_labels/iteration_<loaded_iter>/gaussian_part_label.npy`，找不到才回退到 `part_moe_start_iter`。

运行：

```bash
cd /media/image/mxz/human/SeqAvatar
GPU_id=3 bash scripts/exps_dnarendering.sh part0
```

四卡六序列：

```bash
cd /media/image/mxz/human/SeqAvatar
RUN_TIME=$(date +%Y%m%d_%H%M%S)
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0044_11 0007_04" GPU_id=0 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0051_09 0019_10" GPU_id=1 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh part0
RUN_TIME=$RUN_TIME SEQUENCES_OVERRIDE="0813_05" GPU_id=3 bash scripts/exps_dnarendering.sh part0
```

四卡分配：

- GPU 0：`0044_11` -> `0007_04`
- GPU 1：`0051_09` -> `0019_10`
- GPU 2：`0206_04`
- GPU 3：`0813_05`

本次运行状态：

- 启动时间：`2026-07-11 16:48:30 CST`
- `RUN_TIME`：`20260711_164830`
- launcher PID：`781869`
- launcher 日志：`logs/part/20260711_164830_DNA-Rendering_part0_4gpu_launcher.log`
- nohup 日志：`logs/part/20260711_164830_DNA-Rendering_part0_4gpu_nohup.log`
- worker 日志：`logs/part/20260711_164830_DNA-Rendering_part0_worker_gpu0.log` 到 `gpu3.log`
- 16:56:42 CST 快照：GPU0 `0044_11` 2230/25000，GPU1 `0051_09` 2430/25000，GPU2 `0206_04` 2430/25000，GPU3 `0813_05` 2470/25000。
- 已确认第一批四个序列生成 `part_labels/iteration_0/gaussian_part_label.npy` 和 `point_cloud/iteration_0/point_cloud.ply`；最终 `iteration_25000` 和 metrics 需等待训练、渲染完成后再检查。

重要注意：

- `part0` 和 `part_moe_leg` 的核心差异现在集中在是否经过 10000 步 shared warmup；增密节奏与 DNA 默认一致。
- 动态刷新是全量重算：每个当前高斯都按最近 SMPL-X 顶点重新分组。因此新分裂出来的高斯会自然落到当前近邻对应的 body/hand/face/leg part。

代码流程摘要：

1. `scripts/exps_dnarendering.sh part0` 设置 `experiment_name=part0`、`part_moe_start_iter=0`、`part_moe_warmup=0`、`part_label_schema=part_moe_leg`、`num_parts=7`，并把入口切到 `train_part0.py/render_part0.py`。
2. `train_part0.py` 创建 `GaussianModel` 和 `Scene` 后，如果 `part_moe_start_iter <= 0`，会在 `training_setup()` 之前调用 `PartMoeController.before_training()`，立刻写出 `point_cloud/iteration_0`、`mlp_ckpt/iteration_0` 和初始 `part_labels/iteration_0`。
3. `part0_moe_controller.py` 用当前 `gaussians.get_xyz` 到 `gaussians.canon_vertices` 的 `cKDTree` 最近邻，把 SMPL-X 顶点 part label 投到每个 Gaussian；超过 `part_max_smpl_dist` 的点保留为 `0 unknown/global`。
4. 初始 label 加载后，`GaussianModel.init_part_moe_from_shared()` 将 shared non-rigid MLP 复制成 7 个 expert，加入 `mlp_optimizer` 的 `part_expert_0..6` 参数组，并冻结原 shared 分支。
5. 每个 iteration 里，`compute_part_moe_alpha()` 因 `warmup=0` 从第 1 步起给 `alpha=1-global_keep=0.9`；renderer 把当前 `part_label`、`part_moe_alpha` 和 `global_keep` 传给 `NonrigidDeformer`。
6. `NonrigidDeformer.forward_part_moe()` 先用 expert 0 产生 global/unknown 输出，再对 label 1..6 的点分别调用对应 expert，并按 `global_weight=0.1`、`part_weight=0.9` 混合。
7. 训练保留 `densify_until_iter=1500`。每次 `densify_and_prune()` 改变点数后，controller 会按当前高斯拓扑全量重算内存 label，避免 label 数量和 Gaussian 数量不一致。
8. 保存迭代时写出当前 label；最终 `render_part0.py` 优先读取 `part_labels/iteration_25000/gaussian_part_label.npy`，找不到才回退到 `iteration_0`。

## 训练内部时序

1. `train.py` 先按原 SeqAvatar 流程训练共享 non-rigid deformer。
2. 常规 `part_moe_leg` 会检查 `densify_until_iter <= part_moe_start_iter`，因为固定 part label 依赖高斯点顺序和数量；`part0` 是例外，它会在每次拓扑变化后动态重算 label。
3. 到 `part_moe_start_iter` 后，`PartMoeController` 保存当前 `iteration_10000` 的点云和 MLP。
4. 控制器用 canonical Gaussian 到 canonical SMPL-X 顶点的最近邻生成 `gaussian_part_label.npy` 和 `gaussian_part_conf.npy`。
5. 标签立即加载进 `GaussianModel`，随后 `init_part_moe_from_shared()` 把 shared non-rigid MLP 复制成 7 个 expert。
6. shared 分支被冻结；`mlp_optimizer` 新增 `part_expert_0..6` 参数组继续训练。
7. `part_moe_alpha` 从 0 线性 warmup 到 `1 - global_keep = 0.9`；渲染时直接使用 0.9。

## 输出和检查点

单个 DNA 结果目录格式：

```text
output/DNA-Rendering/<SEQUENCE>/<MODE>/<RUN_TIME>/
```

重点文件：

- `cfg_args`：完整训练参数，可被 `render.py` 自动合并读取。
- `point_cloud/iteration_25000/point_cloud.ply`：最终高斯点。
- `mlp_ckpt/iteration_25000/ckpt.pth`：pose/LBS/non-rigid/Part-MoE expert 权重。
- `metrics/results_novelview_25000.json`：最终指标。
- `part_labels/iteration_10000/gaussian_part_label.npy`：常规 `part_moe_leg` 训练和渲染需要的 part label。
- `part_labels/iteration_0/gaussian_part_label.npy`：`part0` 训练开始前的初始 part label。
- `part_labels/iteration_25000/gaussian_part_label.npy`：`part0` 最终渲染优先加载的 part label。
- `part_labels/iteration_*/gaussian_part_meta.json`：标签统计和 schema 元信息。
- `logs/part/<RUN_TIME>_DNA-Rendering_part_moe_leg.log`：一键脚本总日志。

## 手动渲染

一键脚本训练完会自动调用 `render.py`。手动渲染时可直接用同一输出目录：

```bash
cd /media/image/mxz/human/SeqAvatar

CUDA_VISIBLE_DEVICES=3 /media/image/mxz/.conda/envs/seqavatar/bin/python render.py \
  -s /media/image/mxz/human/SeqAvatar/DNA-Rendering/0007_04/ \
  -m output/DNA-Rendering/0007_04/part_moe_leg/<RUN_TIME>/ \
  --iteration 25000 \
  --skip_train
```

如果 `cfg_args` 存在，`render.py` 会从模型目录读取 `--use_part_moe`、`part_label_schema` 等参数。默认优先加载最终迭代 label：

```text
part_labels/iteration_<loaded_iter>/gaussian_part_label.npy
```

如果最终迭代 label 不存在，再回退到 `part_moe_start_iter` 对应的 label；缺 label 时可手动加 `--part_label_path <path>`。

## I3D/ZJU 使用差异

- `scripts/exps_i3dhuman.sh`、`scripts/exps_zjumocap.sh` 支持同样模式：`orginal`、`part_moe`、`part_moe_leg`、`part_moe_foot`、`part_moe_arm`。
- I3D/ZJU 使用 `smpl_type=smpl`，Part-MoE 分层走 `part_grouping_mode=smpl_vertex_seg`，需要 `SMPL_VERTEX_SEG_PATH=/media/image/mxz/human/SeqAvatar/smpl_model/smpl_vert_segmentation.json`。
- I3D 默认 `part_moe_start_iter=4000`、总步数 `15000`。
- ZJU 默认 `part_moe_start_iter=1000`、总步数 `3000`，且 Part-MoE 时 `densify_until_iter` 默认等于 start iter。

## 指标备注

- 评价文件：`metrics/results_novelview_25000.json`，由训练阶段最终 `testing_iterations=25000` 的 novelview 评估写出；`render_part0.py` 会打印 novelview 指标但当前不写 JSON。
- 指标方向：PSNR 越高越好，SSIM 越高越好，LPIPS 越低越好。当前 novelview 默认是 `#120` 张测试视角。
- 本轮扫到的 DNA `part_moe_leg/20260623_180431` 六序列平均约为：PSNR `32.3739`、SSIM `0.974484`、LPIPS `0.028103`。
- 可用历史指标里 `part_moe_arm/latest` 六序列平均 PSNR 略高，但差距很小；若后续写论文/汇报，应重新按同一 run、同一序列集合、同一渲染结果确认最终“最好”的比较口径。
- 2026-07-11 19:00 CST 扫描 `part0/20260711_164830`：当前只有 `0051_09`、`0813_05` 产出最终 JSON，因此不能作为六序列平均和 `part_moe_leg` 直接比较。两序列平均：PSNR `32.4205`、SSIM `0.979086`、LPIPS `0.024708`。
- `part0/20260711_164830` 已出指标：
  - `0051_09`：PSNR `28.7180`、SSIM `0.971285`、LPIPS `0.030730`；训练完成并写出 JSON，但后续独立 render 在加载 SMPL 参数到 CUDA 时触发 `CUDA error: unknown error`。
  - `0813_05`：PSNR `36.1231`、SSIM `0.986886`、LPIPS `0.018686`；训练完成并写出 JSON，独立 render 也完成，render 日志打印 PSNR `36.1268`、SSIM `0.986899`、LPIPS `0.018672`。
- `part0/20260711_164830` 未完成：`0044_11` 在最终训练评估阶段触发 `CUDA error: unknown error`，`0206_04` 在最终训练评估阶段触发 `CUDA error: unspecified launch failure`；`0007_04`、`0019_10` 未启动到第二批，因为 GPU0/GPU1 worker 在前序任务异常后退出。
- `part0/20260711_164830` 缺失原因补充：日志里没有 `CUDA out of memory`、`OOM killed` 或 `cannot allocate`，因此不是典型 PyTorch OOM。`0044_11/0206_04` 都是在 25000 步最终 novelview 评估阶段崩溃；`0051_09` 训练已写 JSON，但后续 `render_part0.py` 加载 SMPL 参数到 CUDA 时崩溃。当前 `nvidia-smi` 还显示 `Unable to determine the device handle for GPU2 ... Unknown Error`，说明 GPU2/驱动状态异常，优先按 CUDA/驱动或 kernel launch failure 排查，而不是按普通显存不足处理。
- 旧独立脚本 `scripts/exps_dnarendering_part0.sh` 使用 `set -euo pipefail`，单个 worker 中前一个序列失败会直接退出该 worker，所以 GPU0 上的 `0007_04` 和 GPU1 上的 `0019_10` 没有继续启动；该脚本已在 2026-07-12 合并进 `scripts/exps_dnarendering.sh part0` 后删除。
- 2026-07-11 21:27 CST GPU/CUDA 恢复后，已用持久会话重跑 `part0` 缺失四序列，`RUN_TIME=20260711_212950`：GPU0 `0044_11`，GPU1 `0206_04`，GPU2 `0007_04`，GPU3 `0019_10`。会话 ID：`28461/24228/68296/88959`。
- 2026-07-11 23:37 CST 复查：`part0/20260711_212950` 四个补跑序列均完成训练和独立 render，日志均出现 `Training complete`、`Finished sequence`、`Worker finished`；当前无 `train_part0.py/render_part0.py` 进程，GPU 空闲。四个补跑序列的最终 `point_cloud/iteration_25000/point_cloud.ply`、`mlp_ckpt/iteration_25000/ckpt.pth`、`part_labels/iteration_25000/gaussian_part_label.npy`、`metrics/results_novelview_25000.json` 均存在。
- `part0` 当前六序列完整结果需要混合两个 RUN_TIME：`0051_09`、`0813_05` 来自 `20260711_164830`；`0044_11`、`0206_04`、`0007_04`、`0019_10` 来自 `20260711_212950`。单序列指标：`0044_11` PSNR `33.0414`/SSIM `0.978313`/LPIPS `0.020915`；`0051_09` `28.7180`/`0.971285`/`0.030730`；`0206_04` `31.5594`/`0.970893`/`0.032797`；`0813_05` `36.1231`/`0.986886`/`0.018686`；`0007_04` `29.5859`/`0.959016`/`0.042676`；`0019_10` `35.4125`/`0.981384`/`0.020718`。六序列平均：PSNR `32.4067`、SSIM `0.974630`、LPIPS `0.027753`。
- 2026-07-11 23:44 CST `part0` 六序列原始精度指标，LPIPS 统一展示为 `LPIPS*1000`，不四舍五入：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| `0044_11` | `20260711_212950` | `33.04136627515157` | `0.9783132697145144` | `20.915047475136818000` |
| `0051_09` | `20260711_164830` | `28.718026049931844` | `0.9712851017713546` | `30.729611512894432000` |
| `0206_04` | `20260711_212950` | `31.55936663945516` | `0.970892936984698` | `32.796685102706155000` |
| `0813_05` | `20260711_164830` | `36.12305695215861` | `0.9868864913781484` | `18.685678121012947000` |
| `0007_04` | `20260711_212950` | `29.585934511820476` | `0.9590157677729925` | `42.675718385726216000` |
| `0019_10` | `20260711_212950` | `35.412528610229494` | `0.9813835889101028` | `20.717816435111065000` |
| `Average` | mixed, truncated to 30 decimals | `32.406713173124525666666666666666` | `0.974629526088635116666666666666` | `27.753426172097938833333333333333` |

- 当前完整六序列可比结果：`part0` 混合 RUN_TIME 平均 PSNR `32.4067`、SSIM `0.974630`、LPIPS `0.027753`；`part_moe_pair/20260702_004310` 平均 PSNR `32.3956`、SSIM `0.974561`、LPIPS `0.027958`；`part_moe_arm/20260624_174257` 平均 PSNR `32.3939`、SSIM `0.974623`、LPIPS `0.028021`；`part_moe_leg/20260623_180431` 平均 PSNR `32.3739`、SSIM `0.974484`、LPIPS `0.028103`。所以按当前记录，`part0` 是当前最好结果；相比第二名 `part_moe_pair`，PSNR 约高 `0.0111`，SSIM 约高 `0.000069`，LPIPS 约低 `0.000205`。差距都很小，但混合 RUN_TIME 不影响同设置六序列平均；正式表格只需注明 `0051_09/0813_05` 来自 `20260711_164830`、其余四个序列来自 `20260711_212950`，若追求最干净 provenance 再整批重跑。
- `part_moe_pair/20260702_004310` 单序列指标：`0044_11` PSNR `32.99402745564779`/SSIM `0.9782118901610375`/LPIPS `0.021132196540323396`；`0051_09` `28.66549391746521`/`0.9712576518456141`/`0.03093820276359717`；`0206_04` `31.520674228668213`/`0.9704170053203901`/`0.03299284311942756`；`0813_05` `36.19472414652507`/`0.9873321910699209`/`0.017870137429175276`；`0007_04` `29.601748689015707`/`0.9587682584921519`/`0.044110596366226676`；`0019_10` `35.39713214238485`/`0.9813819125294685`/`0.020703238962839047`。Decimal 平均：PSNR `32.395633429951140`、SSIM `0.97456148490309716666666666666666666666666666666667`、LPIPS `0.027957869196931520833333333333333333333333333333333`。

## I3D/ZJU `part0` 结果

`RUN_TIME=20260712_004354`，指标来自各输出目录 `metrics/results_*.json`。表中 LPIPS 统一展示为 `LPIPS*1000`。

I3D-Human novelview，`iteration_15000`：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| `ID1_1` | `32.00631448030472` | `0.966927744448185` | `25.905285484623165000` |
| `ID1_2` | `32.016361581125565` | `0.966272283369495` | `27.459097389251955000` |
| `ID2_1` | `31.609073541103264` | `0.9695865660905838` | `29.3938123071805000` |
| `ID3_1` | `33.818834924697875` | `0.9659751743078232` | `34.13159726187587000` |
| `Average` | `32.362646131807856` | `0.96719044205402175` | `29.2224481107328725000` |

I3D-Human novelpose，`iteration_15000`：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| `ID1_1` | `30.106139866511025` | `0.9601512953639031` | `31.50703700569769000` |
| `ID1_2` | `30.32524145444234` | `0.9587890475988388` | `32.08056630877157000` |
| `ID2_1` | `28.04337288203992` | `0.9548867627194053` | `41.120846355497315000` |
| `ID3_1` | `32.670724026662` | `0.959662814860074` | `38.30440003635748000` |
| `Average` | `30.28636955741382125` | `0.9583724801355553` | `35.75321242658101375000` |

ZJU-MoCap test，`iteration_3000`：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| `CoreView_377` | `31.440909070831736` | `0.9733393915247118` | `17.984711734408684000` |
| `CoreView_386` | `33.60602649052938` | `0.9658285076871063` | `31.29883169318841000` |
| `CoreView_387` | `28.845758577789923` | `0.9559518425452589` | `31.855338328339235000` |
| `CoreView_392` | `32.22259645598927` | `0.9652463427285829` | `27.675325697082937000` |
| `CoreView_393` | `29.43149801128167` | `0.9547290207187006` | `33.66133888359836000` |
| `CoreView_394` | `31.022396808320824` | `0.9569344092160463` | `30.056945577433162000` |
| `Average` | `31.0948642357904671666666666666666666666666666666666666666667` | `0.9620049190700678` | `28.7554153190084646666666666666666666666666666666666666666667` |

## 后续维护约定

- 每次继续讨论本项目时，先读本文件，再把新的结论、命令、实验结果或注意事项追加/更新到这里。
