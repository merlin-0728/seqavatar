## 2026-07-02 part_moe_leg 代码路径

- 每次讨论 Part-MoE 前先读本文件，避免重复查同一问题。
- DNA 启动脚本：`scripts/exps_dnarendering.sh part_moe_leg` 会设置 `experiment_name=part_moe_leg`、`--use_part_moe`、`--part_label_schema part_moe_leg`、`--num_parts 7`，并且 `final_eval_only=1`，只在 25000 步评估/保存，避免 DNA 中间全量 eval 显存紧张。
- DNA 参数：`part_moe_start_iter=10000`、`part_moe_warmup=1000`、`part_moe_global_keep=0.1`、`densify_until_iter=1500`。训练代码要求 Part-MoE 下 `densify_until_iter <= part_moe_start_iter`，否则 label 与高斯数量/顺序不稳定。
- `part_moe_leg` 标签定义在 `part_label/common.py`：0 unknown，1 body，2 left_hand，3 right_hand，4 face，5 left_leg_foot，6 right_leg_foot。相比默认 `anatomy5`，新增左右腿脚两个 expert，body 不再包含腿脚。
- 标签生成在 `ablations/part_moe_controller.py`：第 10000 步 optimizer step 后保存当前高斯，用 canonical Gaussian 到 SMPL-X canonical 顶点的最近邻，把 SMPL-X LBS dominant joint 标签投到 Gaussian，距离超过 `part_max_smpl_dist` 或未知的点标 0。
- MoE 实现在 `nets/mlp_delta_non_rigid.py`：激活时把原 shared non-rigid MLP 深拷贝成 `num_parts` 个 expert。`expert_0` 处理 unknown/global，`expert_1..6` 按 part label 路由；输出按 `global_weight * expert_0 + part_weight * part_expert` 混合。
- 第 10000 步初始化 experts 并冻结原 shared 分支；第 10001 步开始 `part_moe_alpha` 从 0 warmup，最大 part weight 为 `1 - global_keep = 0.9`，global 分支至少保留 0.1。
- `orginal/original` 基线只是不传 `--use_part_moe`，因此不会生成/加载 part label，不会复制 experts，不会冻结 shared non-rigid 分支，渲染时也不加载 `part_labels/iteration_10000/gaussian_part_label.npy`。

## 2026-07-02 part_moe_pair 新增消融

### 目的

- `part_moe_leg` 是 7 个 expert：unknown、body、left_hand、right_hand、face、left_leg_foot、right_leg_foot。
- 新增 `part_moe_pair` 用来验证“左右对称部位共享 expert”是否更有效：左右手合并，左右腿脚合并。
- `part_moe_pair` 是 5 个 expert：0 unknown，1 body，2 hands，3 leg_foot，4 face。

### 代码路径

- 标签定义：`part_label/common.py` 的 `PART_LABEL_SCHEMAS["part_moe_pair"]`。
- DNA/SMPL-X：`smplx_lbs_vertex_labels(..., schema="part_moe_pair")` 把左右手都映射到 2，把左右腿脚都映射到 3，face 仍映射到 4。
- I3D/ZJU/SMPL：`map_smpl_seg_name_to_part_id(..., schema="part_moe_pair")` 把 `leftHand/rightHand` 映射到 2，`leftUpLeg/leftLeg/leftFoot/leftToeBase/rightUpLeg/rightLeg/rightFoot/rightToeBase` 映射到 3，`head` 映射到 4。
- 启动入口已加到 `scripts/exps_dnarendering.sh`、`scripts/exps_i3dhuman.sh`、`scripts/exps_zjumocap.sh`。DNA 下 `part_moe_pair` 跟其他 `part_moe_*` 消融一样只做最终 eval/save，减少中间全量评估显存压力。

### 对基线和其他消融的影响

- 没有改 `orginal/original`、`use_part_moe`、`part_moe_leg`、`part_moe_foot`、`part_moe_arm` 的参数和标签含义。
- 现有 Part-MoE 初始化、warmup、optimizer 和渲染加载逻辑复用原路径；新增内容只是一个 schema 和脚本 mode。

## 2026-07-02 评价指标速记

### 最终对比看什么

- 主指标是 `PSNR / SSIM / LPIPS`。
- 方向：`PSNR` 越高越好，`SSIM` 越高越好，`LPIPS` 越低越好。
- `L1` 主要用于训练中打印和 TensorBoard，不写入最终 `results_*.json`，不作为主要表格指标。

### 输出文件

- 训练过程评估会写 `metrics/results_<split>_<iter>.json`，里面是该 split 的平均 `SSIM / PSNR / LPIPS`。
- 同时写 `metrics/per_view<split>_<iter>.json`，里面是每张图的 `SSIM / PSNR / LPIPS`。
- `render.py` 会保存渲染图和 GT，并打印 `PSNR / SSIM / LPIPS` 以及 `Elapsed time / FPS`，但当前不写 `results_*.json`。

### 当前实验重点

- DNA 脚本对 `part_moe_pair` 只在最终 `25000` 步评估，重点看 `results_novelview_25000.json`。
- I3D 脚本会在 `3000 / 4000 / 15000` 步评估，最终表格优先看 `results_novelview_15000.json` 和 `results_novelpose_15000.json`。
- 做消融表时，同一数据集、同一 sequence、同一 split、同一 iteration 对比 `orginal`、`part_moe_leg`、`part_moe_pair` 等实验。

## 2026-07-02 part_moe_pair 最终指标

- 来源：按全局日志里的最终 `render.py` 评估输出统计，不是训练阶段写入的 `results_*.json`；两者会有极小数值差异。
- DNA run：`logs/part/20260702_004310_DNA-Rendering_part_moe_pair.log`，6 个 sequence 全部完成。novelview 均值：PSNR `32.412569573190475666666666666666666666666666666666666666666666666666666666666667`，SSIM `0.97470079859097795`，LPIPS*1000 `27.845305307871764833333333333333333333333333333333333333333333333333333333333333`。
- I3D run：`logs/part/20260702_004310_I3D-Human_part_moe_pair.log`，4 个 sequence 全部完成。novelview 均值：PSNR `32.391090261741450`，SSIM `0.9674140432022198`，LPIPS*1000 `28.457868593944062000`；novelpose 均值：PSNR `30.3746515964759325`，SSIM `0.9587860212911343`，LPIPS*1000 `34.8481640543423945000`。

## 2026-07-03 I3D Part-MoE iteration 口径确认

- 当前 `scripts/exps_i3dhuman.sh` 的非 TDP 路径默认仍是 `iter=15000`，因此 `part_moe_leg / part_moe_foot / part_moe_arm / part_moe_pair` 等 I3D 消融默认最终看 `results_novelview_15000.json` 和 `results_novelpose_15000.json`。
- 最近用于表格记录的 `part_moe_pair` I3D run：`logs/part/20260702_004310_I3D-Human_part_moe_pair.log`，最终是 `15000`。
- 历史上存在一版单独的普通 `part_moe` 25k 重跑：`logs/part/20260621_223442_I3D-Human_part_moe_25000.log`，输出目录中有 `results_novelview_25000.json` / `results_novelpose_25000.json`。
- 因此说“I3D Part-MoE 是 15000 还是 25000”要区分实验名：
  - `part_moe_pair / leg / foot / arm` 当前默认口径：`15000`。
  - `part_moe_25000` 那次历史重跑：`25000`。
  - 若要和 `logs/20260621_183404_I3D-Human_orginal_25000.log` 做严格对比，只能使用同为 `25000` 的 Part-MoE run，或重新跑对应 Part-MoE 到 `25000`。

## 2026-07-03 Part-MoE 是否影响 STMS

- 普通 `part_moe / part_moe_leg / part_moe_foot / part_moe_arm / part_moe_pair` 不改变 STMS 的 motion condition 构造。
- `scene/dataset_readers.py:get_seq_pose_xyz_cond()` 只根据 `use_msti / use_amc_pair / use_amc_causal / use_tdp` 切换 motion condition 分支；`use_part_moe` 不参与 `seq_pose_conds / seq_xyz_conds` 生成。
- `scene/__init__.py` 传给 dataset reader 的 `motion_cond_options` 也不包含 `use_part_moe`。
- 脚本中普通 Part-MoE 模式没有打开 MSTI / AMC / TDP，因此 `motion_cond_time_step_num = time_step_num`，STMS 输入 channel 与 baseline 相同。
- Part-MoE 在 `NonrigidDeformer.forward()` 中发生在 `SeqPoseEncoder / SeqXYZEncoder` 输出并 concat 成 `features` 之后；它替换的是后续 non-rigid deformation MLP/head 为 part-specific experts。
- 例外是组合实验 `part_moe_leg_msti`：它会打开 `use_msti=1`，所以 motion condition 会变；但这个变化来自 MSTI，不是普通 Part-MoE 本身。
