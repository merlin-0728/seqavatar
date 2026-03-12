GPU_id=3
SEQUENCES=(0007_04 0019_10 0044_11 0051_09 0206_04 0813_05)

# --- WANDB 新增：定义项目名称 ---
export WANDB_PROJECT="SeqAvatar_DNA_Rendering"

# <your DNA-Rendering path>
data_path=/media/image/mxz/human/SeqAvatar/DNA-Rendering
iter=25000
densify_until_iter=1500

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01

for SEQUENCE in ${SEQUENCES[@]}; do
    exp_name=DNA-Rendering/${SEQUENCE}/
    dataset_path=$data_path/$SEQUENCE/
    model_path=output/$exp_name/
    mkdir -p "$model_path/logs"

    # --- WANDB 新增：为当前的 Sequence 运行动态命名 ---
    export WANDB_NAME="train_${SEQUENCE}"

    # Train
    echo "Training on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s $dataset_path --eval --exp_name $exp_name \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral \
        --iterations $iter --densify_until_iter $densify_until_iter \
        --seq_len $seq_len --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num  --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        --l1_loss_w $l1_loss_w --ssim_loss_w $ssim_loss_w --lpips_loss_w $lpips_loss_w \
        2>&1 | tee "$model_path/logs/train_${SEQUENCE}.log"

    # Evaluation
    echo "Evaluating on GPU $GPU_id for sequence $SEQUENCE"
    CUDA_VISIBLE_DEVICES=$GPU_id python render.py -s $dataset_path -m $model_path \
        --motion_offset_flag --smpl_type smplx --actor_gender neutral --iteration $iter --skip_train \
        --seq_len $seq_len   --seq_xyz_knn $seq_xyz_knn \
        --time_step_num $time_step_num  --max_time_step $max_time_step --minimal_time_step $minimal_time_step \
        2>&1 | tee "$model_path/logs/render_${SEQUENCE}.log"
done