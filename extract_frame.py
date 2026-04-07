import os
import shutil

def extract_specific_frame(base_path, frame_index, output_dir):
    """
    base_path: 包含 00, 02, 04... 等摄像头文件夹的根目录
    frame_index: 想要提取的图片序号 (例如 "000000")
    output_dir: 结果保存的文件夹
    """
    
    # 确保序号格式正确 (补齐6位数字)
    frame_name = f"{int(frame_index):06d}.png"
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取 images 目录下所有的摄像头文件夹
    cam_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    cam_folders.sort() # 排序，保证处理顺序

    count = 0
    for cam_id in cam_folders:
        # 构建原文件路径
        src_file = os.path.join(base_path, cam_id, frame_name)
        
        if os.path.exists(src_file):
            # 构建新文件名，例如 "cam_00.png"
            dest_file = os.path.join(output_dir, f"cam_{cam_id}.png")
            
            # 复制文件
            shutil.copy2(src_file, dest_file)
            print(f"已提取: 摄像头 {cam_id} -> {dest_file}")
            count += 1
        else:
            print(f"警告: 摄像头 {cam_id} 文件夹下未找到图片 {frame_name}")

    print(f"\n任务完成！共提取了 {count} 张图片到 {output_dir}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 你的图像根目录
    SOURCE_IMAGES_DIR = "/media/image/mxz/human/SeqAvatar/DNA-Rendering/0007_04/bkgd_masks"
    
    # 你想提取的帧序号（比如想提取 000013.png，就写 13）
    TARGET_FRAME = 25 
    
    # 保存到的目标位置
    SAVE_DIR = f"./extracted_frame_DNA_mask_{TARGET_FRAME:06d}"
    # ----------------

    extract_specific_frame(SOURCE_IMAGES_DIR, TARGET_FRAME, SAVE_DIR)