import os
import math
from PIL import Image
from tqdm import tqdm

def create_image_grid(input_folder, output_path, grid_rows=6, grid_cols=12):
    # 1. 获取所有图片文件
    if not os.path.exists(input_folder):
        print(f"Error: 文件夹不存在 - {input_folder}")
        return

    images = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]
    # 按照文件名排序，确保拼接顺序正确 (circle_000, circle_001...)
    images.sort()

    if len(images) == 0:
        print("Error: 文件夹中没有找到图片。")
        return
    
    print(f"找到 {len(images)} 张图片，准备拼接...")

    # 2. 获取单张图片的尺寸
    first_image_path = os.path.join(input_folder, images[0])
    with Image.open(first_image_path) as img:
        single_w, single_h = img.size
    
    print(f"单张图片尺寸: {single_w}x{single_h}")

    # 3. 计算大图的总尺寸
    total_w = single_w * grid_cols
    total_h = single_h * grid_rows
    
    print(f"目标大图尺寸: {total_w}x{total_h} (Pixels)")

    # 4. 创建新的空白画布 (RGB模式)
    # 注意：如果显存/内存不足，PIL 可能会报错，但在一般服务器上没问题
    grid_image = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))

    # 5. 开始拼接
    # 限制处理图片的数量，不超过 grid_rows * grid_cols (72)
    max_images = grid_rows * grid_cols
    
    for index, filename in enumerate(tqdm(images[:max_images], desc="拼接中")):
        try:
            # 计算当前图片应该在的 行(row) 和 列(col)
            row_idx = index // grid_cols
            col_idx = index % grid_cols

            # 计算粘贴坐标 (x, y)
            x_pos = col_idx * single_w
            y_pos = row_idx * single_h

            # 打开图片并粘贴
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                grid_image.paste(img, (x_pos, y_pos))
                
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")

    # 6. 保存图片
    print(f"正在保存大图到: {output_path} ...")
    # optimize=True 会尝试压缩文件大小但不降低像素分辨率，PNG 是无损的
    grid_image.save(output_path, format='PNG', optimize=True)
    print("完成！")

if __name__ == "__main__":
    # === 配置区域 ===
    
    # 输入图片的文件夹路径 (根据你之前的输出修改这里)
    # 例如: output/DNA-Rendering/0007_04/novelview/ours_25000/renders_circle_72
    input_dir = "/media/image/mxz/human/SeqAvatar/output/DNA-Rendering/0007_04/novelview/ours_25000/renders_circle_72"
    
    # 输出文件名
    output_file = "grid_result_6x12.png"
    
    # 运行函数
    create_image_grid(input_dir, output_file, grid_rows=6, grid_cols=12)