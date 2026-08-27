import pickle
import numpy as np
import sys

# 文件路径设置
input_path = 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
output_path = 'smpl_model/models/SMPL_NEUTRAL.pkl'

print(f"正在读取原始模型: {input_path}")

try:
    # 1. 加载原始 Pickle 文件 (处理编码问题)
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 2. 清理数据：将所有 chumpy 对象转换为纯 numpy 数组
    # SeqAvatar 只需要以下核心参数
    output_data = {}
    
    # 需要保留的关键键值
    keys_to_keep = ['J_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs', 'f', 'kintree_table']
    
    for key in keys_to_keep:
        if key in data:
            # 如果是稀疏矩阵(J_regressor)，先转 dense 再转 numpy，或者直接转 numpy
            val = data[key]
            if hasattr(val, 'toarray'): # 处理稀疏矩阵
                output_data[key] = np.array(val.toarray())
            else:
                output_data[key] = np.array(val)
        else:
            print(f"警告: 原始模型中缺少 {key}，可能导致后续错误。")

    # 3. 保存为新的纯净 Pickle 文件
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"成功！")
    print(f"已生成清洗后的文件: {output_path}")
    print("现在你可以继续进行 SMPL-X 的配置或下一步了。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{input_path}'")
    print("请确认你已经把从官网下载的 basicModel_neutral_...pkl 文件复制到了当前目录！")
except Exception as e:
    print(f"发生未知错误: {e}")