


import numpy as np
ac = np.load('/data2/guest1/files_for_test/protT5/T5local.npy', allow_pickle=True)
print(ac[0])  # 查看数组的形状
#print(ac)        # 打印数组内容

# 用于存储每个蛋白质的均值
mean_data_list = []

# 遍历每个蛋白质序列，计算其均值
for data in ac:
    mean_data = np.mean(data, axis=0, keepdims=True)  # 计算均值
    mean_data_list.append(mean_data)  # 将均值加入结果列表

# 将结果转换为数组（如果你需要）
mean_data_array = np.array(mean_data_list)

# 保存处理后的均值数据
np.save('/data2/guest1/files_for_test/mean/T5local.npy', mean_data_array)

# 打印处理后的数据形状，确认转换结果
print(mean_data_array.shape)

'''
import os
import numpy as np

def process_npy_files(input_dir, output_dir):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入文件夹中的所有.npy文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            ac = np.load(file_path, allow_pickle=True)  # 加载.npy文件
            
            mean_data_list = []
            for data in ac:
                if data.ndim > 0:  # 检查数组是否有多维
                    mean_data = np.mean(data, axis=0, keepdims=True)  # 计算多维数组的均值
                else:
                    mean_data = data  # 如果是一维或零维数组，则直接使用原始数据
                
                mean_data_list.append(mean_data)
            
            mean_data_array = np.array(mean_data_list)  # 转换为数组
            
            # 保存处理后的数据到目标文件夹
            output_file_path = os.path.join(output_dir, file_name)  # 使用相同的文件名
            np.save(output_file_path, mean_data_array)
            
            # 打印信息确认处理
            print(f"Processed {file_name}, shape: {mean_data_array.shape}")

# 使用该函数
input_dir = '/data2/guest1/trainfiles/fortrain_2x1'
output_dir = '/data2/guest1/trainfiles/fortrain_2x1/mean'
process_npy_files(input_dir, output_dir)
'''