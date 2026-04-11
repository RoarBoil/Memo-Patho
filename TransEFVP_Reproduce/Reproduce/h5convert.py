import h5py
import numpy as np

def convert_h5_to_npy(h5_file_path, npy_output_path):
    # 打开 .h5 文件
    with h5py.File(h5_file_path, 'r') as h5_file:
        embeddings = []

        # 遍历所有的键（蛋白质序列ID）
        for seq_id in h5_file.keys():
            # 读取每个蛋白质序列的嵌入
            embedding = np.array(h5_file[seq_id])
            embeddings.append(embedding)

        # 将嵌入数据保存为 .npy 文件
        np.save(npy_output_path, embeddings, allow_pickle=True)
        print(f"Saved embeddings to {npy_output_path}")

# 使用示例
h5_file_path1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_origin.h5'  # 输入的 .h5 文件路径
npy_output_path1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_origin.npy'  # 输出的 .npy 文件路径

convert_h5_to_npy(h5_file_path1, npy_output_path1)

h5_file_path2 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_mutated.h5'  # 输入的 .h5 文件路径
npy_output_path2 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_mutated.npy'  # 输出的 .npy 文件路径

convert_h5_to_npy(h5_file_path2, npy_output_path2)