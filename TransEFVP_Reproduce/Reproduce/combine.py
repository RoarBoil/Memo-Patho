import numpy as np

# 加载所有嵌入文件
esm2_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm2.npy', allow_pickle=True) 
esm1b_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm1b.npy', allow_pickle=True)
esm1v_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm1v.npy', allow_pickle=True)
esm2_mutated_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm2_mutated.npy', allow_pickle=True)
esm1b_mutated_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm1b_mutated.npy', allow_pickle=True)
esm1v_mutated_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm1v_mutated.npy', allow_pickle=True)
T5local_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/ProtT5_Local_origin.npy', allow_pickle=True)
T5local_mutated_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/ProtT5_Local_mutated.npy', allow_pickle=True)
T5global_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_origin.npy', allow_pickle=True)
T5global_mutated_embeddings = np.load('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_mutated.npy', allow_pickle=True)

# 获取所有嵌入数组的长度，确保它们相同
embedding_lengths = [
    len(esm2_embeddings), 
    len(esm1b_embeddings), 
    len(esm1v_embeddings), 
    len(esm2_mutated_embeddings), 
    len(esm1b_mutated_embeddings), 
    len(esm1v_mutated_embeddings), 
    len(T5local_embeddings), 
    len(T5local_mutated_embeddings), 
    len(T5global_embeddings), 
    len(T5global_mutated_embeddings)
]

# 确保所有嵌入数组的长度相同
min_length = min(embedding_lengths)
print(f"All embeddings have been truncated to the length: {min_length}")

# 用于存储合并后的 [3, 1280] 数组
combined_embeddings1 = []
combined_embeddings2 = []

# 遍历每个蛋白质嵌入向量，确保不会超出数组的最大索引
for i in range(min_length):
    # 取出每个文件的第i个嵌入向量，并检查形状
    esm2_embedding = esm2_embeddings[i].reshape(1, 1280)
    esm1v_embedding = esm1v_embeddings[i].reshape(1, 1280)
    esm1b_embedding = esm1b_embeddings[i].reshape(1, 1280)
    esm1v_mutated_embedding = esm1v_mutated_embeddings[i].reshape(1, 1280)   
    esm1b_mutated_embedding = esm1b_mutated_embeddings[i].reshape(1, 1280)
    esm2_mutated_embedding = esm2_mutated_embeddings[i].reshape(1, 1280)
    T5local_embedding = T5local_embeddings[i].reshape(1, 1024)
    T5local_mutated_embedding = T5local_mutated_embeddings[i].reshape(1, 1024)
    T5global_embedding = T5global_embeddings[i].reshape(1, 1024)
    T5global_mutated_embedding = T5global_mutated_embeddings[i].reshape(1, 1024)
    
    # 合并不同的嵌入向量
    combined_embedding1 = np.vstack([T5global_embedding, T5local_embedding, T5global_mutated_embedding, T5local_mutated_embedding])
    combined_embedding2 = np.vstack([esm2_embedding, esm1v_embedding, esm1b_embedding, esm2_mutated_embedding, esm1v_mutated_embedding, esm1b_mutated_embedding])
    
    # 将合并后的结果加入列表
    combined_embeddings1.append(combined_embedding1)
    combined_embeddings2.append(combined_embedding2)

# 将结果转换为数组并保存
combined_embeddings_array1 = np.array(combined_embeddings1)
np.save('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/combined_T5.npy', combined_embeddings_array1)
combined_embeddings_array2 = np.array(combined_embeddings2)
np.save('/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/combined_esm.npy', combined_embeddings_array2)

# 打印处理后的形状，确保为 (n, 3, 1280)，其中 n 是蛋白质数量 
print(f"Combined T5 embeddings shape: {combined_embeddings_array1.shape}")
print(f"Combined ESM embeddings shape: {combined_embeddings_array2.shape}")
