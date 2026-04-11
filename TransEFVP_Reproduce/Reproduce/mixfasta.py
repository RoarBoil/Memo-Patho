from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import random

# 定义输入和输出文件路径
benign_fasta = '/data2/guest1/TransEFVP-master/data/201sequences/training_set_ben.fasta'  # 良性蛋白质序列文件路径
pathogenic_fasta = '/data2/guest1/TransEFVP-master/data/201sequences/training_set_path.fasta'  # 致病性蛋白质序列文件路径
output_fasta_mixed = '/data2/guest1/trainfiles/mixTest_all.fasta'  # 混合后的所有序列输出路径
output_label_npy_mixed = '/data2/guest1/trainfiles/test_log_all.npy'  # 混合后的所有标签输出路径

# 加载良性蛋白质序列
benign_sequences = list(SeqIO.parse(benign_fasta, 'fasta'))
benign_labels = np.zeros(len(benign_sequences))  # 良性标注为 0

# 加载致病性蛋白质序列
pathogenic_sequences = list(SeqIO.parse(pathogenic_fasta, 'fasta'))
pathogenic_labels = np.ones(len(pathogenic_sequences))  # 致病性标注为 1

# 合并所有的序列和标签
all_sequences = benign_sequences + pathogenic_sequences
all_labels = np.concatenate((benign_labels, pathogenic_labels), axis=0)

# 打乱序列和标签
combined = list(zip(all_sequences, all_labels))
random.shuffle(combined)

# 解压混合后的序列和标签
shuffled_sequences, shuffled_labels = zip(*combined)

# 保存混合后的序列到新的 fasta 文件
with open(output_fasta_mixed, 'w') as output_handle:
    SeqIO.write(shuffled_sequences, output_handle, 'fasta')

# 保存标签到 numpy 文件
np.save(output_label_npy_mixed, np.array(shuffled_labels))

print(f"所有混合序列已保存到 {output_fasta_mixed}")
print(f"对应的标签已保存到 {output_label_npy_mixed}")
