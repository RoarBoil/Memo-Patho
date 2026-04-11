import numpy as np
import torch
import esm  # 请确保你已经安装了ESM库
# If you need ESM-1v or ESM-2 model, replace esm1b_t33_650M_UR50S with esm1v_t33_650M_UR90S_1 or esm2_t33_650M_UR50D
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并将其移动到GPU
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)  # 将模型放到GPU上
batch_converter = alphabet.get_batch_converter()

def esm_embed(sequence):
    """
    对给定的蛋白质序列使用ESM模型进行嵌入。
    返回嵌入向量，并转回CPU转换为NumPy数组。
    """
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)  # 将输入放到GPU上
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    # 获取与原始序列对应的嵌入（跳过首尾特殊标记）
    token_representations = token_representations[0][1:len(sequence) + 1]
    return token_representations.cpu().numpy()  # 转回CPU并转换为NumPy数组

def read_fasta(fasta_file):
    """读取fasta文件，返回表头和序列的列表"""
    with open(fasta_file, 'r') as file:
        sequences = []
        header = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # 表头
                if header:  # 如果已经有表头，说明之前的序列处理完了
                    sequences.append((header, ''.join(sequence)))
                header = line
                sequence = []
            else:  # 序列行
                sequence.append(line)
        if header:  # 处理最后一个序列
            sequences.append((header, ''.join(sequence)))
    return sequences

def parse_header(header):
    """解析表头，返回蛋白质ID和突变位置"""
    parts = header.split()
    protein_id = parts[0].split('_')[0]  # 蛋白质ID
    mutation_position_in_subseq = int(parts[1])  # 在序列中的位置
    return protein_id, mutation_position_in_subseq

def embed_protein_sequences(original_fasta_file, mutated_fasta_file, original_output_file, mutated_output_file):
    """读取突变前后的fasta文件，只对突变位点的氨基酸进行嵌入"""
    # 读取突变前和突变后的fasta文件中的表头和序列
    original_sequences = read_fasta(original_fasta_file)
    mutated_sequences = read_fasta(mutated_fasta_file)
    
    original_embeddings = []
    mutated_embeddings = []
    
    # 确保两个文件的顺序和数量是相同的
    if len(original_sequences) != len(mutated_sequences):
        raise ValueError("突变前和突变后的序列数量不匹配。")
    
    for (original_header, original_sequence), (mutated_header, mutated_sequence) in zip(original_sequences, mutated_sequences):
        # 解析表头信息，获取突变位置
        _, mutation_position_in_subseq = parse_header(original_header)
        
        # 获取突变位点的氨基酸
        original_amino_acid = original_sequence[mutation_position_in_subseq - 1]  # 突变前的氨基酸
        mutated_amino_acid = mutated_sequence[mutation_position_in_subseq - 1]    # 突变后的氨基酸
        
        # 对突变前的单个氨基酸进行嵌入
        original_embedding = esm_embed(original_amino_acid)  # 对单个氨基酸进行嵌入
        original_embeddings.append(original_embedding)  # 保存嵌入结果
        
        # 对突变后的单个氨基酸进行嵌入
        mutated_embedding = esm_embed(mutated_amino_acid)  # 对单个氨基酸进行嵌入
        mutated_embeddings.append(mutated_embedding)  # 保存嵌入结果

    # 将突变前后的嵌入结果保存到指定文件
    np.save(original_output_file, np.array(original_embeddings))
    np.save(mutated_output_file, np.array(mutated_embeddings))
    print(f"原始嵌入保存到 {original_output_file}")
    print(f"突变后嵌入保存到 {mutated_output_file}")
'''
# 示例用法
original_fasta_file = '/data2/guest1/trainfiles/fortrain/mix_shuffled.fasta'  # 输入的突变前fasta文件路径
mutated_fasta_file = '/data2/guest1/trainfiles/fortrain/mix_shuffled_mutated.fasta'  # 输入的突变后fasta文件路径
original_output_file = '/data2/guest1/trainfiles/fortrain/esm/esm1b.npy'
mutated_output_file = '/data2/guest1/trainfiles/fortrain/esm/esm1b_mutated.npy'
'''

original_fasta_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/origin.fasta'  # 输入的突变前fasta文件路径
mutated_fasta_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/mutated.fasta'  # 输入的突变后fasta文件路径
original_output_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm2.npy'
mutated_output_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/esm/esm2_mutated.npy'
# 进行嵌入并保存
embed_protein_sequences(original_fasta_file, mutated_fasta_file, original_output_file, mutated_output_file)

