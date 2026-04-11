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
    """解析表头，返回蛋白质ID、突变信息（在子序列中的位置、突变后的氨基酸）"""
    parts = header.split()
    mutation_info = parts[0].split('_')[-1]
    protein_id = parts[0].split('_')[0]
    mutation_position_in_subseq = int(parts[1])  # 在截取的序列中的位置
    mutated_aa = mutation_info[-1]  # 突变后的氨基酸
    return protein_id, mutation_position_in_subseq, mutated_aa


def mutate_sequence(sequence, position_in_subseq, mutated_aa):
    """在给定位置将序列的氨基酸改为突变后的氨基酸"""
    # position_in_subseq 是基于1的位置，Python的索引从0开始，因此需要减1
    pos_index = position_in_subseq - 1
    # 将序列中的氨基酸替换为突变后的氨基酸
    mutated_sequence = sequence[:pos_index] + mutated_aa + sequence[pos_index+1:]
    return mutated_sequence


def write_fasta(output_file, sequences):
    """将突变后的序列写入新的fasta文件，不换行"""
    with open(output_file, 'w') as file:
        for header, sequence in sequences:
            file.write(f"{header}\n")
            file.write(f"{sequence}\n")


def process_fasta(fasta_file, output_file):
    """处理fasta文件，将所有序列中的突变位点修改为突变后的氨基酸"""
    # 读取fasta文件中的表头和序列
    sequences = read_fasta(fasta_file)
    
    mutated_sequences = []
    
    for header, sequence in sequences:
        # 解析表头信息，获取突变相关的数据
        protein_id, mutation_position_in_subseq, mutated_aa = parse_header(header)
        
        # 在给定位置将序列中的氨基酸修改为突变后的氨基酸
        mutated_sequence = mutate_sequence(sequence, mutation_position_in_subseq, mutated_aa)
        
        # 保存修改后的表头和序列
        mutated_sequences.append((header, mutated_sequence))
    
    # 将修改后的序列写入新的fasta文件
    write_fasta(output_file, mutated_sequences)
    print(f"已将修改后的序列保存到 {output_file}")


# 示例用法
fasta_file1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/origin.fasta'  # 输入的fasta文件路径
output_file1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/mutated.fasta'  # 输出的fasta文件路径
process_fasta(fasta_file1, output_file1)