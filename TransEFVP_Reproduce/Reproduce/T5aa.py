'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_T5_model():
    """加载半精度T5模型和分词器"""
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        torch_dtype=torch.float16  # 显式使用半精度
    )
    model = model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc',
        do_lower_case=False,
        legacy=True  # 使用新版tokenizer行为
    )
    return model, tokenizer

def read_fasta(fasta_file):
    """读取FASTA文件"""
    sequences = []
    with open(fasta_file, 'r') as f:
        current_header, current_sequence = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences.append((current_header, "".join(current_sequence)))
                current_header = line
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_header:
            sequences.append((current_header, "".join(current_sequence)))
    return sequences

def parse_header(header):
    """解析表头获取突变位置"""
    parts = header.split()
    return parts[0].split('_')[0], int(parts[1])  # (protein_id, 1-based位置)

def get_single_amino_acid_embedding(model, tokenizer, sequence, position):
    """获取指定位置的氨基酸嵌入"""
    # 转换序列格式并进行截断
    processed_seq = ' '.join(list(sequence))
    inputs = tokenizer(
        processed_seq,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=10000,
        truncation=True
    )
    
    # 移到GPU并检查位置有效性
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 计算有效位置范围（考虑[CLS]和[SEP]标记）
    max_valid_pos = input_ids.size(1) - 2  # 减去首尾特殊标记
    if position > max_valid_pos:
        raise ValueError(f"位置 {position} 超出有效范围 (最大: {max_valid_pos})")
    
    # 获取嵌入并立即释放显存
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    embedding = outputs.last_hidden_state[0, position].cpu().numpy()
    
    # 显式释放资源
    del input_ids, attention_mask, outputs
    torch.cuda.empty_cache()
    
    return embedding

def embed_mutation_sites(fasta_file, output_file):
    """主处理流程"""
    model, tokenizer = get_T5_model()
    sequences = read_fasta(fasta_file)
    
    embeddings = []
    for header, seq in sequences:
        try:
            _, pos = parse_header(header)
            emb = get_single_amino_acid_embedding(model, tokenizer, seq, pos)
            embeddings.append(emb)
        except Exception as e:
            print(f"处理 {header} 时出错: {str(e)}")
            continue
    
    np.save(output_file, np.array(embeddings))
    print(f"保存嵌入到 {output_file}")

# 使用示例
if __name__ == "__main__":
    fasta_path = '/data2/guest1/TransEFVP-master/Last_train/test_on_testmix/mutated.fasta'
    output_path = '/data2/guest1/TransEFVP-master/Last_train/test_on_testmix/ProtT5/ProtT5_Local_mutated.npy'
    embed_mutation_sites(fasta_path, output_path)


'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

def get_T5_model():
    """加载T5模型和分词器"""
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, legacy=False)
    return model, tokenizer

def read_fasta(fasta_file):
    """读取FASTA文件，返回表头和序列的列表"""
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
    """解析表头，返回蛋白质ID和突变位点的位置"""
    parts = header.split()
    protein_id = parts[0].split('_')[0]  # 蛋白质ID
    mutation_position_in_subseq = int(parts[1])  # 在序列中的位置（1-based）
    return protein_id, mutation_position_in_subseq

def get_single_amino_acid_embedding(model, tokenizer, sequence, position):
    """
    获取序列中指定位置的单个氨基酸的嵌入
    :param sequence: 蛋白质序列
    :param position: 目标氨基酸的位置（1-based）
    """
    # T5模型要求将序列分隔为单个字符并用空格连接
    sequence = ' '.join(list(sequence))
    
    # 对序列进行编码
    token_encoding = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = token_encoding['input_ids'].to(device)
    attention_mask = token_encoding['attention_mask'].to(device)
    
    # 获取嵌入
    with torch.no_grad():
        embedding_repr = model(input_ids, attention_mask=attention_mask)
    
    # 提取目标氨基酸的嵌入
    # position是1-based索引，需要减1转换为0-based索引
    # 但还需要再+1，因为T5模型前面有一个特殊标记，所以需要额外的偏移
    aa_embedding = embedding_repr.last_hidden_state[0, position - 1 + 1, :]  # [0, position, :]
    
    return aa_embedding.cpu().numpy()

def embed_mutation_sites(fasta_file, output_file):
    """对FASTA文件中每个突变位点进行嵌入，并将结果保存到.npy文件中"""
    # 加载T5模型和分词器
    model, tokenizer = get_T5_model()
    
    # 读取FASTA文件
    sequences = read_fasta(fasta_file)
    
    mutation_embeddings = []
    
    for header, sequence in sequences:
        # 解析表头信息，获取突变位置
        _, mutation_position_in_subseq = parse_header(header)
        
        # 获取突变位置的氨基酸嵌入
        aa_embedding = get_single_amino_acid_embedding(model, tokenizer, sequence, mutation_position_in_subseq)
        
        mutation_embeddings.append(aa_embedding)
    
    # 将嵌入向量保存到.npy文件
    np.save(output_file, np.array(mutation_embeddings))
    print(f"突变位点嵌入保存到 {output_file}")

# 示例用法
fasta_file1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/origin.fasta'  # 输入的fasta文件路径
output_file1 = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/ProtT5_Local_origin.npy'  # 嵌入输出文件

# 进行嵌入并保存
embed_mutation_sites(fasta_file1, output_file1)

fasta_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/mutated.fasta'  # 输入的fasta文件路径
output_file = '/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/ProtT5_Local_mutated.npy'  # 嵌入输出文件

# 进行嵌入并保存
embed_mutation_sites(fasta_file, output_file)
