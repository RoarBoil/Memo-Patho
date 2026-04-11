import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


seq_path = "/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/origin.fasta"

per_residue = False
per_residue_path = ""  # where to store the embeddings

per_protein = True
per_protein_path = "/data2/guest2/TransEFVP-master/Last_train/test_on_KCNQ1/ProtT5/T5_Global_origin.h5"  # where to store the embeddings

sec_struct = False
sec_struct_path = "" # file for storing predictions
assert per_protein is True or per_residue is True or sec_struct is True, print(
    "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat


def load_sec_struct_model():
    checkpoint_dir = "./protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)
    print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

    return model


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def read_fasta(fasta_path, split_char=" ", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char=" " and id_field=0 for your FASTA headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()  # 用字典存储所有的序列
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # 如果是标头行（以 > 开头）
            if line.startswith('>'):
                # 解析标头并提取ID
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                
                # 处理 ID 中的特殊字符，避免在保存为 HDF5 文件时出现问题
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                
                # 在字典中添加这个蛋白质的 ID，初始化序列为空
                seqs[uniprot_id] = ''
            else:
                # 如果是序列行，处理并拼接多行的序列
                seq = ''.join(line.split()).upper().replace("-", "")
                
                # 替换非标准氨基酸（如 'U', 'Z', 'O'）为 'X'
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                
                # 将该序列拼接到之前的条目中
                seqs[uniprot_id] += seq
    
    # 打印读取的序列数和示例
    example_id = next(iter(seqs))  # 获取字典中的第一个条目
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id, seqs[example_id]))

    return seqs



import logging

# 配置日志
logging.basicConfig(filename='embedding_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=21000, max_seq_len=1000, max_batch=100):
    if sec_struct:
        sec_struct_model = load_sec_struct_model()

    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()

    # 遍历序列
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        try:
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((pdb_id, seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                # add_special_tokens adds extra token at the end of each sequence
                token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest", truncation=True, max_length=model.config.n_positions)
                input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)

                if sec_struct:  # in case you want to predict secondary structure from embeddings
                    d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

                for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                    s_len = seq_lens[batch_idx]
                    # slice off padding --> batch-size x seq_len x embedding_dim
                    emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                    if sec_struct:  # get classification results
                        results["sec_structs"][identifier] = torch.max(d3_Yhat[batch_idx, :s_len], dim=1)[
                            1].detach().cpu().numpy().squeeze()
                    if per_residue:  # store per-residue embeddings (Lx1024)
                        results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                    if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                        protein_emb = emb.mean(dim=0)
                        results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

                # Log the successful processing of the current batch
                logging.info(f"Successfully processed protein {pdb_id} with sequence length {seq_len}")

        except RuntimeError:
            logging.error(f"RuntimeError during embedding for {pdb_id} (L={len(seq)})")
            continue
        except Exception as e:
            logging.error(f"Unexpected error during embedding for {pdb_id} (L={len(seq)}): {str(e)}")
            continue

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')

    # Log the total number of processed sequences
    logging.info(f"Processed {len(results['protein_embs'])} proteins.")
    
    return results



def save_embeddings(emb_dict, out_path):
    with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # 记录每个序列被保存
            logging.info(f"Saving embedding for {sequence_id}")
            hf.create_dataset(sequence_id, data=embedding)



def write_prediction_fasta(predictions, out_path):
    class_mapping = {0:"H",1:"E",2:"L"}
    with open(out_path, 'w+') as out_f:
        out_f.write( '\n'.join(
            [ ">{}\n{}".format(
                seq_id, ''.join( [class_mapping[j] for j in yhat] ))
            for seq_id, yhat in predictions.items()
            ]
              ) )
    return None


# Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
model, tokenizer = get_T5_model()

# Load example fasta.
seqs = read_fasta(seq_path)

# Compute embeddings and/or secondary structure predictions
results = get_embeddings(model, tokenizer, seqs,
                         per_residue, per_protein, sec_struct)

# Store per-residue embeddings
if per_residue:
    save_embeddings(results["residue_embs"], per_residue_path)
if per_protein:
    save_embeddings(results["protein_embs"], per_protein_path)
if sec_struct:
    write_prediction_fasta(results["sec_structs"], sec_struct_path)
