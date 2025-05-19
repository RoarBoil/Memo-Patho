import pandas as pd
from tqdm import tqdm
import os


df_1 = pd.read_csv('preprocessed_feature/mutation_step1.csv')

# generate fasta file
co = 0
for i, sequence in tqdm(enumerate(df_1['mutated_sequence'])):
    if i % 800 == 0:
        if i != 0:
            out_file.close()
        out_file = open('preprocessed_feature/fasta_split/after/after_' + str(co) + '.fasta',
                        'a+')
        co += 1
    out_file.write('>' + str(i) + '\n')
    out_file.write(sequence + '\n')
out_file.close()


co = 0
for i, sequence in tqdm(enumerate(df_1['trimmed_sequence'])):
    if i % 800 == 0:
        if i != 0:
            out_file.close()
        out_file = open('preprocessed_feature/fasta_split/before/before_' + str(co) + '.fasta',
                        'a+')
        co += 1
    out_file.write('>' + str(i) + '\n')
    out_file.write(sequence + '\n')
out_file.close()

for filename in os.listdir('preprocessed_feature/fasta_split/before'):
    # 检查文件扩展名是否为.fasta
    if filename.endswith('.fasta'):
        if not os.path.exists('../tools/NetSurfP-3.0_standalone/before.sh'):
            with open('../tools/NetSurfP-3.0_standalone/before.sh', 'w') as f:
                f.write('#!/bin/bash\n')
        with open('../tools/NetSurfP-3.0_standalone/before.sh', 'a+') as f:
            in_file_path = os.path.join('preprocessed_feature/fasta_split/before', filename)
            out_file_no = filename.split('.')[0].split('_')[-1]
            f.write(f"python nsp3.py -m models/nsp3.pth -i {in_file_path} -o preprocessed_feature/netSurfP/ -w be{out_file_no} -gpu cuda:1\n")

for filename in os.listdir('preprocessed_feature/fasta_split/after'):
    # 检查文件扩展名是否为.fasta
    if filename.endswith('.fasta'):
        if not os.path.exists('../tools/NetSurfP-3.0_standalone/after.sh'):
            with open('../tools/NetSurfP-3.0_standalone/after.sh', 'w') as f:
                f.write('#!/bin/bash\n')
        with open('../tools/NetSurfP-3.0_standalone/after.sh', 'a+') as f:
            in_file_path = os.path.join('preprocessed_feature/fasta_split/after', filename)
            out_file_no = filename.split('.')[0].split('_')[-1]
            f.write(f"python nsp3.py -m models/nsp3.pth -i {in_file_path} -o preprocessed_feature/netSurfP/ -w af{out_file_no} -gpu cuda:0\n")

