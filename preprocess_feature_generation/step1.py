import json
import warnings
import pandas as pd
from tqdm import tqdm
import os
import copy
import pickle
import requests
from multiprocessing import Pool, cpu_count



df_1 = pd.read_csv('preprocessed_feature/mutation_dataset.csv')


def check_residue(row):
    original_residue = row['wildType']
    position = int(row['begin']) - 1
    return row['sequence'][position] != original_residue

def check_position(row):
    return row['begin'] > len(row['sequence'])

df_2 = df_1

position_check = df_2[df_2.apply(check_position, axis=1)]
df_2 = df_2[~df_2.index.isin(position_check.index)]

inconsistent_rows = df_2[df_2.apply(check_residue, axis=1)]
if inconsistent_rows.shape[0] > 0:
    print("Inconsistent rows found!")
    print(inconsistent_rows)
else:
    print("No inconsistent rows found.")

df_2 = df_2[df_2['sequence'].str.len() >= 40]
def trim_sequence(row):
    seq = row['sequence']
    begin = int(row['begin']) - 1
    if len(seq) > 501:
        start = max(0, begin - 250)
        end = start + 501

        if end > len(seq):
            end = len(seq)
            start = end - 501
            start = max(0, start)
        return seq[start:end], start, end
    return seq, 0, len(seq)
df_2[['trimmed_sequence', 'new_start', 'new_end']] = df_2.apply(trim_sequence, axis=1, result_type='expand')

def check_mutation(row):
    seq = row['trimmed_sequence']
    original_begin = int(row['begin']) - 1
    new_start = row['new_start']
    new_mutation_index = original_begin - new_start
    if 0 <= new_mutation_index < len(seq) and seq[new_mutation_index] == row['wildType']:
        return True, new_mutation_index
    return False, new_mutation_index

df_2[['mutation_check', 'new_mutation_index']] = df_2.apply(check_mutation, axis=1, result_type='expand')

def apply_mutation(row):
    seq = row['trimmed_sequence']
    original_begin = int(row['begin']) - 1
    new_start = row['new_start']
    new_mutation_index = original_begin - new_start
    if row['mutation_check']:
        seq = seq[:new_mutation_index] + row['mutatedType'] + seq[new_mutation_index + 1:]
    return seq

df_2['mutated_sequence'] = df_2.apply(apply_mutation, axis=1)

df_3 = df_2[df_2['mutation_check'] == True]

df_3.to_csv('preprocessed_feature/mutation_step1.csv', index=False)