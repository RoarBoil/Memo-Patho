import pandas as pd
from tqdm import tqdm
import os


root_dir = 'preprocessed_feature/netSurfP'
all_dataframes = []
for entry in os.listdir(root_dir):
    if not entry.startswith('af'):
        continue
    sub_dir_path = os.path.join(root_dir, entry)
    print(sub_dir_path)
    for dir_name in tqdm(os.listdir(sub_dir_path)):
        if dir_name.endswith('.txt'):
            continue
        file_path = os.path.join(sub_dir_path, dir_name + '/' + dir_name + '.csv')
        df = pd.read_csv(file_path)
        df.drop(columns=['id', ' n', ' q3', ' q8'], inplace=True)
        agg_result = pd.DataFrame({
            'no': [int(dir_name.split('_')[1])],
            'seq': [''.join(df[' seq'])],
            'rsa': [list(df[' rsa'])],
            'asa': [list(df[' asa'])],
            'p[q3_H]': [list(df[' p[q3_H]'])],
            'p[q3_E]': [list(df[' p[q3_E]'])],
            'p[q3_C]': [list(df[' p[q3_C]'])],
            'p[q8_G]': [list(df[' p[q8_G]'])],
            'p[q8_H]': [list(df[' p[q8_H]'])],
            'p[q8_I]': [list(df[' p[q8_I]'])],
            'p[q8_B]': [list(df[' p[q8_B]'])],
            'p[q8_E]': [list(df[' p[q8_E]'])],
            'p[q8_S]': [list(df[' p[q8_S]'])],
            'p[q8_T]': [list(df[' p[q8_T]'])],
            'p[q8_C]': [list(df[' p[q8_C]'])],
            'phi': [list(df[' phi'])],
            'psi': [list(df[' psi'])],
            'disorder': [list(df[' disorder'])]
        })
        if type(all_dataframes) == list:
            all_dataframes = agg_result
        else:
            all_dataframes = pd.concat([all_dataframes, agg_result], ignore_index=True)
    # break
all_dataframes.to_csv('preprocessed_feature/netSurfP_after.csv', index=False)


root_dir = 'preprocessed_feature/netSurfP'
all_dataframes = []
for entry in os.listdir(root_dir):
    if not entry.startswith('be'):
        continue
    sub_dir_path = os.path.join(root_dir, entry)
    print(sub_dir_path)
    for dir_name in tqdm(os.listdir(sub_dir_path)):
        if dir_name.endswith('.txt'):
            continue
        file_path = os.path.join(sub_dir_path, dir_name + '/' + dir_name + '.csv')
        df = pd.read_csv(file_path)
        df.drop(columns=['id', ' n', ' q3', ' q8'], inplace=True)
        agg_result = pd.DataFrame({
            'no': [int(dir_name.split('_')[1])],
            'seq': [''.join(df[' seq'])],
            'rsa': [list(df[' rsa'])],
            'asa': [list(df[' asa'])],
            'p[q3_H]': [list(df[' p[q3_H]'])],
            'p[q3_E]': [list(df[' p[q3_E]'])],
            'p[q3_C]': [list(df[' p[q3_C]'])],
            'p[q8_G]': [list(df[' p[q8_G]'])],
            'p[q8_H]': [list(df[' p[q8_H]'])],
            'p[q8_I]': [list(df[' p[q8_I]'])],
            'p[q8_B]': [list(df[' p[q8_B]'])],
            'p[q8_E]': [list(df[' p[q8_E]'])],
            'p[q8_S]': [list(df[' p[q8_S]'])],
            'p[q8_T]': [list(df[' p[q8_T]'])],
            'p[q8_C]': [list(df[' p[q8_C]'])],
            'phi': [list(df[' phi'])],
            'psi': [list(df[' psi'])],
            'disorder': [list(df[' disorder'])]
        })
        if type(all_dataframes) == list:
            all_dataframes = agg_result
        else:
            all_dataframes = pd.concat([all_dataframes, agg_result], ignore_index=True)
all_dataframes.to_csv('preprocessed_feature/netSurfP_before.csv', index=False)