#%%
import pandas as pd
from tqdm import tqdm
import copy
import ast
import pickle
import os

if not os.path.exists('preprocessed_feature/netSurfP_before.pkl'):
    before_data = pd.read_csv('preprocessed_feature/netSurfP_before.csv')
    for column in before_data.columns:
        if column in ['no', 'seq']:
            continue
        before_data[column] = before_data[column].apply(ast.literal_eval)
    with open('preprocessed_feature/netSurfP_before.pkl', 'wb') as f:
        pickle.dump(before_data, f)

if not os.path.exists('preprocessed_feature/netSurfP_after.pkl'):
    after_data = pd.read_csv('preprocessed_feature/netSurfP_after.csv')
    for column in after_data.columns:
        if column in ['no', 'seq']:
            continue
        after_data[column] = after_data[column].apply(ast.literal_eval)
    with open('preprocessed_feature/netSurfP_after.pkl', 'wb') as f:
        pickle.dump(after_data, f)


# Load the data
with open('preprocessed_feature/netSurfP_before.pkl', 'rb') as f:
    before_data = pickle.load(f)

df_1 = pd.read_csv('preprocessed_feature/single_mutation_step1.csv')
df_final = copy.deepcopy(df_1)

df_final['netSurfP_before_local'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    chunk = df_1.iloc[bat:bat + 1]
    pos = int(chunk.iloc[0]['new_mutation_index'])
    before_row = before_data[before_data['no'] == bat]
    temp_value = []
    if not before_row.empty:
        for column in before_row.columns:
            if column in ['no', 'seq']:
                continue
            temp_value.append(before_row[column].values[0][pos])
        df_final.at[bat, 'netSurfP_before_local'] = temp_value

with open('preprocessed_feature/netSurfP_after.pkl', 'rb') as f:
    after_data = pickle.load(f)

df_1 = copy.deepcopy(df_final)
df_final = copy.deepcopy(df_1)

df_final['netSurfP_after_local'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    chunk = df_1.iloc[bat:bat + 1]
    pos = int(chunk.iloc[0]['new_mutation_index'])
    after_row = after_data[after_data['no'] == bat]
    temp_value = []
    if not after_row.empty:
        for column in after_row.columns:
            if column in ['no', 'seq']:
                continue
            temp_value.append(after_row[column].values[0][pos])
        df_final.at[bat, 'netSurfP_after_local'] = temp_value

with open('preprocessed_feature/netSurfP_local.pkl', 'wb') as f:
    pickle.dump(df_final, f)

with open('preprocessed_feature/netSurfP_before.pkl', 'rb') as f:
    before_data = pickle.load(f)
with open('preprocessed_feature/netSurfP_local.pkl', 'rb') as f:
    df_1 = pickle.load(f)

df_final = copy.deepcopy(df_1)

df_final['netSurfP_before_global'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    chunk = df_1.iloc[bat:bat + 1]
    pos = int(chunk.iloc[0]['new_mutation_index'])
    before_row = before_data[before_data['no'] == bat]
    temp_value = []
    if not before_row.empty:
        temp_value = []

        for column in before_row.columns:
            if column in ['no', 'seq']:
                continue

            feature_values = []

            for offset in range(-4, 5):
                index = pos + offset
                if 0 <= index < len(before_row[column].values[0]):
                    feature_values.append(before_row[column].values[0][index])
                else:
                    feature_values.append(-1)

            temp_value.append(feature_values)
        df_final.at[bat, 'netSurfP_before_global'] = temp_value

with open('preprocessed_feature/netSurfP_after.pkl', 'rb') as f:
    after_data = pickle.load(f)
df_1 = copy.deepcopy(df_final)
df_final = copy.deepcopy(df_1)


df_final['netSurfP_after_global'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    chunk = df_1.iloc[bat:bat + 1]
    pos = int(chunk.iloc[0]['new_mutation_index'])
    after_row = after_data[after_data['no'] == bat]
    temp_value = []
    if not after_row.empty:
        temp_value = []

        for column in after_row.columns:
            if column in ['no', 'seq']:
                continue

            feature_values = []

            for offset in range(-4, 5):
                index = pos + offset
                if 0 <= index < len(after_row[column].values[0]):
                    feature_values.append(after_row[column].values[0][index])
                else:
                    feature_values.append(-1)

            temp_value.append(feature_values)
        df_final.at[bat, 'netSurfP_after_global'] = temp_value
with open('preprocessed_feature/all_feature_netSurfP.pkl', 'wb') as f:
    pickle.dump(df_final, f)


directory = "preprocessed_feature/esm2/point_representations_before"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):

    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/ESM_point_representations_before.pkl")

directory = "preprocessed_feature/esm2/point_representations_after"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):
    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/ESM_point_representations_after.pkl")

directory = "preprocessed_feature/esm2/sequence_representations_before"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):
    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/ESM_sequence_representations_before.pkl")

directory = "preprocessed_feature/esm2/sequence_representations_after"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):

    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/ESM_sequence_representations_after.pkl")



with open('preprocessed_feature/ESM_point_representations_before.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
with open('preprocessed_feature/all_feature_netSurfP.pkl', 'rb') as f:
    df_1 = pickle.load(f)

df_final = copy.deepcopy(df_1)

df_final['ESM_Point_before'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'ESM_Point_before'] = before_row['representation'].values[0]

df_final['ESM_Point_after'] = None
with open('preprocessed_feature/ESM_point_representations_after.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'ESM_Point_after'] = before_row['representation'].values[0]

df_final['ESM_Seq_before'] = None
with open('preprocessed_feature/ESM_sequence_representations_before.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'ESM_Seq_before'] = before_row['representation'].values[0]

df_final['ESM_Seq_after'] = None
with open('preprocessed_feature/ESM_sequence_representations_after.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'ESM_Seq_after'] = before_row['representation'].values[0]

df_final.to_pickle("preprocessed_feature/all_feature_net_esm.pkl")


directory = "preprocessed_feature/protT5/sequence_representations_after"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):

    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()


        no_value = file_name.split('.')[0].split('_')[0]


        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/T5_sequence_representations_after.pkl")

directory = "preprocessed_feature/protT5/sequence_representations_before"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):

    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()
        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/T5_sequence_representations_before.pkl")

directory = "preprocessed_feature/protT5/point_representations_before"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):
    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/T5_point_representations_before.pkl")

directory = "preprocessed_feature/protT5/point_representations_after"
pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
data_list = []
for file_name in tqdm(pickle_files):
    with open(os.path.join(directory, file_name), 'rb') as f:
        data_dict = pickle.load(f)
        representation_list = data_dict["representation"].tolist()

        no_value = file_name.split('.')[0].split('_')[0]

        data_list.append({
            "no": no_value,
            "representation": representation_list
        })
df = pd.DataFrame(data_list)
df.to_pickle("preprocessed_feature/T5_point_representations_after.pkl")

with open('preprocessed_feature/T5_point_representations_before.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
with open('preprocessed_feature/all_feature_net_esm.pkl', 'rb') as f:
    df_1 = pickle.load(f)

df_final = copy.deepcopy(df_1)

df_final['T5_Point_before'] = None
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'T5_Point_before'] = before_row['representation'].values[0]

df_final['T5_Point_after'] = None
with open('preprocessed_feature/T5_point_representations_after.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'T5_Point_after'] = before_row['representation'].values[0]

df_final['T5_Seq_before'] = None
with open('preprocessed_feature/T5_sequence_representations_before.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'T5_Seq_before'] = before_row['representation'].values[0]

df_final['T5_Seq_after'] = None
with open('preprocessed_feature/T5_sequence_representations_after.pkl', 'rb') as f:
    before_data = pickle.load(f)
before_data['no'] = before_data['no'].apply(lambda x: int(x))
for bat in tqdm(range(0, len(df_1), 1)):
    before_row = before_data[before_data['no'] == bat]
    if not before_row.empty:
        df_final.at[bat, 'T5_Seq_after'] = before_row['representation'].values[0]

df_final.to_pickle("preprocessed_feature/final_preprocessed_result.pkl")

