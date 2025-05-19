import torch
import esm
import pandas as pd
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

df_1 = pd.read_csv('preprocessed_feature/mutation_step1.csv')

for bat in range(0, len(df_1), 1):
    chunk = df_1.iloc[bat:bat+1]
    data = []
    for ifx, row in chunk.iterrows():
        data.append((row['accession'] + str(row['begin']) + '_origin', row['mutated_sequence']))

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.cuda()
    model = model.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=True)
    del batch_tokens
    torch.cuda.empty_cache()
    token_representations = results["representations"][36].cpu()


    for i, tokens_len in enumerate(batch_lens):
        sequence_representations = {'accession': chunk.iloc[0]['accession'], 'sequence': chunk.iloc[0]['mutated_sequence'], 'pos': int(chunk.iloc[0]['new_mutation_index']), 'mutated': chunk.iloc[0]['mutatedType'],'clinicalSignificances': int(chunk.iloc[0]['clinicalSignificances']),'representation': token_representations[i, 1: tokens_len - 1].mean(0).numpy()}
        point_representations = {'accession': chunk.iloc[0]['accession'], 'sequence': chunk.iloc[0]['mutated_sequence'],'clinicalSignificances': int(chunk.iloc[0]['clinicalSignificances']),
          'pos': int(chunk.iloc[0]['new_mutation_index']), 'mutated': chunk.iloc[0]['mutatedType'],
          'representation': token_representations[i, 1 + chunk.iloc[0]['new_mutation_index']].numpy()}
        with open('preprocessed_feature/esm2/sequence_representations_after/' + str(bat) + '_' + str(sequence_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(sequence_representations, f)
        with open('preprocessed_feature/esm2/point_representations_after/' + str(bat) + '_' + str(point_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(point_representations, f)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()


df_1 = pd.read_csv('preprocessed_feature/mutation_step1.csv')

for bat in range(0, len(df_1), 1):
    chunk = df_1.iloc[bat:bat+1]
    data = []
    for ifx, row in chunk.iterrows():
        data.append((row['accession'] + str(row['begin']) + '_origin', row['trimmed_sequence']))

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.cuda()
    model = model.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=True)
    del batch_tokens
    torch.cuda.empty_cache()
    token_representations = results["representations"][36].cpu()

    for i, tokens_len in enumerate(batch_lens):
        sequence_representations = {'accession': chunk.iloc[0]['accession'], 'sequence': chunk.iloc[0]['trimmed_sequence'], 'pos': int(chunk.iloc[0]['new_mutation_index']), 'mutated': chunk.iloc[0]['mutatedType'],'clinicalSignificances': int(chunk.iloc[0]['clinicalSignificances']),'representation': token_representations[i, 1: tokens_len - 1].mean(0).numpy()}
        point_representations = {'accession': chunk.iloc[0]['accession'], 'sequence': chunk.iloc[0]['trimmed_sequence'],'clinicalSignificances': int(chunk.iloc[0]['clinicalSignificances']),
          'pos': int(chunk.iloc[0]['new_mutation_index']), 'mutated': chunk.iloc[0]['mutatedType'],
          'representation': token_representations[i, 1 + chunk.iloc[0]['new_mutation_index']].numpy()}
        with open('preprocessed_feature/esm2/sequence_representations_before/' + str(bat) + '_' + str(sequence_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(sequence_representations, f)
        with open('preprocessed_feature/esm2/point_representations_before/' + str(bat) + '_' + str(point_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(point_representations, f)
