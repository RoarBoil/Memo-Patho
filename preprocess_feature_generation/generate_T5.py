from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
import os
import pickle


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

df_1 = pd.read_csv('preprocessed_feature/mutation_step1.csv')
co = 0


for bat in range(0, len(df_1), 72):
    chunk = df_1.iloc[bat:bat+72]
    data = []
    for ifx, row in chunk.iterrows():
        data.append( row['mutated_sequence'])
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data]
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    for i, sequence in enumerate(sequence_examples):
        sequence_representations = {'accession': chunk.iloc[i]['accession'], 'sequence': chunk.iloc[i]['mutated_sequence'], 'pos': int(chunk.iloc[i]['new_mutation_index']), 'mutated': chunk.iloc[i]['mutatedType'],'clinicalSignificances': int(chunk.iloc[i]['clinicalSignificances']),'representation': embedding_repr.last_hidden_state[i, :len(sequence)].mean(0).cpu().numpy()}
        point_representations = {'accession': chunk.iloc[i]['accession'], 'sequence': chunk.iloc[i]['mutated_sequence'],'clinicalSignificances': int(chunk.iloc[i]['clinicalSignificances']),
          'pos': int(chunk.iloc[i]['new_mutation_index']), 'mutated': chunk.iloc[i]['mutatedType'],
          'representation': embedding_repr.last_hidden_state[i, chunk.iloc[i]['new_mutation_index']].cpu().numpy()}
        with open('preprocessed_feature/protT5/sequence_representations_after/' + str(bat + i) + '_' + str(sequence_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(sequence_representations, f)
        with open('preprocessed_feature/protT5/point_representations_after/' + str(bat + i) + '_' + str(point_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(point_representations, f)
    co += 1
    print(co)


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)

model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

df_1 = pd.read_csv('preprocessed_feature/mutation_step1.csv')
co = 0

for bat in range(0, len(df_1), 72):
    chunk = df_1.iloc[bat:bat+72]
    data = []
    for ifx, row in chunk.iterrows():
        data.append( row['trimmed_sequence'])
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data]
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    for i, sequence in enumerate(sequence_examples):
        sequence_representations = {'accession': chunk.iloc[i]['accession'], 'sequence': chunk.iloc[i]['trimmed_sequence'], 'pos': int(chunk.iloc[i]['new_mutation_index']), 'mutated': chunk.iloc[i]['mutatedType'],'clinicalSignificances': int(chunk.iloc[i]['clinicalSignificances']),'representation': embedding_repr.last_hidden_state[i, :len(sequence)].mean(0).cpu().numpy()}
        point_representations = {'accession': chunk.iloc[i]['accession'], 'sequence': chunk.iloc[i]['trimmed_sequence'],'clinicalSignificances': int(chunk.iloc[i]['clinicalSignificances']),
          'pos': int(chunk.iloc[i]['new_mutation_index']), 'mutated': chunk.iloc[i]['mutatedType'],
          'representation': embedding_repr.last_hidden_state[i, chunk.iloc[i]['new_mutation_index']].cpu().numpy()}
        with open('preprocessed_feature/protT5/sequence_representations_before/' + str(bat + i) + '_' + str(sequence_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(sequence_representations, f)
        with open('preprocessed_feature/protT5/point_representations_before/' + str(bat + i) + '_' + str(point_representations['clinicalSignificances']) + '.pkl', 'wb') as f:
            pickle.dump(point_representations, f)

