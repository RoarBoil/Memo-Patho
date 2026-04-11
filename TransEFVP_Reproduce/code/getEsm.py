import torch
import esm
# Load ESM-1b model
# If you need ESM-1v or ESM-2 model, replace esm1b_t33_650M_UR50S with esm1v_t33_650M_UR90S_1 or esm2_t33_650M_UR50D
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()


def esm(sequence):
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations[0][1:len(sequence)+1]
    return token_representations.numpy()
