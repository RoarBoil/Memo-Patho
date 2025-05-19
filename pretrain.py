import pandas as pd
import copy
import pickle

from torch.utils.data import DataLoader
from src.load_data import ContrastiveLearningDataset
from model.model_contra import TransformerBasedFusionNet
import torch.optim as optim
import torch.nn as nn
import torch
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('/data3/baoyh/Memo/test_center/dataset/train_mix.pkl', 'rb') as f:
    df_1 = pickle.load(f)
df_1.reset_index(drop=True, inplace=True)


with open('/data3/baoyh/Memo/model_final/triples_train_mix.pkl', 'rb') as f:
    triples_train = pickle.load(f)


train_dataset = ContrastiveLearningDataset(df_1, triples_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerBasedFusionNet().to(device)
criterion = nn.TripletMarginLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for anchor_data, positive_data, negative_data in train_loader:
        optimizer.zero_grad()

        anchor_data = {key: value.to(device) for key, value in anchor_data.items()}
        positive_data = {key: value.to(device) for key, value in positive_data.items()}
        negative_data = {key: value.to(device) for key, value in negative_data.items()}

        anchor_output = model(anchor_data, branch='before')
        positive_output = model(positive_data, branch='after')
        negative_output = model(negative_data, branch='after')

        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    model.eval()

    all_embeddings = []
    all_labels = []

    for anchor_data, positive_data, negative_data in train_loader:

        anchor_data = {key: value.to(device) for key, value in anchor_data.items()}
        positive_data = {key: value.to(device) for key, value in positive_data.items()}
        negative_data = {key: value.to(device) for key, value in negative_data.items()}

        anchor_output = model(anchor_data, branch='before').detach().cpu()
        positive_output = model(positive_data, branch='after').detach().cpu()
        negative_output = model(negative_data, branch='after').detach().cpu()

        pos_combined = torch.cat((anchor_output, positive_output), dim=1)
        neg_combined = torch.cat((anchor_output, negative_output), dim=1)

        pos_labels = torch.ones(pos_combined.size(0), dtype=torch.long).cpu()
        neg_labels = torch.zeros(neg_combined.size(0), dtype=torch.long).cpu()

        all_embeddings.append(pos_combined)
        all_embeddings.append(neg_combined)
        all_labels.append(pos_labels)
        all_labels.append(neg_labels)

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

