import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from src.load_data import MutationsDataset
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, f1_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score
)
from model_final.model_supervised import SupervisedClassificationModel
from model_final.model_contra import TransformerBasedFusionNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('dataset/train_mix.pkl', 'rb') as f:
    train_data = pickle.load(f)

kf = KFold(n_splits=10, shuffle=True, random_state=200)

val_losses = []
val_accuracies = []
val_mccs = []
val_f1s = []
val_precisions = []
val_recalls = []
val_aucs = []
conf_matrix_values = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(f"Fold {fold + 1}/10")
    train_fold = train_data.iloc[train_idx]
    val_fold = train_data.iloc[val_idx]

    train_dataset = MutationsDataset(train_fold)
    val_dataset = MutationsDataset(val_fold)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    contrastive_model = TransformerBasedFusionNet()
    contrastive_model.load_state_dict(torch.load('model_weight/model_contra_mix.pth'))
    supervised_model = SupervisedClassificationModel(contrastive_model)
    supervised_model = supervised_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(supervised_model.parameters(), lr=1e-3)

    num_epochs = 30
    for epoch in range(num_epochs):
        supervised_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = supervised_model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        supervised_model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                labels = batch["label"].to(device)

                outputs = supervised_model(batch)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_mcc = matthews_corrcoef(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='binary')
        val_precision = precision_score(all_labels, all_preds, average='binary')
        val_recall = recall_score(all_labels, all_preds, average='binary')

        val_auc = roc_auc_score(all_labels, all_probs)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = conf_matrix.ravel()

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
              f"MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
              f"AUC: {val_auc:.4f}, Conf Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

