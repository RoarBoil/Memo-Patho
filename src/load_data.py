import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MutationsDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        data = {}
        data['netSurfP_before_local'] = torch.tensor(sample['netSurfP_before_local'], dtype=torch.float32)
        data['netSurfP_after_local'] = torch.tensor(sample['netSurfP_after_local'], dtype=torch.float32)

        data['netSurfP_before_global'] = torch.tensor(np.array(sample['netSurfP_before_global']).reshape(9, 16),
                                                      dtype=torch.float32)
        data['netSurfP_after_global'] = torch.tensor(np.array(sample['netSurfP_after_global']).reshape(9, 16),
                                                     dtype=torch.float32)

        data['ESM_Point_before'] = torch.tensor(sample['ESM_Point_before'], dtype=torch.float32)
        data['ESM_Point_after'] = torch.tensor(sample['ESM_Point_after'], dtype=torch.float32)
        data['ESM_Seq_before'] = torch.tensor(sample['ESM_Seq_before'], dtype=torch.float32)
        data['ESM_Seq_after'] = torch.tensor(sample['ESM_Seq_after'], dtype=torch.float32)
        data['T5_Point_before'] = torch.tensor(sample['T5_Point_before'], dtype=torch.float32)
        data['T5_Point_after'] = torch.tensor(sample['T5_Point_after'], dtype=torch.float32)
        data['T5_Seq_before'] = torch.tensor(sample['T5_Seq_before'], dtype=torch.float32)
        data['T5_Seq_after'] = torch.tensor(sample['T5_Seq_after'], dtype=torch.float32)

        label = int(sample['clinicalSignificances'])
        data['label'] = torch.tensor(label, dtype=torch.long)
        return data


class ContrastiveLearningDataset(Dataset):
    def __init__(self, df, triples):
        self.df = df.reset_index(drop=True)
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        anchor_idx, positive_idx, negative_idx = self.triples[index]

        anchor_sample = self.df.iloc[anchor_idx]
        anchor_data = {
            'netSurfP_before_local': torch.tensor(anchor_sample['netSurfP_before_local'], dtype=torch.float32),
            'netSurfP_before_global': torch.tensor(np.array(anchor_sample['netSurfP_before_global']).reshape(9, 16), dtype=torch.float32),
            'ESM_Point_before': torch.tensor(anchor_sample['ESM_Point_before'], dtype=torch.float32),
            'ESM_Seq_before': torch.tensor(anchor_sample['ESM_Seq_before'], dtype=torch.float32),
            'T5_Point_before': torch.tensor(anchor_sample['T5_Point_before'], dtype=torch.float32),
            'T5_Seq_before': torch.tensor(anchor_sample['T5_Point_before'], dtype=torch.float32)
        }

        positive_sample = self.df.iloc[positive_idx]
        positive_data = {
            'netSurfP_after_local': torch.tensor(positive_sample['netSurfP_after_local'], dtype=torch.float32),
            'netSurfP_after_global': torch.tensor(np.array(positive_sample['netSurfP_after_global']).reshape(9, 16), dtype=torch.float32),
            'ESM_Point_after': torch.tensor(positive_sample['ESM_Point_after'], dtype=torch.float32),
            'ESM_Seq_after': torch.tensor(positive_sample['ESM_Seq_after'], dtype=torch.float32),
            'T5_Point_after': torch.tensor(positive_sample['T5_Point_after'], dtype=torch.float32),
            'T5_Seq_after': torch.tensor(positive_sample['T5_Point_after'], dtype=torch.float32)

        }

        negative_sample = self.df.iloc[negative_idx]
        negative_data = {
            'netSurfP_after_local': torch.tensor(negative_sample['netSurfP_after_local'], dtype=torch.float32),
            'netSurfP_after_global': torch.tensor(np.array(negative_sample['netSurfP_after_global']).reshape(9, 16), dtype=torch.float32),
            'ESM_Point_after': torch.tensor(negative_sample['ESM_Point_after'], dtype=torch.float32),
            'ESM_Seq_after': torch.tensor(negative_sample['ESM_Seq_after'], dtype=torch.float32),
            'T5_Point_after': torch.tensor(negative_sample['T5_Point_after'], dtype=torch.float32),
            'T5_Seq_after': torch.tensor(negative_sample['T5_Point_after'], dtype=torch.float32)
        }

        return anchor_data, positive_data, negative_data