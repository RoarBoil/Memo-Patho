import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_emb = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        positions = self.position_emb(self.position_ids[:, :seq_len])
        return x + positions


class TransformerBlock(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[3].weight)

    def forward(self, src, return_attn=False, **kwargs):
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(
            src2, src2, src2,
            need_weights=return_attn,
            average_attn_weights=False,
            **kwargs
        )
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + self.dropout(src2)
        return (src, attn_weights) if return_attn else src

class TransformerBasedFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_before = nn.Sequential(
            nn.Linear(3600, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self.global_before = nn.Sequential(
            nn.Linear(3728, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self.pos_encoder_before = PositionalEncoding(256)
        self.transformer_before = TransformerBlock(256)

        self.local_after = nn.Sequential(
            nn.Linear(3600, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self.global_after = nn.Sequential(
            nn.Linear(3728, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self.pos_encoder_after = PositionalEncoding(256)
        self.transformer_after = TransformerBlock(256)

        self.final_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, data, branch='before'):
        if branch == 'before':
            local_before = torch.cat([data['netSurfP_before_local'],
                                      data['ESM_Point_before'],
                                      data['T5_Point_before']], dim=1)
            local_before = self.local_before(local_before)

            global_before = data['netSurfP_before_global'].flatten(1)
            global_before = torch.cat([global_before, data['ESM_Seq_before']], dim=1)
            global_before = torch.cat([global_before, data['T5_Seq_before']], dim=1)
            global_before = self.global_before(global_before)

            local_before = self.pos_encoder_before(local_before.unsqueeze(1))
            global_before = self.pos_encoder_before(global_before.unsqueeze(1))
            fusion_before = self.transformer_before(torch.cat([local_before, global_before], dim=1).permute(1, 0, 2))
            fusion_before = fusion_before.mean(dim=0)

            embedding = self.final_projection(fusion_before)
            return embedding

        elif branch == 'after':
            local_after = torch.cat([data['netSurfP_after_local'],
                                     data['ESM_Point_after'],
                                     data['T5_Point_after']], dim=1)
            local_after = self.local_after(local_after)

            global_after = data['netSurfP_after_global'].flatten(1)
            global_after = torch.cat([global_after, data['ESM_Seq_after']], dim=1)
            global_after = torch.cat([global_after, data['T5_Seq_after']], dim=1)
            global_after = self.global_after(global_after)

            local_after = self.pos_encoder_after(local_after.unsqueeze(1))
            global_after = self.pos_encoder_after(global_after.unsqueeze(1))
            fusion_after = self.transformer_after(torch.cat([local_after, global_after], dim=1).permute(1, 0, 2))
            fusion_after = fusion_after.mean(dim=0)

            embedding = self.final_projection(fusion_after)
            return embedding

        else:
            raise ValueError("Invalid branch value. Use 'before' or 'after'.")