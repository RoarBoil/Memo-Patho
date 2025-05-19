import torch
import torch.nn as nn


class SupervisedClassificationModel(nn.Module):
    def __init__(self, contrastive_model):
        super().__init__()

        self.local_before = contrastive_model.local_before
        self.global_before = contrastive_model.global_before
        self.pos_encoder_before = contrastive_model.pos_encoder_before
        self.transformer_before = contrastive_model.transformer_before

        self.local_after = contrastive_model.local_after
        self.global_after = contrastive_model.global_after
        self.pos_encoder_after = contrastive_model.pos_encoder_after
        self.transformer_after = contrastive_model.transformer_after

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, data):
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
        fusion_after, attn_after = self.transformer_after(torch.cat([local_after, global_after], dim=1).permute(1, 0, 2), return_attn=True)
        fusion_after = fusion_after.mean(dim=0)

        fused = torch.cat([fusion_before, fusion_after], dim=1)

        logits = self.classifier(fused)
        # return logits, fused
        # return logits, fusion_before, fusion_after
        # return logits, attn_after
        return logits