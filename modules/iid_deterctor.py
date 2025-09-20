import os
import datetime
import logging
import random

import numpy as np
import yaml
from sklearn import metrics
from typing import Union
from collections import defaultdict

from modules.utils.iid_api import FC_ddp,FC_ddp2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from modules.base_detector import AbstractDetector
from modules.utils.iid_api import l2_norm
from modules.networks.xception import Xception

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

class IIDDetector(AbstractDetector):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.BCE_LOSS = FC_ddp(512, 2).cuda()
    
    def build_backbone(self):
        backbone = Xception()
        state_dict = torch.load('./checkpoints/xception-b5690688.pth')
        
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)

        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = f'backbone.{k}'
            new_state_dict[new_key] = v
        
        backbone.load_state_dict(new_state_dict, strict=False)
        logger.info('Load pretrained model successfully!')
        return backbone
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor,id_f=None) -> torch.tensor:
        return self.backbone.classifier(features,id_f)

    def get_train_loss(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].cuda()                  
        pred = pred_dict['cls'].cuda()                     
        source_index = data_dict['source_index'].cuda()  
        target_index = data_dict['target_index'].cuda()  
        id_feat = pred_dict['id_feat'].cuda()    
        embed = pred_dict['embed'].cuda()  
        flag = data_dict['flag'].cuda() 
        envID = data_dict['env_id'].cuda()   

        real_id = (label == 0)
        fake_id = (label == 1)
        im_embs = l2_norm(embed)
        em_embs = l2_norm(id_feat)
        loss = 0

        self.IIE_LOSS.update_kernel(
            embeddings=im_embs, 
            target_indices=target_index,
            label=label,
            envID=envID
        )

        cos_sim = F.cosine_similarity(embed, id_feat, dim=1)

        target_sim = torch.ones_like(cos_sim)

        f2f_nt_mask = (flag == 1)
        target_sim[fake_id & ~f2f_nt_mask] = 0
        

        loss_ce = F.binary_cross_entropy_with_logits(
            cos_sim, 
            target_sim,
            reduction='mean'
        )
        loss += loss_ce

        scale = 2.7  
        real_losses = []
        if real_id.any():
            sim3 = (im_embs[real_id] * em_embs[real_id]).sum(dim=1)
            real_losses.append(torch.exp(scale * (1 - sim3)))

        if fake_id.any():
            fake_target_indices = target_index[fake_id]     # [num_fake]
            fake_source_indices = source_index[fake_id]     # [num_fake]
            valid_target_mask = torch.tensor([idx.item() not in self.missing_ids 
                                    for idx in fake_target_indices]).cuda()
            valid_source_mask = torch.tensor([idx.item() not in self.missing_ids 
                            for idx in fake_source_indices]).cuda()
            
            valid_mask = valid_target_mask & valid_source_mask
            
            if valid_mask.any():
                valid_fake_embs = im_embs[fake_id][valid_mask]
                valid_target_indices = fake_target_indices[valid_mask]
                valid_source_indices = fake_source_indices[valid_mask]
                fake_flags = flag[fake_id][valid_mask]
                fake_envIDs = envID[fake_id][valid_mask]
                other_mask = (fake_flags == 0)

                fake_losses = []    
                iie_losses = []
                if other_mask.any():
                    valid_fake_embs_other = valid_fake_embs[other_mask]
                    valid_target_indices_other = valid_target_indices[other_mask]
                    valid_source_indices_other = valid_source_indices[other_mask]
                    vaild_fake_envID_other = fake_envIDs[other_mask]
                    new_iie_loss = self.IIE_LOSS.compute_new_loss(
                        valid_fake_embs_other,
                        valid_target_indices_other,
                        valid_source_indices_other,
                        vaild_fake_envID_other
                    )
                    iie_losses.append(new_iie_loss)
                    target_features = []
                    for target_id, env_id in zip(valid_target_indices_other, vaild_fake_envID_other):
                        sub_em_kernel = self.em_kernel.get(target_id.item())
                        if sub_em_kernel is not None and env_id < sub_em_kernel.size(1):
                            target_feature = sub_em_kernel[:, env_id]
                            target_features.append(target_feature)
                    target_explicit = torch.stack(target_features)
                    fake_implicit_explicit_dist = F.cosine_similarity(
                        valid_fake_embs_other.cuda(),
                        target_explicit.cuda(),
                        dim=1
                    )
                    
                    fake_losses.append(torch.exp(scale * (1 - fake_implicit_explicit_dist)))
        iiic_losses = []
        new_iiic_loss = self.IIE_LOSS.compute_IIIC_loss(im_embs, target_index, envID, self.em_kernel)
        iiic_losses.append(new_iiic_loss)
        k = min(4, torch.sum(real_id).item())
        flat_real_losses = [item for sublist in real_losses for item in sublist]
        topk_losses, _ = torch.topk(torch.tensor(flat_real_losses), k, sorted=False)
        real_loss1 = topk_losses.mean()
        real_loss2 = (torch.cat(real_losses).mean() if real_losses else 
                    torch.tensor(0.0).cuda())
        real_loss = 0.6 * real_loss1 + 0.4 * real_loss2
        fake_loss = (torch.cat(fake_losses).mean() if fake_losses else 
                    torch.tensor(0.0).cuda())
        
        loss_eic = fake_loss + real_loss
        loss += 0.9 * loss_eic
        
        
        iie_loss = (torch.cat(iie_losses).mean() if iie_losses else 
                    torch.tensor(0.0).cuda())
        loss += 0.8 * iie_loss

        iiic_loss = (torch.cat(iiic_losses).mean() if iiic_losses else torch.tensor(0.0).cuda())
        loss += 0.9 * iiic_loss

        loss_dict = {'overall': loss,'loss_bce': loss_ce, 'loss_iie': iie_loss, 'loss_eic': loss_eic, 'loss_eic_fake': fake_loss, 'loss_eic_real': real_loss, 'loss_iiic': iiic_loss}
        return loss_dict

    def forward(self, image_dict: dict, inference=False) -> dict:
        features = self.features(image_dict)
        pred = self.classifier(features, image_dict['id_feat'])         
        embed=self.backbone.last_emb
        return embed
