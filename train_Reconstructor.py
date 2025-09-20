import os
import cv2
import time
import argparse
import torch.nn as nn
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from traindata.dataset import FaceDataset
from utils import laplacian_blending, make_image
from modules.encoder128 import Backbone128
from modules.se import SE
from modules.aii_generator import AII512
from modules.decoder512 import UnetDecoder512
from modules.discriminator import MultiscaleDiscriminator
from preprocess.mtcnn import MTCNN
from modules.encoder128 import load_custom_state_dict
from tqdm import tqdm
import random
import os
import torch
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def init_distributed():
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    torch.cuda.set_device(local_rank)
    
    return local_rank

TRANSFORMS = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()

def set_seed(seed):
    random.seed(seed)            
    np.random.seed(seed)           
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_images(Xs, Xt, Y, save_dir, epoch, batch_idx):
    """
    Save input images Xs, Xt, and generated image Y to a single image.
    Each sample in the batch will be saved separately.
    
    Args:
        Xs, Xt, Y: shape [batch_size, 3, 512, 512] tensors
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(Xs.size(0)):  # Iterate over each sample in the batch
        # Get the i-th sample
        xs_i = Xs[i:i+1]  # Keep 4D shape: [1, 3, 512, 512]
        xt_i = Xt[i:i+1]
        y_i = Y[i:i+1]
        
        # Process single sample
        I = [xs_i, xt_i, y_i]
        image = make_image(I, 1)
        
        # Include sample index when saving
        save_path = os.path.join(save_dir, f'generate.png')
        cv2.imwrite(save_path, image.transpose([1, 2, 0]),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def calculate_IID_loss(Xs_id_initial, Xt_id_initial, Y_id_initial, Xs, Xt, Y, iid_discriminator):
    """
    Calculate IID discriminator loss based on explicit and implicit feature comparison
    
    Args:
        Xs_id_initial: source image's original identity feature [batch_size, 512]
        Xt_id_initial: target image's original identity feature [batch_size, 512]
        Xs: source image
        Xt: target image
        Y: generated image
        iid_discriminator: IID discriminator model
        
    return:
        loss_IID: calculated IID loss
    """
    # Normalize source and target identity features
    Xs_id_norm = F.normalize(Xs_id_initial, p=2, dim=1)
    Xt_id_norm = F.normalize(Xt_id_initial, p=2, dim=1)
    y_id_norm = F.normalize(Y_id_initial, p=2, dim=1)
    sim_Xs_D_Xs = F.cosine_similarity(Xs_id_norm, F.normalize(iid_discriminator({"image": Xs, "id_feat": Xs_id_initial}), p=2, dim=1), dim=1)
    sim_Xt_D_Xt = F.cosine_similarity(Xt_id_norm, F.normalize(iid_discriminator({"image": Xt, "id_feat": Xt_id_initial}), p=2, dim=1), dim=1)
    sim_Y_D_Y_Xs = F.cosine_similarity(Xs_id_norm, F.normalize(iid_discriminator({"image": Y.detach(), "id_feat": Xs_id_initial}), p=2, dim=1), dim=1)
    sim_Y_D_Xt_D_Y = F.cosine_similarity(F.normalize(iid_discriminator({"image": Xt, "id_feat": Xt_id_initial}), p=2, dim=1), F.normalize(iid_discriminator({"image": Y.detach(), "id_feat": Xs_id_initial}), p=2, dim=1), dim=1)
    sim_Y_D_Y = F.cosine_similarity(y_id_norm, F.normalize(iid_discriminator({"image": Y.detach(), "id_feat": Xs_id_initial}), p=2, dim=1), dim=1)
    
    # Calculate IID loss
    loss = -2 * (sim_Xs_D_Xs.mean() + sim_Xt_D_Xt.mean()) + torch.mean(torch.abs(sim_Y_D_Y)) - 1.1 * sim_Y_D_Xt_D_Y.mean()
    return loss


def inference(save_dir, se=None, decoder=None, G=None, encoder=None):
    """
    :param encoder: backbone encoder model
    :return: no return
    """
    local_rank = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    os.makedirs(save_dir, exist_ok=True)
    writer = None
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(save_dir, 'runs'))

    dataset = FaceDataset(
        root=args.data_root,
        transform=TRANSFORMS,
        cache_file='dataset_cache_500persons_hq.pkl',
        max_persons=500
    )

    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    discriminator = MultiscaleDiscriminator(
        input_nc=3, 
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        num_D=3,
        getIntermFeat=False
    ).to(device)
    discriminator.train() 
    for param in discriminator.parameters():
        param.requires_grad = True

    iid_discriminator = IIDDetector().to(device)
    iid_discriminator.train()
    for param in iid_discriminator.parameters():
        param.requires_grad = True
    se = se.to(device)
    decoder = decoder.to(device)
    G = G.to(device)
    encoder = encoder.to(device)
    se = DDP(se, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    G = DDP(G, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    iid_discriminator = DDP(iid_discriminator, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

    for param in encoder.parameters():
        if param.device != device:
            param.data = param.data.to(device)

    d_optimizer = Adam(discriminator.parameters(), 
                    lr=args.lr, 
                    betas=(0.5, 0.999),
                    eps=1e-8)  

    """ load pre-calculated mean and std: """
    param_dict = []
    for i in range(N + 1):
        state = torch.load(f'./modules/weights128/readout_layer{i}.pth', map_location=device)
        n_samples = state['n_samples'].float()
        std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
        neuron_nonzero = state['neuron_nonzero'].float()
        active_neurons = (neuron_nonzero / n_samples) > 0.01
        param_dict.append([state['m'].to(device), std, active_neurons.to(device)])

    gd_optimizer = Adam([
        {'params': G.parameters()},
        {'params': decoder.parameters()}
    ], lr=args.lr)

    iid_optimizer = Adam([
        {'params': iid_discriminator.parameters()}
    ], lr=args.lr,
        betas=(0.5, 0.999),
        eps=1e-8)  

    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=True, disable=local_rank != 0)
        se.eval()
        G.train()
        decoder.train()
        for param in se.parameters():
            param.requires_grad = False
        for param in G.parameters():
            param.requires_grad = True
        for param in decoder.parameters():
            param.requires_grad = True
        
        current_optimizer = gd_optimizer
        
        
        for batch_idx, (Xs, Xt) in enumerate(pbar):
            torch.cuda.empty_cache()
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            B = Xs.size(0)
            X_id = encoder(
                F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                                mode='bilinear', align_corners=True),
                cache_feats=True
            )
            # 01 Get Inter-features After One Feed-Forward:
            # batch size is 2 * B, [:B] for Xs and [B:] for Xt
            min_std = torch.tensor(0.01, device=device)
            readout_feats = [(encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                                for i in range(N + 1)]

            # 02 information restriction:
            X_id_restrict = torch.zeros_like(X_id).to(device)  # [2*B, 512]
            Xt_feats = []
            Xt_lambda = []
            for i in range(N):
                R = encoder.features[i]  # [2*B, Cr, Hr, Wr]
                Z, lambda_, _ = getattr(se.module, f'iba_{i}')(
                    R, readout_feats,
                    m_r=param_dict[i][0], std_r=param_dict[i][1],
                    active_neurons=param_dict[i][2],
                )
                
                X_id_restrict += encoder.restrict_forward(Z, i)
                Rs = R[:B]
                lambda_t = lambda_[B:]
                m_s = torch.mean(Rs, dim=0)  # [C, H, W]
                std_s = torch.mean(Rs, dim=0)

                eps_s = torch.randn(size=Rs.shape).to(Rs.device) * std_s + m_s
                # eps_t = torch.randn(size=R[B:].shape).to(R[B:].device) * std_t + m_t
                feat_t = R[B:] * (1. - lambda_t) + lambda_[B:] * eps_s
                Xt_feats.append(feat_t)
                Xt_lambda.append(lambda_t)

            X_id_restrict /= float(N)
            Xs_id, Xt_id = X_id_restrict[:B], X_id_restrict[B:]
            # Xs_id_initial, Xt_id_initial = X_id[:B], X_id[B:]
            Xt_feats[0] = Xt
            Xt_attr, Xt_attr_lamb = decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True, first_use_img=True)
            Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
            Y = torch.clamp(Y, -1, 1) 
            encoder.features = []

            Y_id = encoder(
                F.interpolate(torch.cat((Y, Y), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                                mode='bilinear', align_corners=True),
                cache_feats=True
            )
            
            min_std = torch.tensor(0.01, device=device)
            readout_feats = [(encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                                for i in range(N + 1)]
            
            _, _, Y_feats, _, Y_lambda_double, _, _ = se(
                        encoder, 
                        B, 
                        N, 
                        readout_feats, 
                        param_dict
                    )
            real_pred = discriminator(Xt)
            fake_pred = discriminator(Y.detach())
            d_loss = 0
            for real_pred_i, fake_pred_i in zip(real_pred, fake_pred):
                real_target = torch.ones_like(real_pred_i[0]).to(device)
                fake_target = torch.zeros_like(fake_pred_i[0]).to(device)
                real_loss = F.mse_loss(real_pred_i[0], real_target)
                fake_loss = F.mse_loss(fake_pred_i[0], fake_target)
                d_loss += (real_loss + fake_loss) * 0.5

            d_loss /= len(real_pred)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.max_grad_norm)
            d_optimizer.step()
            iid_optimizer.zero_grad()
            loss_IID = calculate_IID_loss(X_id[:B], X_id[B:], Y_id[:B], Xs, Xt, Y, iid_discriminator)
            loss_IID.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(iid_discriminator.parameters(), max_norm=args.max_grad_norm)
            iid_optimizer.step()
            
            fake_pred = discriminator(Y)
            loss_adv = 0
            for fake_pred_i in fake_pred:
                loss_adv += F.mse_loss(fake_pred_i[0], torch.ones_like(fake_pred_i[0]))
            loss_adv /= len(fake_pred)
            loss_IID_adv = 0
            with torch.no_grad():
                D_Xt = iid_discriminator({"image": Xt, "id_feat": X_id[B:]})
                D_Xt_norm = F.normalize(D_Xt, p=2, dim=1)
            D_Y = iid_discriminator({"image": Y, "id_feat": Y_id[:B]})
            D_Y_norm = F.normalize(D_Y, p=2, dim=1)
            sim_D_Xt_D_Y = F.cosine_similarity(D_Xt_norm, D_Y_norm, dim=1)
            loss_IID_adv = torch.mean(torch.abs(sim_D_Xt_D_Y))
            Y_feats[0] = Y

            l_per = 0
            layer = len(Y_feats)
            for j in range(layer):
                l_per += torch.mean((Y_feats[j] - Xt_feats[j]) ** 2)
            l_per /= layer
            l_pos = -F.cosine_similarity(Y_id[:B], X_id[:B]).mean()
            cos_yt = torch.stack([F.cosine_similarity(Y_id[i:i+1], X_id_restrict[B+i:B+i+1]) 
                                for i in range(B)]).squeeze()
            cos_st = torch.stack([F.cosine_similarity(X_id[i:i+1], X_id[B+i:B+i+1]) 
                                for i in range(B)]).squeeze()
            mask = cos_st < 0.3
            if mask.any():
                valid_cos_yt = cos_yt[mask]
                valid_cos_st = cos_st[mask]
                l_neg = torch.mean((valid_cos_yt - valid_cos_st) ** 2)
            else:
                l_neg = torch.tensor(0.0, device=cos_st.device)

            l_icl = 0.5 * l_neg + 2 * l_pos
            g_loss = args.lambda_adv * loss_adv + 1.6 * loss_IID_adv + args.lambda_per * l_per + args.lambda_icl * l_icl
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                torch.cuda.empty_cache()
                continue
            current_optimizer.zero_grad()
            g_loss.backward()

            
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.max_grad_norm)
            
            current_optimizer.step()
            if batch_idx % 10 == 0:
                save_images(Xs, Xt, Y, save_dir=args.save_dir)
            
            if batch_idx % 100 == 0 and writer is not None:
                global_step = epoch * len(train_loader) + batch_idx

                writer.add_scalar('Loss/D_loss', d_loss.item(), global_step)
                writer.add_scalar('Loss/G_loss', g_loss.item(), global_step)
                writer.add_scalar('Loss/L_pos', l_pos.item(), global_step)
                writer.add_scalar('Loss/L_neg', l_neg.item(), global_step)
                writer.add_scalar('Loss/Loss_adv', loss_adv.item(), global_step)
                writer.add_scalar('Loss/Loss_IID_adv', loss_IID_adv.item(), global_step)
                writer.add_scalar('Loss/L_per', l_per.item(), global_step)
                writer.add_scalar('Loss/Loss_IID', loss_IID.item(), global_step)

            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'batch': f'{batch_idx}/{len(train_loader)}',
                'L_pos': f'{l_pos.item():.4f}',
                'L_neg': f'{l_neg.item():.4f}',
                "Loss_adv": f'{loss_adv.item():.4f}',
                "Loss_IID_adv": f'{loss_IID_adv.item():.4f}',
                "Loss_IID": f'{loss_IID.item():.4f}',
                "L_per": f'{l_per.item():.4f}',
                "cos_yt":f'{cos_yt.item() if cos_yt.dim() == 0 else cos_yt[0].item():.4f}',
                "cos_st":f'{cos_st.item() if cos_st.dim() == 0 else cos_st[0].item():.4f}'
            })

            if (batch_idx+1) % 1000 == 0:
                save_path = f'E://FaceShield/checkpoints_IRC/{epoch}_{batch_idx}.pth'
                torch.save({
                    'epoch': epoch,
                    'se': se.state_dict(),
                    'G': G.state_dict(),
                    'decoder': decoder.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'iid_discriminator': iid_discriminator.state_dict(),
                    'gd_optimizer': gd_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'iid_optimizer': iid_optimizer.state_dict(),
                }, save_path)
                print(f'\nSaved checkpoint to {save_path}')
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    
    p.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lambda_info', type=float, default=1.0)
    p.add_argument('--lambda_id', type=float, default=1.0) 
    p.add_argument('--lambda_per', type=float, default=1.0)
    p.add_argument('--lambda_adv', type=float, default=3.3)
    p.add_argument('--lambda_icl', type=float, default=1.0)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm for clipping')
    p.add_argument('-save', '--save_dir', type=str, default='E://FaceShield/generateImages/IRC')
    p.add_argument('--data_root', type=str, default=r'E://FaceShield/content/FF_HQ')
    args = p.parse_args()

    """ Prepare Models: """
    encoder = Backbone128(50, 0.6, 'ir_se').eval()
    original_state_dict = torch.load('modules/model_128_ir_se50.pth', map_location='cpu')
    load_custom_state_dict(encoder, original_state_dict)
        
    G = AII512().train()
    decoder = UnetDecoder512().train()
    N = 10
    with torch.no_grad():
        _ = encoder(torch.rand(1, 3, 128, 128), cache_feats=True)
        _readout_feats = encoder.features[:(N + 1)]
        in_c = sum(map(lambda f: f.shape[-3], _readout_feats))
        out_c_list = [_readout_feats[i].shape[-3] for i in range(N)]
        encoder.features = []
    checkpoint = torch.load("E://FaceShield/checkpoints_IRC/model.pth")
    se_model = SE(in_c, out_c_list, 'cuda', kernel_size=1)
    se_model = se_model.eval()
    se_model.load_state_dict(new_checkpoint['se'])
    G.load_state_dict(torch.load(os.path.join(root, pathG)), strict=False)
    decoder.load_state_dict(torch.load(os.path.join(root, pathE)), strict=False)
    inference(args.save_dir, se_model, decoder, G, encoder)

