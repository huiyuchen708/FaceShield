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



def frobenius_cosine_similarity(A, B):
    batch_size, channels, height, width = A.shape
    cosine_similarities = []
    
    for c in range(1, channels):
        A_c = A[:, c, :, :]
        B_c = B[:, c, :, :]
        A_c = F.normalize(A_c.view(batch_size, -1), p=2, dim=1).view(batch_size, height, width)
        B_c = F.normalize(B_c.view(batch_size, -1), p=2, dim=1).view(batch_size, height, width)
        frob_inner_product = torch.sum(A_c * B_c, dim=(1, 2))
        norm_A = torch.norm(A_c, p='fro', dim=(1, 2))
        norm_B = torch.norm(B_c, p='fro', dim=(1, 2))
        cos_sim = frob_inner_product / (norm_A * norm_B + 1e-8)
        if torch.isnan(cos_sim).any():
            print("cos_sim[{}] is nan".format(c))
        cosine_similarities.append(cos_sim)
    mean_cosine_similarity = torch.mean(torch.stack(cosine_similarities), dim=0)
    return mean_cosine_similarity

def calculate_fd_loss_angle_optimized(Z_id, Xs_feats, N):
    fd_losses = []
    for i in range(1, N):
        theta_Z_Xs_cos = frobenius_cosine_similarity(Z_id[i], Xs_feats[i])
        theta_Z_Xs = torch.acos(torch.clamp(theta_Z_Xs_cos, -1.0, 1.0))
        theta_Z_Xs = torch.where(theta_Z_Xs > math.pi/2, 
                                math.pi - theta_Z_Xs,
                                theta_Z_Xs)
        
        angle_diff = - theta_Z_Xs
        fd_losses.append(angle_diff)
    mean_fd_loss = torch.mean(torch.stack(fd_losses))
    return mean_fd_loss

def gaussian_kernel(X, sigma=None):
    n = X.size(0)
    batch_size = 128
    K = torch.zeros((n, n), device=X.device)
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        X_i = X[i:end_i]
        
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            X_j = X[j:end_j]
            dist_ij = torch.cdist(X_i, X_j, p=2).pow(2)
            if sigma is None:
                med = torch.median(dist_ij.view(-1))
                sigma = torch.sqrt(med / 2)
            if sigma == 0:
                sigma = torch.tensor(1e-5, device=X.device)
            K[i:end_i, j:end_j] = torch.exp(-dist_ij / (2 * sigma * sigma))
            del dist_ij
            torch.cuda.empty_cache()
    
    return K

def calculate_fd_loss_hsic_optimized(Z_id, X_feats, N):
    ind_losses = []
    
    for i in range(1, N):
        batch_size, C_z, H, W = Z_id[i].shape
        C_x = X_feats[i].shape[1]      
        
        batch_losses = []
        for b in range(batch_size):
            Z_sample = Z_id[i][b]  # [C_z, H, W]
            X_sample = X_feats[i][b]  # [C_x, H, W]
            Z_reshaped = Z_sample.permute(1, 2, 0).reshape(-1, C_z)  # [H*W, C_z]
            X_reshaped = X_sample.permute(1, 2, 0).reshape(-1, C_x)  # [H*W, C_x]
            n = Z_reshaped.shape[0]
            K_Z = gaussian_kernel(Z_reshaped) 
            K_X = gaussian_kernel(X_reshaped) 
            H = torch.eye(n, device=Z_reshaped.device) - torch.ones(n, n, device=Z_reshaped.device) / n
            K_Z_centered = torch.matmul(torch.matmul(H, K_Z), H)
            K_X_centered = torch.matmul(torch.matmul(H, K_X), H)
            hsic_value = torch.sum(K_Z_centered * K_X_centered) / ((n-1)**2) 
            ind = torch.log(torch.tensor(C_z, dtype=torch.float, device=Z_reshaped.device)) * hsic_value
            batch_losses.append(ind)
        layer_loss = torch.mean(torch.stack(batch_losses))
        ind_losses.append(layer_loss)
    mean_ind_loss = torch.mean(torch.stack(ind_losses))
    return mean_ind_loss

def calculate_naturalness_loss(Y, Xs, Xt, detector, is_generator=True):
    real_images = torch.cat([Xs, Xt], dim=0) 
    real_dict = {'image': real_images}
    real_pred_dict = detector(real_dict)
    real_probs = real_pred_dict['prob']        
    fake_dict = {'image': Y}
    fake_pred_dict = detector(fake_dict)
    fake_probs = fake_pred_dict['prob']        
    if is_generator:
        generator_loss = F.binary_cross_entropy(fake_probs, 
                                               torch.zeros_like(fake_probs))
        return generator_loss
    else:
        real_loss = F.binary_cross_entropy(real_probs, 
                                          torch.zeros_like(real_probs))
        fake_loss = F.binary_cross_entropy(fake_probs, 
                                          torch.ones_like(fake_probs))
        discriminator_loss = 0.4 * real_loss + 0.6 * fake_loss
        return discriminator_loss

def calculate_PI_loss(Y_attr, Xt_attr):
    norm_diffs = []
    for i in range(len(Y_attr)):
        Y_normalized = F.normalize(Y_attr[i], p=2, dim=1)
        Xt_normalized = F.normalize(Xt_attr[i], p=2, dim=1)
        
        norm_diff = torch.norm(Y_normalized - Xt_normalized, p=2)
        norm_diffs.append(norm_diff)

    pull_loss = torch.mean(torch.stack(norm_diffs))
    return pull_loss

def calculate_MD_loss(Xs_id_initial, Xt_id_initial, Y_id_initial):
    with torch.no_grad():
        Xs_id_initial_norm = F.normalize(Xs_id_initial, p=2, dim=1)
        Xt_id_initial_norm = F.normalize(Xt_id_initial, p=2, dim=1)
    Y_id_initial_norm = F.normalize(Y_id_initial, p=2, dim=1)
    cos_sim1 = F.cosine_similarity(Xt_id_initial_norm, Y_id_initial_norm, dim=1)
    cos_sim2 = F.cosine_similarity(Xt_id_initial_norm, Xs_id_initial_norm, dim=1)
    return torch.mean(torch.abs(cos_sim1 - cos_sim2))

def save_images(Xs, Xt, Y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(Xs.size(0)): 
        xs_i = Xs[i:i+1] 
        xt_i = Xt[i:i+1]
        y_i = Y[i:i+1]
        I = [xs_i, xt_i, y_i]
        image = make_image(I, 1)
        save_path = os.path.join(save_dir, f'generate.png')
        cv2.imwrite(save_path, image.transpose([1, 2, 0]),
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def transform_loss(loss):
    return torch.exp(loss) if isinstance(loss, torch.Tensor) else math.exp(loss)

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
    se = se.to(device)
    decoder = decoder.to(device)
    G = G.to(device)
    encoder = encoder.to(device)
    se = DDP(se, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    G = DDP(G, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    for param in encoder.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    se_optimizer = Adam([
        {'params': se.parameters()}
    ], lr=args.lr)

    """ load pre-calculated mean and std: """
    param_dict = []
    for i in range(N + 1):
        state = torch.load(f'./modules/weights128/readout_layer{i}.pth', map_location=device)
        n_samples = state['n_samples'].float()
        std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
        neuron_nonzero = state['neuron_nonzero'].float()
        active_neurons = (neuron_nonzero / n_samples) > 0.01
        param_dict.append([state['m'].to(device), std, active_neurons])
    training_phase = 1 
    
    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        
        pbar = tqdm(train_loader, 
                    desc=f'Epoch {epoch}/{args.epochs} - Phase {training_phase}',
                    leave=True,
                    disable=local_rank != 0) 
        if training_phase == 1:
            se.train()
            G.eval()
            decoder.eval()
            for param in G.parameters():
                param.requires_grad = False
            for param in decoder.parameters():
                param.requires_grad = False
            for param in se.parameters():
                param.requires_grad = True
        
        if training_phase == 1:
            se_optimizer.zero_grad()
        epoch_losses = {
            'Encoder_loss': 0.0,
            'FD_loss': 0.0,
            'IR_loss': 0.0,
            'pos_loss': 0.0
        }
        batch_count = 0
        
        for batch_idx, (Xs, Xt) in enumerate(pbar):
            torch.cuda.empty_cache() 
            """ inference: """
            Xs = Xs.to(device)
            Xt = Xt.to(device)
            '''(2) generate Y: '''
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
            Xs_feats, Xt_feats = [], []
            Xs_lambda, Xt_lambda = [], []
            Zs_id, Zt_id = [], []
            for i in range(N):
                R = encoder.features[i]  # [2*B, Cr, Hr, Wr]
                Z, lambda_, _ = getattr(se.module, f'iba_{i}')(
                    R, readout_feats,
                    m_r=param_dict[i][0], std_r=param_dict[i][1],
                    active_neurons=param_dict[i][2],
                )
                Zs_id.append(Z[:B])
                Zt_id.append(Z[B:])
                X_id_restrict += encoder.restrict_forward(Z, i)

                m_s = torch.mean(R[:B], dim=0)  # [C, H, W]
                std_s = torch.mean(R[:B], dim=0)
    
                m_t = torch.mean(R[B:], dim=0)  # [C, H, W]
                std_t = torch.mean(R[B:], dim=0)

                eps_s = torch.randn(size=R[:B].shape).to(R[:B].device) * std_s + m_s
                eps_t = torch.randn(size=R[B:].shape).to(R[B:].device) * std_t + m_t
                feat_t = R[B:] * (1. - lambda_[B:]) + lambda_[B:] * eps_s
                feat_s = R[:B] * (1. - lambda_[:B]) + lambda_[:B] * eps_s
                Xs_feats.append(feat_s)
                Xt_feats.append(feat_t)
                Xs_lambda.append(lambda_[:B])
                Xt_lambda.append(lambda_[B:])

            X_id_restrict /= float(N)
            Xs_id, Xt_id = X_id_restrict[:B], X_id_restrict[B:]
            Xs_id_initial, Xt_id_initial = X_id[:B], X_id[B:]
            Xt_feats[0] = Xt
            Xs_feats[0] = Xs
            Xt_attr, Xt_attr_lamb = decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True)
            Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
            Y = torch.clamp(Y, -1, 1)
            encoder.features = []
            if training_phase == 1:
                loss_FD_angle = calculate_fd_loss_angle_optimized(Zs_id, Xs_feats, N) + calculate_fd_loss_angle_optimized(Zt_id, Xt_feats, N)
                loss_FD_hsic = calculate_fd_loss_hsic_optimized(Zs_id, Xs_feats, N) + calculate_fd_loss_hsic_optimized(Zt_id, Xt_feats, N)
                loss_FD = loss_FD_angle + 0.5 * loss_FD_hsic
            else:
                loss_FD = torch.tensor(0.0, device=device)
            if training_phase == 1:
                loss_IR = 2 - torch.mean(F.cosine_similarity(F.normalize(Xs_id_initial, p=2, dim=1), F.normalize(Xs_id, p=2, dim=1), dim=1)) - \
                torch.mean(F.cosine_similarity(F.normalize(Xt_id_initial, p=2, dim=1), F.normalize(Xt_id, p=2, dim=1), dim=1))
            else:
                loss_IR = torch.tensor(0.0, device=device)
            Y_id = encoder(
                F.interpolate(torch.cat((Y, Y), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                                mode='bilinear', align_corners=True),
                cache_feats=True
            )
            
            loss_pos = -1 * F.cosine_similarity(F.normalize(Xs_id, p=2, dim=1), F.normalize(Y_id[:B], p=2, dim=1), dim=1).mean()

            loss_encoder = loss_FD + 0.5 * loss_IR

            if training_phase == 1:
                loss_encoder.backward()
                torch.nn.utils.clip_grad_norm_(se.parameters(), max_norm=args.max_grad_norm)
                se_optimizer.step()
                epoch_losses['Encoder_loss'] += loss_encoder.item()
                epoch_losses['FD_loss'] += loss_FD.item()
                epoch_losses['IR_loss'] += loss_IR.item()
                epoch_losses['pos_loss'] += loss_pos.item()
            
            batch_count += 1
            if batch_idx % 10 == 0:
                save_images(Xs, Xt, Y, save_dir=args.save_dir)

            if batch_idx % 100 == 0 and local_rank == 0 and writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Encoder_loss', loss_encoder.item(), global_step)
                writer.add_scalar('Loss/FD_loss', loss_FD.item(), global_step)
                writer.add_scalar('Loss/IR_loss', loss_IR.item(), global_step)
                writer.add_scalar('Loss/pos_loss', loss_pos.item(), global_step)
            if training_phase == 1:
                pbar.set_postfix({
                    'Phase': f'{training_phase}',
                    'Encoder_loss': f'{loss_encoder.item():.4f}',
                    'FD_loss': f'{loss_FD.item():.4f}',
                    'IR_loss': f'{loss_IR.item():.4f}',
                    'pos_loss': f'{loss_pos.item():.4f}'
                })
            if batch_idx % 2000 == 0:
                save_path = f'E://FaceShield/checkpoints_FSE/{epoch}_{batch_idx}_phase{training_phase}.pth'
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'training_phase': training_phase,
                    'se': se.state_dict(),
                    'se_optimizer': se_optimizer.state_dict(),
                    'G': G.state_dict(),
                    'decoder': decoder.state_dict()
                }, save_path)

    if local_rank == 0 and writer is not None:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lambda_per', type=float, default=1.0)
    p.add_argument('--lambda_adv', type=float, default=1.0)
    p.add_argument('--lambda_se', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=5)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=20)
    p.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm for clipping')
    p.add_argument('-save', '--save_dir', type=str, default='E://FaceShield/generateImages/FSE')
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
    checkpoint = torch.load("E://FaceShield/checkpoints_FSE/model.pth")
    se_model = SE(in_c, out_c_list, 'cuda', kernel_size=1)
    se_model = se_model.eval()
    se_model.load_state_dict(new_checkpoint['se'])
    G.load_state_dict(torch.load(os.path.join(root, pathG)), strict=False)
    decoder.load_state_dict(torch.load(os.path.join(root, pathE)), strict=False)
    inference(args.save_dir, se_model, decoder, G, encoder)

