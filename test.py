import os
import cv2
import time
import argparse
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime, timedelta
import random
import math
import scipy.optimize as optimize

from utils import laplacian_blending, make_image
from modules.encoder128 import Backbone128, load_custom_state_dict
from modules.se import SE
from modules.aii_generator import AII512
from modules.decoder512 import UnetDecoder512
from preprocess.mtcnn import MTCNN


mtcnn = MTCNN()

TRANSFORMS_512 = transforms.Compose([
            transforms.Resize((512, 512), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def to_np(t: torch.Tensor):
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    return t.numpy()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def perturb_tensor(v: torch.Tensor, epsilon: float, delta_theta: float = math.pi):
    assert v.shape[0] == 1,
    d = v.shape[1]
    v = v / torch.norm(v, dim=1, keepdim=True)

    while True:
        z = torch.randn(1, d, device=v.device)
        u_raw = z - torch.sum(z * v, dim=1, keepdim=True) * v
        norm_u = torch.norm(u_raw, dim=1, keepdim=True)
        if norm_u.item() > 1e-6:
            break

    u = u_raw / norm_u
    U = torch.rand(1, device=v.device)
    CDF = lambda theta: (1 - math.exp(-epsilon * theta / delta_theta)) / (1 - math.exp(-epsilon * math.pi / delta_theta))
    theta = optimize.brentq(lambda x: float(CDF(x)) - U.item(), 0, math.pi)
    theta = torch.tensor(theta, device=v.device)
    v_perturbed = v * torch.cos(theta) + u * torch.sin(theta)
    return v_perturbed


def align_face(image_pil: Image.Image, crop_size: int = 512, reverse: bool = False):
    if reverse:
        out = mtcnn.align_multi(image_pil, min_face_size=64., thresholds=[0.6, 0.7, 0.7],
                                crop_size=(crop_size, crop_size), reverse=True)
        if out is None:
            mini = 20.
            th1, th2, th3 = 0.6, 0.6, 0.6
            while out is None and mini >= 5.:
                out = mtcnn.align_multi(image_pil, min_face_size=mini, thresholds=[th1, th2, th3],
                                        crop_size=(crop_size, crop_size), reverse=True)
                if out is None:
                    th1 *= 0.8
                    th2 *= 0.8
                    th3 *= 0.8
                    mini *= 0.8
        return out
    else:
        faces = mtcnn.align_multi(image_pil, min_face_size=64., thresholds=[0.6, 0.7, 0.8],
                                  factor=0.707, crop_size=(crop_size, crop_size))
        return faces


def inference(src_img_path: str,
              tar_img_path: str,
              save_dir: str,
              encoder: Backbone128,
              se: SE,
              decoder: UnetDecoder512,
              G: AII512,
              device: torch.device,
              N: int,
              args):
    os.makedirs(save_dir, exist_ok=True)
    test_date = str(datetime.strptime(time.strftime(
        "%a, %d %b %Y %H:%M:%S", time.localtime()), "%a, %d %b %Y %H:%M:%S") + timedelta(hours=12)).split(' ')[0]
    save_dir = os.path.join(save_dir, test_date)
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger('inference')
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    train_handler = logging.FileHandler(filename=os.path.join(save_dir, f'similarity_{test_date}.log'))
    train_formatter = logging.Formatter('%(message)s')
    train_handler.setFormatter(train_formatter)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == train_handler.baseFilename for h in logger.handlers):
        logger.addHandler(train_handler)
    xs_bgr = cv2.imread(src_img_path)
    Xs_pil = Image.fromarray(xs_bgr)
    face_s = align_face(Xs_pil, crop_size=512, reverse=False)
    if face_s is not None:
        Xs_face = face_s[0]
    else:
        Xs_face = Xs_pil
    Xs = TRANSFORMS_512(Xs_face).unsqueeze(0).to(device)
    xt_bgr = cv2.imread(tar_img_path)
    Xt_pil = Image.fromarray(xt_bgr)
    out = align_face(Xt_pil, crop_size=512, reverse=True)
    if out is None:
        prefix = os.path.splitext(os.path.basename(tar_img_path))[0]
        save_path = os.path.join(save_dir, f'{prefix}_gen.png')
        plt.imsave(save_path, cv2.cvtColor(xt_bgr.astype(np.uint8), cv2.COLOR_RGB2BGR))
        return
    faces, tfm_invs, boxes = out
    fi = 0
    if boxes is not None and len(boxes) > 0:
        ss = 0
        for j, box in enumerate(boxes):
            w = box[2] - box[0] + 1.0
            h = box[3] - box[1] + 1.0
            s = w * h
            if s > ss:
                ss = s
                fi = j
    Xt_face = faces[fi]
    tfm_inv = tfm_invs[fi]
    Xt = TRANSFORMS_512(Xt_face).unsqueeze(0).to(device)
    param_dict = []
    for i in range(N + 1):
        state = torch.load(f'./modules/weights128/readout_layer{i}.pth', map_location=device)
        n_samples = state['n_samples'].float()
        std = torch.sqrt(state['s'] / (n_samples - 1)).to(device)
        neuron_nonzero = state['neuron_nonzero'].float()
        active_neurons = (neuron_nonzero / n_samples) > 0.01
        param_dict.append([state['m'].to(device), std, active_neurons])

    with torch.no_grad():
        B = 1
        X_id = encoder(
            F.interpolate(torch.cat((Xs, Xt), dim=0)[:, :, 37:475, 37:475], size=[128, 128],
                          mode='bilinear', align_corners=True),
            cache_feats=True
        )

        min_std = torch.tensor(0.01, device=device)
        readout_feats = [(encoder.features[i] - param_dict[i][0]) / torch.max(param_dict[i][1], min_std)
                         for i in range(N + 1)]

        X_id_restrict = torch.zeros_like(X_id).to(device)
        Xt_feats, Xt_lambda = [], []
        for i in range(N):
            R = encoder.features[i]
            Z, lambda_, _ = getattr(se, f'iba_{i}')(
                R, readout_feats,
                m_r=param_dict[i][0], std_r=param_dict[i][1],
                active_neurons=param_dict[i][2],
            )
            X_id_restrict += encoder.restrict_forward(Z, i)

            Rs, Rt = R[:B], R[B:]
            lambda_t = lambda_[B:]
            m_s = torch.mean(Rs, dim=0)
            std_s = torch.mean(Rs, dim=0)
            eps_s = torch.randn(size=Rt.shape).to(Rt.device) * std_s + m_s
            feat_t = Rt * (1. - lambda_t) + lambda_t * eps_s

            Xt_feats.append(feat_t)
            Xt_lambda.append(lambda_t)

        X_id_restrict /= float(N)
        Xs_id = X_id_restrict[:B]
        Xs_id = perturb_tensor(Xs_id, args.epsilon)
        Xt_feats[0] = Xt
        Xt_attr, Xt_attr_lamb = decoder(Xt_feats, lambs=Xt_lambda, use_lambda=True, first_use_img=True)
        Y = G(Xs_id, Xt_attr, Xt_attr_lamb)
        Y = torch.clamp(Y, -1, 1)
        encoder.features = []
        Y_id_gt = encoder(
            F.interpolate(Y[:, :, 37:485, 37:485], size=[128, 128], mode='bilinear', align_corners=True),
            cache_feats=False
        )
        Xs_id_gt, Xt_id_gt = X_id[:B], X_id[B:]
        msg = ''
        msg += "cos<Xs, Xt>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Xt_id_gt, dim=1).mean().detach().cpu().numpy()
        msg += "cos<Y, Xt>=%.3f | " % torch.cosine_similarity(Xt_id_gt, Y_id_gt, dim=1).mean().detach().cpu().numpy()
        msg += "cos<Y, Xs>=%.3f | " % torch.cosine_similarity(Xs_id_gt, Y_id_gt, dim=1).mean().detach().cpu().numpy()
        logger.info(msg)
    prefix = os.path.splitext(os.path.basename(tar_img_path))[0]
    save_path_face = os.path.join(save_dir, f'{prefix}_gen_face.png')
    save_path_comp = os.path.join(save_dir, f'{prefix}_gen.png')

    img_Y = (Y[0].detach().cpu().numpy().transpose([1, 2, 0]) * 0.5 + 0.5) * 255
    img_Y = img_Y.astype(np.uint8)
    plt.imsave(save_path_face, cv2.cvtColor(img_Y.astype(np.uint8), cv2.COLOR_RGB2BGR))

    H, W, _ = xt_bgr.shape
    frame = cv2.warpAffine(img_Y.astype(np.float32), tfm_inv.astype(np.float32),
                           dsize=(int(W), int(H)), borderValue=0)
    m = np.zeros(img_Y.shape, img_Y.dtype)
    m[40:472, 80:455, :] = 1
    m = cv2.warpAffine(m, tfm_inv.astype(np.float32), dsize=(int(W), int(H)), borderValue=0)

    try:
        src = np.array([255., 255., 1.]).reshape(3, 1)
        x, y = np.matmul(tfm_inv, src)
        res_possion = cv2.seamlessClone(
            frame.astype(np.uint8),
            xt_bgr.astype(np.uint8),
            (m.astype(np.uint8) * 255),
            (int(x[0]), int(y[0])),
            cv2.NORMAL_CLONE
        )
        plt.imsave(save_path_comp, cv2.cvtColor(res_possion.astype(np.uint8), cv2.COLOR_RGB2BGR))
    except Exception:
        res = laplacian_blending(A=frame, B=xt_bgr, m=m)
        plt.imsave(save_path_comp, cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-img', '--image_path', type=str, default='E://FaceShield/Original/img1.jpg')
    p.add_argument('-e', '--epsilon', type=float, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-save', '--save_dir', type=str, default='E://FaceShield/Original/Perturbed')
    p.add_argument('--device', type=str, default='cuda:1')
    args = p.parse_args()

    if args.seed == -1:
        args.seed = random.randint(0, 1000000)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

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
    with torch.no_grad():
        inference(args.img, arg.img, args.save_dir, encoder, se_model, decoder, G, device, N)


