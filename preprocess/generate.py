import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from modules.encoder128 import Backbone128
from traindata.dataset import FaceDataset

def compute_layer_statistics(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Backbone128(50, 0.6, 'ir_se').eval().to(device)
    encoder.load_state_dict(torch.load('modules/model_128_ir_se50.pth', map_location=device))
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FaceDataset(
        root=args.data_root,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    N = 10 
    statistics = []
    for _ in range(N + 1):
        statistics.append({
            'm_sum': 0,  
            's_sum': 0, 
            'neuron_active': 0,
            'n_samples': 0 
        })
    
    with torch.no_grad():
        for batch in tqdm(dataloader):

            if isinstance(batch, (tuple, list)):
                imgs = batch[0]
                
            else:
                imgs = batch
                
                
            imgs = imgs.to(device)
            
            face_region = F.interpolate(imgs[:, :, 37:475, 37:475], 
                                      size=[128, 128],
                                      mode='bilinear', 
                                      align_corners=True)
            _ = encoder(face_region, cache_feats=True)
            
            for i in range(N + 1):
                feat_flat = encoder.features[i]
                batch_size = feat_flat.size(0)
                statistics[i]['m_sum'] += feat_flat.sum(0)
                statistics[i]['s_sum'] += (feat_flat ** 2).sum(0)
                statistics[i]['neuron_active'] += (feat_flat.abs() > 0.01).sum(0)
                statistics[i]['n_samples'] += batch_size

            
            encoder.features = []
    for i in range(N + 1):
        n_samples = statistics[i]['n_samples']
        m = statistics[i]['m_sum'] / n_samples  
        s = statistics[i]['s_sum'] 
        neuron_nonzero = statistics[i]['neuron_active'] 
        state = {
            'm': m.cpu(),
            's': s.cpu(), 
            'n_samples': torch.tensor(n_samples),
            'neuron_nonzero': neuron_nonzero.cpu() 
        }
        
        save_path = f'./modules/weights128/readout_layer{i}.pth'
        torch.save(state, save_path)
        print(f"Layer {i} statistical information has been saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,default=r'E:\\FaceShield\content',
                        help='The root directory path of the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    args = parser.parse_args()
    
    import os
    os.makedirs('./modules/weights128', exist_ok=True)
    
    compute_layer_statistics(args)