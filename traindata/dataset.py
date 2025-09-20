import os
import random
from PIL import Image
from torch.utils.data import Dataset
import logging
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import time
import cv2

class FaceDataset(Dataset):
    def __init__(self, root, transform=None, is_train=True, cache_file=None, max_persons=1000, force_rebuild_cache=False):
        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.img_extensions = ('.jpg', '.jpeg', '.png')
        self.max_persons = max_persons
        start_time = time.time()
        cache_path = cache_file or os.path.join(root, f'dataset_index_{max_persons}persons.pkl')
        
        if os.path.exists(cache_path) and not force_rebuild_cache:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.person_dirs = cache_data['person_dirs']
                self.person_to_imgs = cache_data['person_to_imgs']
                self.all_imgs = cache_data['all_imgs']
                self.person_to_idx = cache_data['person_to_idx']
        else:
            if os.path.exists(cache_path) and force_rebuild_cache:
                print(f"Force rebuild cache: {cache_path}")
            else:
                print(f"First build dataset index (limit {max_persons} persons), this may take a few minutes...")
            start_time = time.time()
            root_path = Path(root)
            self.person_dirs = []
            self.person_to_imgs = {}
            self.all_imgs = []
            self.person_to_idx = {}
            person_paths = [p for p in root_path.iterdir() if p.is_dir()]
            person_paths = person_paths[:max_persons]
            
            for person_idx, person_path in enumerate(person_paths):
                person_imgs = [
                    str(img_path) 
                    for img_path in person_path.iterdir() 
                    if img_path.suffix.lower() in self.img_extensions
                ]
                
                if person_imgs:
                    person_name = person_path.name
                    self.person_dirs.append(person_name)
                    self.person_to_imgs[person_name] = person_imgs
                    self.all_imgs.extend(person_imgs)
                    self.person_to_idx[person_name] = person_idx
            
            cache_data = {
                'person_dirs': self.person_dirs,
                'person_to_imgs': self.person_to_imgs,
                'all_imgs': self.all_imgs,
                'person_to_idx': self.person_to_idx
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Index building time: {time.time() - start_time:.2f} seconds")
        
        self.num_persons = len(self.person_dirs)
        self.num_images = len(self.all_imgs)
        self.imgs_per_person = {
            person: len(imgs) for person, imgs in self.person_to_imgs.items()
        }

        
    def __getitem__(self, index):
        src_path = self.all_imgs[index]
        src_person = Path(src_path).parent.name
        while True:
            tgt_person = self.person_dirs[random.randint(0, self.num_persons - 1)]
            if tgt_person != src_person:
                break
        tgt_path = self.person_to_imgs[tgt_person][
            random.randint(0, self.imgs_per_person[tgt_person] - 1)
        ]
        try:
            src_img = cv2.imread(src_path)
            src_img = Image.fromarray(src_img)
            tgt_img = cv2.imread(tgt_path)
            tgt_img = Image.fromarray(tgt_img)
            
            if self.transform:
                src_img = self.transform(src_img)
                tgt_img = self.transform(tgt_img)
                
            return src_img, tgt_img
            
        except Exception as e:
            logging.error(f"Image loading error: {src_path} or {tgt_path}")
            logging.error(str(e))
            return self.__getitem__(random.randint(0, self.num_images - 1))
    
    def __len__(self):
        return self.num_images