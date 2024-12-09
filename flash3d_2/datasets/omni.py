import os
import random
import pickle
import gzip
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from typing import Optional
from pathlib import Path
from datasets.tardataset import TarDataset

from datasets.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from misc.depth import estimate_depth_scale_ransac
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir


def load_seq_data(data_path, split):
    file_path = data_path / f"{split}.pickle.gz"
    with gzip.open(file_path, "rb") as f:
        seq_data = pickle.load(f)
    return seq_data

class OmniDataset(data.Dataset) : 
    def __init__(self,
                 cfg,
                 split : Optional[str]=None,
                 )-> None : 
        super().__init__()
        
        self.cfg = cfg
        self.data_path = Path(self.cfg.dataset.data_path)
        
        if not self.data_path.is_absolute(): 
            code_dir = Path(__file__).parents[1]
            relative_path = self.data_path
            self.data_path = code_dir / relative_path
            if not self.data_path.exists():
                raise FileNotFoundError(f"Relative path {relative_path} does not exist")
        elif not self.data_path.exists():
            raise fileNotFoundError(f"Absolute path {self.data_path} does not exist")

        self.depth_path = None
        if self.cfg.dataset.preload_depths:
            assert cfg.dataset.depth_path is not None
            self.depth_path = Path(self.cfg.dataset.depth_path)

        self.split = split
        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        self.color_aug = self.cfg.dataset.color_aug
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales)
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.interp = Image.LANCZOS
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()

        self.is_train = self.split == "train"
        
        if self.is_train:
            self.split_name_for_loading = "train"
        else:
            self.split_name_for_loading = "test"

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
          
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)

        self.resize_depth = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

        
        