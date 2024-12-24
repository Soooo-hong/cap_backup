from pathlib import Path
import numpy as np 
import random 
import torch 
import os 
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image

from datasets.data import pil_loader
from GaussianObject.scene.colmap_loader import (read_intrinsics_binary,read_extrinsics_binary,
                                                read_extrinsics_text,read_intrinsics_text)
from GaussianObject.scene.dataset_readers import readColmapCameras
from GaussianObject.utils.graphics_utils import fov2focal,getWorld2View


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

class OmniDataset(data.Dataset) : 
    def __init__(self, cfg, split)-> None : 
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_path = Path(self.cfg.dataset.data_path)
        self.object_names = [d.name for d in self.data_path.iterdir() if d.is_dir()]
        
        self.filenames = []
        
        for object_name in self.object_names : 
            fpath = os.path.join(self.cfg.dataset.split_path,object_name,f"{split}_sparse.txt")
            self.filenames.append([object_name,readlines(fpath)]) # (object_name, [index 번호들]) 로 구성됨 
        
        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        
        self.color_aug = self.cfg.dataset.color_aug
        
        
        
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, 
                                        self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales) # 1
        
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.interp = Image.LANCZOS
        self.loader = pil_loader  #RGB 이미지 load함수 
        self.to_tensor = T.ToTensor()
        self.resolution = cfg.dataset.resolution
        self.flip_left_right = cfg.dataset.flip_left_right


        if cfg.model.gaussian_rendering:
            frame_idxs = [0] + cfg.model.gauss_novel_frames
            if cfg.dataset.stereo:
                if split == "train":
                    stereo_frames = []
                    for frame_id in frame_idxs:
                        stereo_frames += [f"s{frame_id}"]
                    frame_idxs += stereo_frames
                else:
                    frame_idxs = [0, "s0"]
        else:
            # SfMLearner frames, eg. [0, -1, 1]
            frame_idxs = cfg.model.frame_ids.copy() #[0,-1,1]
        self.frame_idxs = frame_idxs

        self.is_train = split == "train"
        self.img_ext = '.png' if cfg.dataset.png else '.jpg'

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
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
        
        # multiple resolution support
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)
        self.resize_depth = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

        self._sequences = self._get_sequences(self.data_path)
        self.pose_path = cfg.dataset.pose_path
        self.depth_path = cfg.dataset.depth_path
        self.gt_depths = True if self.depth_path is not None else False
        self.gt_poses = True if self.pose_path is not None else False
        if self.pose_path is not None:
            self._poses,self.cam_infos = self._load_poses(self.cfg,self.pose_path, self._sequences)
                
    def __len__(self):
        length_data = 0 
        for idx in range(len(self.filenames)) : 
            length_data += len(self.filenames[idx][1])
        return length_data
    
    def get_color(self, folder, frame_index, do_flip):
        image_path = self.get_image_path(folder, frame_index)
        color = self.loader(image_path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def get_depth_anything(self, folder, frame_index,  do_flip):
        # f_str = f"{frame_index:010d}.npy"
        depth_path = os.path.join(
            self.data_path, folder, "zoe_depth_colored",f'{frame_index:05d}.png')
        depth_images = Image.open(depth_path)
        depth_gt = np.squeeze(np.array(depth_images))
        # depth_gt = np.squeeze(np.load(depth_path))

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def get_image_path(self, folder, frame_index) : 
        f_str = f"{frame_index:05d}{self.img_ext}"
        image_path = os.path.join(
            self.data_path, folder, "images", f_str)
        return image_path
    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                
                inputs[(n, im, i)] = self.to_tensor(f)
                if self.cfg.dataset.pad_border_aug != 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(self.pad_border_fn(color_aug(f)))
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
        

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        data_path = Path(data_path)
        for day in data_path.iterdir(): #day는 backpack_016부터 다른 객체가 들어감
            if not day.is_dir():
                continue
            image_dir = day/"images"
            day_sequences = [seq for seq in image_dir.iterdir() if seq.suffix =='.jpg']
            
            # lengths = [len(list((seq / "image_02" / "data").iterdir())) for seq in day_sequences]
            # day_sequences = [(day.name, seq.name, length) for seq, length in zip(day_sequences, lengths)]
            day_sequences = [(day.name, seq.name) for seq in day_sequences] 
            #day.name은 각 객체 이름 seq.name은 파일명  index로 설정 ('toy_plane_005', '00089.jpg')
            all_sequences.extend(day_sequences)

        return all_sequences
    
    @staticmethod
    def _load_poses(cfg, pose_path, sequences):
        
        day_list=[]
        
        for name in sequences : 
            if name[0] not in day_list : 
                day_list.append(name[0])
        poses = {}    
        cam_infos_dict = {}
        for day in day_list :

            try:
                cameras_extrinsic_file = os.path.join(pose_path,day, "sparse/0", "images.bin")
                cameras_intrinsic_file = os.path.join(pose_path,day, "sparse/0", "cameras.bin")
                cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            except:
                cameras_extrinsic_file = os.path.join(pose_path,day, "sparse/0", "images.txt")
                cameras_intrinsic_file = os.path.join(pose_path,day, "sparse/0", "cameras.txt")
                cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
                cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            
            cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(pose_path,day, 'images'), extra_opts= cfg.dataset)  
            cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) # cam_info 에 depth, mask, image이름, image, camera 정보 다 들어있음 
            # cam_infos = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                    #   image_path=image_path, image_name=image_name, 
                    #   width=width, height=height, mask=mask, mono_depth=mono_depth)
                    # cam_infops.mask  이런식으로 표현 가능 

            poses_seq  = []        
            try : 
                for idx in range(len(cam_infos)):                 
                    # T_w_cam0 = np.hstack((cam_infos[idx].R,cam_infos[idx].T.reshape((3,1))))
                    # T_w_cam0 = np.vstack((T_w_cam0,[0,0,0,1]))
                    T_w_cam0 =getWorld2View(torch.tensor(cam_infos[idx].R),torch.tensor(cam_infos[idx].T)).transpose(0, 1)
                    poses_seq.append(T_w_cam0)
            except FileNotFoundError:
                pass 
            
            poses_seq = np.array(poses_seq, dtype=np.float32)
            poses[(day, 'images')] = poses_seq
            cam_infos_dict[((day, 'images'))] = cam_infos # cam_infos에 이미 객체별로 나눠져 있음 
        return poses,cam_infos_dict    

    def __getitem__(self, index):
        inputs = {}
        cfg = self.cfg 
        
        do_color_aug = cfg.dataset.color_aug and self.is_train and random.random() > 0.5
        do_flip = cfg.dataset.flip_left_right and self.is_train and random.random() > 0.5
    
        # 각 txt파일의 index에 접근하도록하는 주소를 코드로 작성해야됨(frame_index에 할당)
         
        # line의 정확한 구성 요소를 알필요가 있음 
        # line = self.filenames[index]
        folder_index = index//len(self.filenames[0][1]) # 각 dir 접근하는 index
        line = self.filenames[folder_index]
        
        file_index = index%len(line[1])
        # filenames가 list안에 list형태로 각 list에는 각 객체(backpack_015 등)으로 구성되어야됨 
        # folder = line[index][0] # lines의 0번째 자리에는 파일 경로가 있어야되는 것으로 추정됨
        folder = line[0]
        # line은 list임  
        day, sequence = folder, "images"
        # backpack_016/images형태 띄도록 
        # frame_index = int(line[1])
        if line[1][file_index] == '':
            raise ValueError(f"Empty string found at line {line} and index {file_index}")

        frame_index = int(line[1][file_index]) ## 원래는 index가 들어감 
        frame_idxs = list(self.frame_idxs).copy()
        src_frame_index = frame_index
        for f_id in frame_idxs:
            i  = f_id
            if -1<frame_index+i<200 : 
                inputs[("color", f_id, -1)] = self.get_color(folder, frame_index + i, do_flip)
                frame_index = frame_index + i 
            else : 
                frame_index = (frame_index+i)%200
                # if (frame_index+i == -1 or frame_index+i==200 or frame_index+i ==201) : 
                inputs[("color",f_id,-1)] = self.get_color(folder, frame_index , do_flip)
                # inputs[('sparse',0)] = f"{folder}/{frame_index:05d}"
        
            for scale in range(self.num_scales): #scale = [0]
                cam_param = self.cam_infos[(day,sequence)][frame_index]
                fx = fov2focal(cam_param.FovX,cam_param.width)
                fy = fov2focal(cam_param.FovY,cam_param.height)
                cx = cam_param.width/2
                cy = cam_param.height/2
                K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.float32)
                # K = self.K
                # 마지막 16프레임만 예측하도록 만듦 
                
                 
                K_tgt = K.copy()
                if i == -1 :
                    cam_param_src = self.cam_infos[(day,sequence)][src_frame_index]
                    fx = fov2focal(cam_param_src.FovX,cam_param_src.width)
                    fy = fov2focal(cam_param_src.FovY,cam_param_src.height)
                    cx = cam_param_src.width/2
                    cy = cam_param_src.height/2
                    K_src = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]], dtype=np.float32)
                    inv_K_src = np.linalg.pinv(K_src)
                    inputs[("K_src", scale)] = torch.from_numpy(K_src)[..., :3, :3]
                    inputs[("inv_K_src", scale)] = torch.from_numpy(inv_K_src)[..., :3, :3]

                assert not do_flip
                
                inputs[("K_tgt", f_id)] = torch.from_numpy(K_tgt)[..., :3, :3]
            inputs[("frame_id",0)] = f"{folder}/{src_frame_index:05d}"

        if do_color_aug:
            # raise NotImplementedError
            color_aug = T.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
            
        self.preprocess(inputs, color_aug)
        
        # for idx in range(len(inputs.keys())):
        for i in frame_idxs:
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]
            # del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]
            
        if self.gt_depths:
            depth_gt = self.get_depth_anything(folder, src_frame_index, do_flip)
            depth_gt = np.expand_dims(depth_gt, 0)
            depth_gt = torch.from_numpy(depth_gt.astype(np.float32))
            depth_gt = self.resize_depth(depth_gt)
            inputs[("depth_gt", 0, 0)] = depth_gt   
            
        if self.gt_poses:
            # Load "GT" poses
            for f_id in frame_idxs:
                i = f_id
                id = (src_frame_index+i)%200
                # else : 
                #     id = frame_index
                pose = self._poses[(day, sequence)][id, :, :]
                inputs[("T_c2w", f_id)] = pose

        return inputs 
        