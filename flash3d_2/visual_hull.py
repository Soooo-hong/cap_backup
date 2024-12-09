import argparse
import math
import os
from argparse import Namespace

import camtools as ct
import numpy as np
import open3d as o3d
import torch
from tqdm import trange
from GaussianObject.scene.dataset_readers import sceneLoadTypeCallbacks
from GaussianObject.utils.camera_utils import cameraList_from_camInfos
from torch.nn import functional as F
import copy
from typing import NamedTuple
from torchvision import transforms



class SceneInfo(NamedTuple):
    Ks: list
    Ts: list
    images: list
    masks: list
    depths : list 

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def simple_resize_image(img, size):
    return transforms.Resize(size, antialias=True)(img)

def points2homopoints(points):
    assert points.shape[-1] == 3
    bottom = torch.ones_like(points[...,0:1])
    return torch.cat([points, bottom], dim=-1)

def batch_projection(Ks, Ts, points):
    '''
    Ks: B, 3, 3
    Ts: B, 4, 4
    points: B, N, 3
    '''
    pre_fix = points.shape[:-1] # [100, 100]
    points = points.reshape(-1, 3) # [M, 3]

    Ts = torch.stack(Ts, dim=0) # [N, 4, 4]
    Ks = torch.stack(Ks, dim=0).to(Ts.device) # [N, 3, 3]
    camera_num = Ks.shape[0]
    homopts = points2homopoints(points) # [M, 4]
    # world to camera # [N, M, 4] @ [N, 4, 4] = [N, M, 4]
    homopts_cam = torch.bmm(homopts.unsqueeze(0).repeat_interleave(Ts.shape[0], dim=0), Ts.transpose(1,2)) 
    # camera to image space  # [N, M, 4] @ [N, 4, 3] = [N, M, 3]
    homopts_img = torch.bmm(homopts_cam[...,:3], Ks.transpose(1,2))
    # normalize
    homopts_img = homopts_img / (homopts_img[...,2:] + 1e-6)
    # reshape back
    homopts_img = homopts_img.reshape(camera_num, *pre_fix, 3)
    homopts_cam = homopts_cam.reshape(camera_num, *pre_fix, 4)
    return homopts_img[...,0:2], homopts_cam[...,2]
def query_from_list_with_list(listA: list, listB: list):
    '''
    listA: [1, 2, 3]
    listB: [3, 2, 1]
    return: [2, 1, 0]
    '''
    return [listB[i] for i in listA]

def get_visual_hull(N, bbox, scene_info, cam_center):
    pcs = []
    color = []
    all_pts = []
    Ks = scene_info.Ks
    Ts = scene_info.Ts
    images = scene_info.images
    masks = scene_info.masks
    depths = scene_info.depths

    [xs, ys, zs], [xe, ye, ze] = bbox[0], bbox[1]

    # please note that in vasedeck, the images are not same size, for simplify, just resize them
    new_images = []
    new_masks = []
    new_depth = []
    img_size = images[0].shape[1:]
    for image, mask, depth in zip(images, masks,depths):
        new_images.append(simple_resize_image(image, img_size))
        new_masks.append(simple_resize_image(mask, img_size))
        new_depth.append(simple_resize_image(depth, img_size))
    images = torch.stack(new_images) # N C H W
    masks = torch.stack(new_masks) # N 1 H W
    depths = torch.stack(new_depth)

    for h_id in trange(N):
        # 2D grid 생성 
        i, j = torch.meshgrid(torch.linspace(xs, xe, N).cuda(),
                              torch.linspace(ys, ye, N).cuda())
        i, j = i.t(), j.t()
        #i,j. torch.ones_like(i)를 채널 기준으로 쌓음 ( N * N * 3 )
        pts = torch.stack([i, j, torch.ones_like(i).cuda()], -1) 
         
        #pts의 z좌표를 zs ~ ze로 균일하게 분포하도록 함, view frustrum에 균일한 위치에 속하게됨 
        pts[...,2] = h_id / N * (ze - zs) + zs # 200, 200, 3

        # shift the pts to be centered at the camera center (카메라 중심을 기준으로 하게끔 점들을 옮김)
        pts[...,0] += cam_center[0]  # note the order, [x, y, z], width, height, depth
        pts[...,1] += cam_center[1]
        pts[...,2] += cam_center[2]

        all_pts.append(pts)

        # now we have the pts, we need to project them to the image plane
        # batched projection
        #uv는 image plane에서의 x.y 좌표, z는  camera coord에서의 z의 좌표
        uv, z = batch_projection(Ks, Ts, pts) # [N, 200, 200, 2] -> x좌표와 y좌표 각각 존재, [N, 200, 200] 
        valid_z_mask = z > 0  #[200,200,200]
        valid_x_y_mask = (uv[...,0] > 0) & (uv[...,0] < cam_info.image_width) & (uv[...,1] > 0) & (uv[...,1] < cam_info.image_height)
        # [200,200,200]
        # [200,200,200]
        valid_pt_mask = valid_z_mask & valid_x_y_mask

        # simple resize the uv(x,y) to [-1, 1] for grid sampling 
        uv[...,0] = uv[...,0] / cam_info.image_width * 2 - 1
        uv[...,1] = uv[...,1] / cam_info.image_height * 2 - 1

        # now we have the uv, we use grid_sample to sample the image to get the color
        result = F.grid_sample(images.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 3
        # sample mask
        result_mask = F.grid_sample(masks.float(), uv, padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1) # N, 100, 100, 1
        # sample depth 
        depth_sampled = F.grid_sample(depths.float(), uv, padding_mode='zeros', align_corners=False).permute(0,2,3,1)  # 깊이 샘플링
        depth_sampled = depth_sampled.squeeze(-1)  # [N, 100, 100]

        valid_pt_mask = result_mask.squeeze() > 0 & valid_pt_mask #[200,200,200]

        valid_depth_mask = (depth_sampled > 0) & (depth_sampled < 1e3)
        
        valid_pt_mask = valid_pt_mask & valid_depth_mask

        pcs.append(valid_pt_mask.float().sum(0) >= (images.shape[0] - 1)) # [200, 200]
        color.append(result.mean(0)) # [200, 200, 3]
    
    pcs = torch.stack(pcs, -1)
    color = torch.stack(color, -1)

    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    idx = torch.where(pcs > 0)

    color = torch.stack((r[idx] * 255, g[idx] * 255, b[idx] * 255), -1)

    idx = torch.stack([idx[1], idx[0], idx[2]], -1) # note the order is hwz -> xyz
    # turn the idx to the point position used in batch_projection
    idx = idx.float() / N
    idx[...,0] = idx[...,0] * (xe - xs) + xs + cam_center[0]
    idx[...,1] = idx[...,1] * (ye - ys) + ys + cam_center[1]
    idx[...,2] = idx[...,2] * (ze - zs) + zs + cam_center[2]

    print("visual hull is Okay, with {} points".format(idx.shape[0]))
    # we get the point cloud, use open3d to visualize it
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(idx.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().numpy() / 255)
    # get bbox
    bbox = pcd.get_axis_aligned_bounding_box()
    return pcd, bbox


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate k views covering object')
    parser.add_argument('--data_dir', type=str, default='../capston2/dataset/omni3d/broccoli_003', help='data directory, we only support colmap type data, kitchen, garden')
    parser.add_argument("--cube_size", type=float, default=4.0, help="size of the cube in meters")
    parser.add_argument("--voxel_num", type=int, default=200, help="size of a voxel in meters")
    parser.add_argument('--sparse_id', type=int, default=-1, help='sparse id')
    parser.add_argument('--reso', type=int, default=1, help='the resolution of image, 1 for omni3d, 4 or 8 for mip360')
    parser.add_argument('--not_vis', action='store_true', help='whether vis the visual hull, is enable, not vis')
    parser.add_argument("--cube_size_shift_x", type=float, default=0.0, help="shift sizex of the cube in meters")
    parser.add_argument("--cube_size_shift_y", type=float, default=0.0, help="shift sizey of the cube in meters")
    parser.add_argument("--cube_size_shift_z", type=float, default=0.0, help="shift sizez of the cube in meters")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extra_opts = Namespace()
    extra_opts.sparse_view_num = -1
    extra_opts.resolution = args.reso
    extra_opts.use_mask = True
    extra_opts.use_depth = True 
    extra_opts.data_device = 'cuda'
    extra_opts.init_pcd_name = 'origin'
    extra_opts.white_background = False

    # load the camera parameters
    # we assume that the camera parameters are stored in the data_dir
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.data_dir, 'images', False, extra_opts=extra_opts) 
    camlist = cameraList_from_camInfos(scene_info.train_cameras, 1.0, extra_opts)

    # if sparse id is not zero, we only use given frames to construct the visual hull
    if args.sparse_id >= 0:
        selected_id = np.loadtxt(os.path.join(args.data_dir, f"sparse_{str(args.sparse_id)}.txt"), dtype=np.int32)
        print("the sparse id is {}, with {} frames".format(args.sparse_id, len(selected_id)))
        assert args.sparse_id == len(selected_id)
    else:
        selected_id = np.arange(len(camlist))

    # get all camera locations to recenter the scene
    cam_locations = []
    cam_rotations = []
    cam_T = []
    Ts = []
    Ks = []
    images = []
    masks = []
    depths = []
    
    for cam_info in camlist:
        cam_locations.append(cam_info.camera_center)
        cam_rotations.append(cam_info.R)
        cam_T.append(cam_info.T)
        Ts.append(cam_info.world_view_transform.T)
        fx = fov2focal(cam_info.FoVx, cam_info.image_width)
        fy = fov2focal(cam_info.FoVy, cam_info.image_height)
        Ks.append(torch.tensor([[fx, 0, cam_info.image_width/2], [0, fy, cam_info.image_height/2], [0, 0, 1]]))
        images.append(cam_info.original_image)
        masks.append(cam_info.mask)
        depths.append(cam_info.mono_depth)
        

    # in this time, we already have the camera parameters
    # first, we get the cemera locations center
    cam_center = torch.stack(cam_locations).mean(0)
    print('the camera center is:', cam_center)
 
    Ks = query_from_list_with_list(selected_id, Ks)
    Ts = query_from_list_with_list(selected_id, Ts)
    images = query_from_list_with_list(selected_id, images)
    masks = query_from_list_with_list(selected_id, masks)
    depths = query_from_list_with_list(selected_id,depths)
    
    scene_info = SceneInfo(Ks, Ts, images, masks,depths)
    Ks_clone = copy.deepcopy(Ks)

    bx = args.cube_size
    init_bbox = [[args.cube_size_shift_x-bx, args.cube_size_shift_y-bx, args.cube_size_shift_z-bx], 
                 [args.cube_size_shift_x+bx, args.cube_size_shift_y+bx, args.cube_size_shift_z+bx]]
    # we run the get_visual_hull twice, first to get the bound, second to get the visual hull
    pcd, bbox = get_visual_hull(args.voxel_num, init_bbox, scene_info, cam_center)
    
    # since we get the bound, we use this bound to better recon
    # we use the center of the bound as the center of the scene
    # please note that the bbox may need bigger, since the camera may not cover the whole scene
    bbox_min = bbox.get_min_bound()
    bbox_max = bbox.get_max_bound()
    # Calculate the center point of the original bounding box
    center = (bbox_min + bbox_max) / 2
    # Calculate the extents of the original bounding box
    extents = bbox_max - bbox_min
    # Calculate the scale factor to increase the size by 20% (1.2 times)
    scale_factor = 2
    # Calculate the scaled extents
    scaled_extents = extents * scale_factor
    # Calculate the new minimum and maximum points of the enlarged bounding box
    enlarged_bbox_min = center - scaled_extents / 2
    enlarged_bbox_max = center + scaled_extents / 2

    pcd, bbox_new = get_visual_hull(64, [enlarged_bbox_min, enlarged_bbox_max], scene_info, [0,0,0])
    # save the pointcloud
    if args.sparse_id >= 0:
        o3d.io.write_point_cloud(os.path.join(args.data_dir, f"visual_hull_{str(args.sparse_id)}.ply"), pcd)
    else:
        o3d.io.write_point_cloud(os.path.join(args.data_dir, "visual_hull_full.ply"), pcd)

    if not args.not_vis:
        Ts = np.array([i.cpu().numpy() for i in Ts])
        Ks = np.array(Ks_clone)
        cameras = ct.camera.create_camera_frustums(Ks, Ts, highlight_color_map={0: [1, 0, 0], -1: [0, 1, 0]})
        # build LineSet to represent the coordinate
        world_coord = o3d.geometry.LineSet()
        world_coord.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [2, 0, 0], 
                                                                [0, 0, 0], [0, 2, 0], 
                                                                [0, 0, 0], [0, 0, 2]]))
        world_coord.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 3], [0, 5]]))
        # X->red, Y->green, Z->blue
        world_coord.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        
        pcdo = o3d.io.read_point_cloud(os.path.join(args.data_dir, "sparse/0/points3D.ply"))
        
        