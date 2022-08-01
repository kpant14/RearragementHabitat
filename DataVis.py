import os
from os import path as osp
import re
import numpy as np
import pickle
from tqdm import tqdm
from utils.mpt_utils import geom2pix, geom2pix_mat_neg, get_patch, geom2pix_mat_pos, get_start_goal_map,get_grid_points
import torch
import torch.nn.functional as F
import json

from transformer import Models
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")
import time  
import matplotlib.pyplot as plt
import gzip

# Define the network
device='cuda' if torch.cuda.is_available() else 'cpu'

modelFolder = 'transformer_models/train_neg_anchor_0dropout'
epoch = 214

modelFile = osp.join(modelFolder, f'model_params.json')
model_param = json.load(open(modelFile))

transformer = Models.Transformer(**model_param)
_ = transformer.to(device)

checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
transformer.load_state_dict(checkpoint['state_dict'])

_ = transformer.eval()




# Plot the patches
env_num = 9 
stepNum_ = 500
plt.rcParams['figure.facecolor'] = 'white'
base_dir = 'tmp/data/val_mini/'
for stepNum in tqdm(range(stepNum_)):
    env_folder = base_dir + f'env{env_num}'
    with gzip.open(open(osp.join(env_folder, f'data{stepNum:06d}.p'), 'rb')) as f:
        data = pickle.load(f)

    explored_map = data['explored_map']
    collision_map = data['collision_map']
    gt_map = data['gt_map']
    rgb = data['curr_rgb']
    depth = data['curr_depth']
    path = data['path_to_go']
    goal_pos =  (data['goal'][1], data['goal'][0]) 
    start_pos = (data['curr_loc'][1], data['curr_loc'][0]) 
    gt_path = data['prm_star_path']['path_interpolated']
    true_patch_map = np.zeros_like(explored_map)
    map_size = explored_map.shape
    gt_goal_pos = geom2pix(gt_path[-1, :], map_size)
    gt_start_pos = geom2pix(gt_path[0, :], map_size)   
   
    receptive_field = 48
    # found = False
    # grid = get_grid_points(map_size, receptive_field)
    # for i in range(path.shape[0]):
    #     start_idx = geom2pix_mat_pos([start_pos], map_size, receptive_field)[0][0]
    #     next_point_idx = geom2pix_mat_pos([geom2pix(path[i], size = map_size)], map_size, receptive_field)[0][0]
    #     if (start_idx != next_point_idx):
    #         next_point_to_go = geom2pix(path[i] , size=map_size)
    #         found = True
    #         break
    # if not found:    
    #     next_point_to_go = geom2pix(path[-1,:] , size=map_size)

    # next_point_idx = geom2pix_mat_pos([next_point_to_go], map_size, receptive_field)[0][0]
    # path_pixel_pos = np.array(next_point_to_go)# Generate Patch Maps

    # goal_start_x = max(0, int(grid[next_point_idx][0])- receptive_field//2)
    # goal_start_y = max(0, int(grid[next_point_idx][1])- receptive_field//2)
    # goal_end_x = min(map_size[0], int(grid[next_point_idx][0])+ receptive_field//2)
    # goal_end_y = min(map_size[1], int(grid[next_point_idx][1])+ receptive_field//2)

    # #print(goal_start_x,goal_start_y,goal_end_x,goal_end_y)
    # true_patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    grid = get_grid_points(map_size, receptive_field)
    AnchorPointsPos = []
    AnchorPointsNeg = []
    path_pixel_pos =[]
    path_pixel_neg =[]
    for pos in path:
        index = geom2pix_mat_pos([geom2pix(pos, size = map_size)], map_size, receptive_field)[0][0]
        neg_indices = geom2pix_mat_neg([geom2pix(pos, size = map_size)], map_size, receptive_field)
        if index not in AnchorPointsPos:
            AnchorPointsPos.append(index)
            path_pixel_pos.append(grid[index])# Generate Patch Maps
        for index in neg_indices:     
            if index not in AnchorPointsNeg:
                AnchorPointsNeg.append(index)                   
    AnchorPointsNeg = list(set(AnchorPointsNeg)-set(AnchorPointsPos))
    for index in AnchorPointsNeg:     
        path_pixel_neg.append(grid[index])# Generate Patch Maps
    true_patch_map = np.zeros_like(explored_map)
    map_size = explored_map.shape
    path_pixel_pos = np.array(path_pixel_pos)
    path_pixel_neg = np.array(path_pixel_neg)
    
    for index in AnchorPointsPos:
        goal_start_x = max(0, int(grid[index][0]) - receptive_field//2)
        goal_start_y = max(0, int(grid[index][1]) - receptive_field//2)
        goal_end_x = min(map_size[0], int(grid[index][0]) + receptive_field//2)
        goal_end_y = min(map_size[1], int(grid[index][1]) + receptive_field//2)
        true_patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0

    fig, ax = plt.subplots(1, 4, figsize=(15,15), dpi=100)

    ax[0].set_title('Exploration Map')
    ax[0].imshow(explored_map, cmap='gray')
    ax[0].imshow(true_patch_map, cmap='gray', alpha=0.5)
    ax[0].plot(path_pixel_pos[:,0], path_pixel_pos[:,1], marker='o', linewidth=2)
    ax[0].scatter(path_pixel_neg[:,0], path_pixel_neg[:,1], marker='o', color='y', linewidth=2)
    ax[0].scatter(goal_pos[0], goal_pos[1], color='r', zorder=3)
    ax[0].scatter(start_pos[0], start_pos[1], color='g', zorder=3)
    
    ax[0].axis('off')
    
    agent_size = 8
    dx = 0
    dy = 0
    fc = 'Green'
    dx = np.cos(np.deg2rad(data['curr_loc'][2]))
    dy = np.sin(np.deg2rad(data['curr_loc'][2]))
    ax[0].arrow(start_pos[0] - 1 * dx, start_pos[1] - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)
    ax[0].invert_yaxis()
    #print(collision_map)
    
    ax[1].set_title('Obstacle Map')
    ax[1].imshow(1 - collision_map, cmap='gray')
    #ax[1].imshow(true_patch_map, cmap='gray', alpha=0.5)
    # ax[1].plot(path_pixel_pos[:,0], path_pixel_pos[:,1], marker='o', linewidth=2)
    # ax[1].scatter(path_pixel_neg[:,0], path_pixel_neg[:,1], marker='o', color='y', linewidth=2)
    ax[1].scatter(goal_pos[0], goal_pos[1], color='r', zorder=3)
    ax[1].scatter(start_pos[0], start_pos[1], color='g', zorder=3)
   
    ax[1].axis('off')
    ax[1].arrow(start_pos[0] - 1 * dx, start_pos[1] - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)
    ax[1].invert_yaxis()
    ax[2].set_title('StartGoal Map')
    ax[2].imshow(get_start_goal_map(collision_map, start_pos, goal_pos, receptive_field), cmap='gray')
    ax[2].plot(path_pixel_pos[:,0], path_pixel_pos[:,1], marker='o', linewidth=2)
    ax[2].scatter(path_pixel_neg[:,0], path_pixel_neg[:,1], marker='o', color='y', linewidth=2)
    ax[2].scatter(goal_pos[0], goal_pos[1], color='r', zorder=3)
    ax[2].scatter(start_pos[0], start_pos[1], color='g', zorder=3)
    ax[2].invert_yaxis()
    ax[2].axis('off')

    # rgb = data['curr_rgb']
    # im = ax[1][0].imshow(rgb)
    # ax[1][0].set_title('RGB')
    # ax[1][0].axis('off')

    # depth = data['curr_depth']
    # im = ax[1][1].imshow(depth, cmap='gray')
    # ax[1][1].set_title('Depth')
    # ax[1][1].axis('off')

    
    # ax[1][2].set_title('Ground Truth Map')
    # ax[1][2].imshow(gt_map, cmap='gray')
    # gt_path_pixel_pos = np.array([geom2pix(pos,size = map_size) for pos in gt_path])
    # ax[1][2].plot(gt_path_pixel_pos[:,0], gt_path_pixel_pos[:,1], marker='o', linewidth=2)
    # ax[1][2].scatter(gt_goal_pos[0], gt_goal_pos[1], color='r', zorder=3)
    # ax[1][2].scatter(gt_start_pos[0], gt_start_pos[1], color='g', zorder=3)
    # ax[1][2].invert_yaxis()
    # ax[1][2].axis('off')

    patch_map, pred_map = get_patch(transformer,start_pos, goal_pos, explored_map, collision_map, rgb, depth, receptive_field)
    # #patch_map = get_patch(transformer,start_pos, goal_pos, explored_map, collision_map, rgb, depth, receptive_field)
    # ax[1][3].set_title('Region Proposal Patch Map')
    # ax[1][3].imshow(explored_map, cmap='gray')
    # ax[1][3].imshow(patch_map, cmap='gray', alpha=0.5)
    # ax[1][3].scatter(goal_pos[0], goal_pos[1], color='r', zorder=3)
    # ax[1][3].scatter(start_pos[0], start_pos[1], color='g', zorder=3)
    # ax[1][3].invert_yaxis()
    # ax[1][3].axis('off')


    
    pred_im = ax[3].imshow(pred_map, cmap="YlOrRd")
    ax[3].set_title('Predictions Probability')
    ax[3].axis('off')
    #cbar = fig.colorbar(pred_im, ax=ax[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout() 
    out_dir = base_dir + f'out{env_num}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_dir+f'data{stepNum:06d}.png')
    plt.close()


import cv2
from natsort import natsorted
image_folder = out_dir
video_name = env_folder+'video.avi'

images = [img for img in natsorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width//2,height//2))
for image in images:
    image_ = cv2.imread(os.path.join(image_folder, image))
    image_ = cv2.resize(image_,(width//2,height//2),interpolation=cv2.INTER_AREA)
    video.write(image_)

cv2.destroyAllWindows()
video.release()