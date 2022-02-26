''' A dataloader for training Mask+Transformers
'''

import torch
from torch.utils.data import Dataset

import skimage.io
import pickle
import numpy as np

import os
from os import path as osp
from einops import rearrange

from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
from utils.utils import geom2pix

def PaddedSequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :] for batch_i in batch if batch_i is not None])
    data['rgb'] = torch.cat([batch_i['rgb'][None, :] for batch_i in batch if batch_i is not None])
    data['depth'] = torch.cat([batch_i['depth'][None, :] for batch_i in batch if batch_i is not None])
    data['anchor'] = pad_sequence([batch_i['anchor'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['labels'] = pad_sequence([batch_i['labels'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['length'] = torch.tensor([batch_i['anchor'].shape[0] for batch_i in batch if batch_i is not None])
    return data

def get_grid_points(res=0.05, size=(240,240)):
    # Convert Anchor points to points on the axis.
    X = np.arange(0, size[0], 20) * res
    Y = (size[1] - np.arange(0, size[1], 20)) * res
    grid_2d = np.meshgrid(X, Y)
    grid_points = rearrange(grid_2d, 'c h w->(h w) c')
    return grid_points

def get_hash_table(res=0.05, size=(240,240)):
    hashTable = [(20*r, 20*c) for c in range(int(size[1]*res)) for r in range(int(size[0]*res))]
    return hashTable
    
def geom2pix_mat_pos(pos, res=0.05, size=(240,240), receptive_field=32):
    """
    Find the nearest index of the discrete map state.
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    indices = np.where(np.linalg.norm(get_grid_points()-pos, axis=1)<=receptive_field*res*0.7)
    return indices


def get_encoder_input(explored_map, collision_map , goal_pos, start_pos, receptive_field=32):
    '''
    Returns the input map appended with the goal, and start position encoded.
    :param InputMap: The grayscale map
    :param goal_pos: The goal pos of the robot on the costmap.
    :param start_pos: The start pos of the robot on the costmap.
    :returns np.array: The map concatentated with the encoded start and goal pose.
    '''
    map_size = explored_map.shape
    assert len(map_size) == 2, "This only works for 2D maps"
    
    context_map = np.zeros(map_size)
    goal_start_y = max(0, goal_pos[0]- receptive_field//2)
    goal_start_x = max(0, goal_pos[1]- receptive_field//2)
    goal_end_y = min( map_size[1], goal_pos[0]+ receptive_field//2)
    goal_end_x = min( map_size[0], goal_pos[1]+ receptive_field//2)
    context_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0
    # Mark start region
    start_start_y = max(0, start_pos[0]- receptive_field//2)
    start_start_x = max(0, start_pos[1]- receptive_field//2)
    start_end_y = min( map_size[1], start_pos[0]+ receptive_field//2)
    start_end_x = min( map_size[0], start_pos[1]+ receptive_field//2)
    context_map[start_start_x:start_end_x, start_start_y:start_end_y] = -1.0
    return torch.as_tensor(np.concatenate((explored_map[None, :], collision_map[None,:], context_map[None, :])))


class PathDataLoader(Dataset):
    '''Loads each path, and extracts the masked positive and negative regions
    '''

    def __init__(self, env_list, dataFolder):
        '''
        :param env_list: The list of map environments to collect data from.
        :param samples: The number of paths to use from each folder.
        :param dataFolder: The parent folder where the files are located.
            It should follow the following format:
                env1/path_0.p
                    ...
                env2/path_0.p
                    ...
                    ...
        '''
        assert isinstance(env_list, list), "Needs to be a list"
        self.num_env = len(env_list)
        self.env_list = env_list
        self.indexDict = [(envNum, int(i)) 
            for envNum in env_list 
                for i in range(len(os.listdir(osp.join(dataFolder, f'env{envNum}'))))
            ]
        self.dataFolder = dataFolder
    

    def __len__(self):
        return len(self.indexDict)
    
    def __getitem__(self, idx):
        '''
        Returns the sample at index idx.
        returns dict: A dictonary of the encoded map and target points.
        '''
        env, idx_sample = self.indexDict[idx]
        with open(osp.join(self.dataFolder, f'env{env}', f'data{idx_sample:06d}.p'), 'rb') as f:
            data = pickle.load(f)

        explored_map = data['explored_map']
        collison_map = data['collision_map']
        map_size = collison_map.shape
        path = [data['next_waypoint']] #data['path_to_go']
        goal_index = geom2pix(data['path_to_go'][-1, :], size = (240,240))
        start_index = (data['curr_loc'][1], data['curr_loc'][0]) 
        
        mapEncoder = get_encoder_input(explored_map, collison_map , goal_index, start_index)            

        AnchorPointsPos = []
        for pos in path:
            indices, = geom2pix_mat_pos(pos, size = map_size)
            #print (pos,indices)
            for index in indices:
                if index not in AnchorPointsPos:
                    AnchorPointsPos.append(index)

        backgroundPoints = list(set(range(len(get_hash_table())))-set(AnchorPointsPos))
        numBackgroundSamp = min(len(backgroundPoints), 2*len(AnchorPointsPos))
        AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
        anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
        labels = torch.zeros_like(anchor)
        labels[:len(AnchorPointsPos)] = 1
        preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        rgb = Image.fromarray(data['curr_rgb'])
        rgb = preprocess(rgb)
        depth = Image.fromarray(data['curr_depth'])
        depth = preprocess(depth)
        return {
            'map':torch.as_tensor(mapEncoder),
            'rgb': torch.as_tensor(rgb),
            'depth': torch.as_tensor(depth), 
            'anchor':anchor, 
            'labels':labels
        }