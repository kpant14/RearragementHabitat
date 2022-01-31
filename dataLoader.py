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

#from utils.utils import geom2pix

def PaddedSequence(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :] for batch_i in batch if batch_i is not None])
    data['anchor'] = pad_sequence([batch_i['anchor'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['labels'] = pad_sequence([batch_i['labels'] for batch_i in batch if batch_i is not None], batch_first=True)
    data['length'] = torch.tensor([batch_i['anchor'].shape[0] for batch_i in batch if batch_i is not None])
    return data


def PaddedSequenceUnet(batch):
    '''
    This should be passed to DataLoader class to collate batched samples with various length.
    :param batch: The batch to consolidate
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :] for batch_i in batch if batch_i is not None])
    data['mask'] = torch.cat([batch_i['mask'][None, :] for batch_i in batch if batch_i is not None])
    return data


def PaddedSequenceMPnet(batch):
    '''
    This should be passed to the dataLoader class to collate batched samples with various lengths
    :param batch: The batch to consolidate.
    '''
    data = {}
    data['map'] = torch.cat([batch_i['map'][None, :, :] for batch_i in batch])
    data['inputs'] = pad_sequence([batch_i['inputs'] for batch_i in batch], batch_first=True)
    data['targets'] = pad_sequence([batch_i['targets'] for batch_i in batch], batch_first=True)
    data['length'] = torch.tensor([batch_i['inputs'].size(0) for batch_i in batch])
    return data

#map_size = (480, 480)
receptive_field = 32
res = 0.05 # meter/pixels

# Convert Anchor points to points on the axis.
X = np.arange(4, 24*20+4, 20)*res
Y = 24-np.arange(4, 24*20+4, 20)*res

grid_2d = np.meshgrid(X, Y)
grid_points = rearrange(grid_2d, 'c h w->(h w) c')
hashTable = [(20*r+4, 20*c+4) for c in range(24) for r in range(24)]

def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    NOTE: The Pixel co-ordinates are represented as follows:
    (0,0)------ X ----------->|
    |                         |  
    |                         |  
    |                         |  
    |                         |  
    Y                         |
    |                         |
    |                         |  
    v                         |  
    ---------------------------  
    """
    return (np.int(np.floor(pos[0]/res)), np.int(size[0]-1-np.floor(pos[1]/res)))
    
def geom2pixMatpos(pos, res=0.05, size=(480, 480)):
    """
    Find the nearest index of the discrete map state.
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    indices = np.where(np.linalg.norm(grid_points-pos, axis=1)<=receptive_field*res*0.7)
    return indices

def geom2pixMatneg(pos, res=0.05, size=(480, 480), num=1):
    """
    Find the nearest index of the discrete map state.
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :param num: The number of random sample index to select.
    :returns (int, int): The associated pixel co-ordinates.
    """
    dist = np.linalg.norm(grid_points-pos, axis=1)
    indices, = np.where(dist>receptive_field*res*0.7)
    indices = np.random.choice(indices, size=num)
    return indices,

def get_encoder_input(InputMap, goal_pos, start_pos):
    '''
    Returns the input map appended with the goal, and start position encoded.
    :param InputMap: The grayscale map
    :param goal_pos: The goal pos of the robot on the costmap.
    :param start_pos: The start pos of the robot on the costmap.
    :returns np.array: The map concatentated with the encoded start and goal pose.
    '''
    map_size = InputMap.shape
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
    return torch.as_tensor(np.concatenate((InputMap[None, :], context_map[None, :])))


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
                for i in range(len(os.listdir(osp.join(dataFolder, f'env{envNum:06d}')))-1)
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
        mapEnvg = skimage.io.imread(osp.join(self.dataFolder, f'env{env:06d}', f'map_{env}.png'), as_gray=True)
        
        with open(osp.join(self.dataFolder, f'env{env:06d}', f'path_{idx_sample}.p'), 'rb') as f:
            data = pickle.load(f)

        if data['success']:
            path = data['path_interpolated']
            goal = data['path'][-1, :]
            # Mark goal region
            goal_index = geom2pix(goal)
            #goal_index = geom2pix(path[-1, :])
            start_index = geom2pix(path[0, :])
            mapEncoder = get_encoder_input(mapEnvg, goal_index, start_index)            

            AnchorPointsPos = []
            for pos in path:
                indices, = geom2pixMatpos(pos, size = mapEnvg.shape)
                for index in indices:
                    if index not in AnchorPointsPos:
                        AnchorPointsPos.append(index)

            backgroundPoints = list(set(range(len(hashTable)))-set(AnchorPointsPos))
            numBackgroundSamp = min(len(backgroundPoints), 2*len(AnchorPointsPos))
            AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
            anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
            labels = torch.zeros_like(anchor)
            labels[:len(AnchorPointsPos)] = 1
            return {
                'map':torch.as_tensor(mapEncoder), 
                'anchor':anchor, 
                'labels':labels
            }