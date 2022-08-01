''' A dataloader for training Mask+Transformers
'''

import gzip
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os
from os import path as osp

from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
from utils.mpt_utils import geom2pix, geom2pix_mat_neg, geom2pix_mat_pos, get_encoder_input, get_grid_points

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
                #for i in range(len(os.listdir(osp.join(dataFolder, f'env{envNum}'))))
                for i in range(2000)
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
        with gzip.open(osp.join(self.dataFolder, f'env{env}', f'data{idx_sample:06d}.p'), 'rb') as f:
            data = pickle.load(f)

        explored_map = data['explored_map']
        collison_map = data['collision_map']
        map_size = collison_map.shape
        receptive_field = 32
        goal_index = (data['goal'][1], data['goal'][0])
        start_index = (data['curr_loc'][1], data['curr_loc'][0]) 
        path = data['path_to_go']
        #mapEncoder = get_encoder_input(explored_map, collison_map , goal_index, start_index, receptive_field)            
        mapEncoder = get_encoder_input(explored_map, 1 - collison_map , goal_index, start_index, receptive_field)  
        #start_pos = data['path_to_go'][0]
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
        
        #next_point_idx = geom2pix_mat_pos([next_point_to_go], map_size, receptive_field)[0][0]
        AnchorPointsPos = []
        #AnchorPointsNeg = []
        # AnchorPointsPos.append(next_point_idx)
        for pos in path:
            #index = geom2pix_mat_pos([geom2pix(pos, size = map_size)], map_size, receptive_field)[0][0]
            indices, = geom2pix_mat_pos([geom2pix(pos, size = map_size)], map_size, receptive_field)
            #neg_indices = geom2pix_mat_neg([geom2pix(pos, size = map_size)], map_size, receptive_field)
            for index in indices:
                if index not in AnchorPointsPos:
                     AnchorPointsPos.append(index)
            # if index not in AnchorPointsPos:
            #     AnchorPointsPos.append(index) 
            # for index in neg_indices:     
            #     if index not in AnchorPointsNeg:
            #         AnchorPointsNeg.append(index)              
        #AnchorPointsNeg = list(set(range(len(get_grid_points(map_size, receptive_field))))-set(AnchorPointsPos))
        backgroundPoints = list(set(range(len(get_grid_points(map_size, receptive_field))))-set(AnchorPointsPos))
        numBackgroundSamp = min(len(backgroundPoints), 2*len(AnchorPointsPos))
        AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()

        #AnchorPointsNeg = list(set(AnchorPointsNeg)-set(AnchorPointsPos))
        anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))


        #labels = torch.tensor(AnchorPointsPos)
        
        
        labels = torch.zeros_like(anchor)
        labels[:len(AnchorPointsPos)] = 1
        preprocess_rgb = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        preprocess_depth = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])                
        rgb = Image.fromarray(data['curr_rgb'])
        rgb = preprocess_rgb(rgb)
        depth = np.squeeze(data['curr_depth'], axis=2)
        depth = Image.fromarray((depth* 255).astype(np.uint8))
        depth = preprocess_depth(depth)
        return {
            'map':torch.as_tensor(mapEncoder),
            'rgb': torch.as_tensor(rgb),
            'depth': torch.as_tensor(depth), 
            'anchor':anchor, 
            'labels':labels
        }