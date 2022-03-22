#!/usr/bin/python3
''' Common functions used in this library.
'''
import skimage.io
import skimage.morphology as skim

import io

import numpy as np

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    raise ImportError("Container does not have OMPL installed")
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def png_decoder(key, value):
    '''
    PNG decoder with gray images.
    :param key:
    :param value:
    '''
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return skimage.io.imread(io.BytesIO(value), as_gray=True)


def cls_decoder(key, value):
    '''
    Converts class represented as bytes to integers.
    :param key:
    :param value:
    :returns the decoded value
    '''
    if not key.endswith(".cls"):
        return None
    assert isinstance(value, bytes)
    return int(value)


def geom2pix(pos, res=0.05, size=(240, 240)):
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
    return (np.int(np.floor(pos[0]/res) -1), np.int(size[0]-1-np.floor(pos[1]/res)))


class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.
    '''
    def __init__(self, si, CurMap, MapMask=None, res=0.05, robot_radius=0.1):
        '''
        Intialize the class object, with the current map and mask generated
        from the transformer model.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        :param MapMask: Areas of the map to be masked.
        '''
        super().__init__(si)
        self.size = CurMap.shape
        # Dilate image for collision checking
        InvertMap = np.abs(1-CurMap)
        InvertMapDilate = skim.dilation(InvertMap, skim.disk(robot_radius/res))
        MapDilate = abs(1-InvertMapDilate)
        if MapMask is None:
            self.MaskMapDilate = MapDilate>0.5
        else:
            self.MaskMapDilate = np.logical_and(MapDilate, MapMask)
            
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        pix_dim = geom2pix(state, size=self.size)
        return self.MaskMapDilate[pix_dim[1], pix_dim[0]]

def get_patch(model, start_pos, goal_pos, explored_map, collision_map, rgb, depth):
    '''
    Return the patch map for the given start and goal position, and the network
    architecture.
    :param model:
    :param start: 
    :param goal:
    :param input_map:
    '''
    preprocess_rgb = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    preprocess_depth = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),])          
    rgb = Image.fromarray(rgb.astype(np.uint8))
    rgb = torch.as_tensor(preprocess_rgb(rgb))
    depth = np.squeeze(depth, axis=2)
    depth = Image.fromarray((depth* 255).astype(np.uint8))
    depth = torch.as_tensor(preprocess_depth(depth))    
    # Identitfy Anchor points
    encoder_input = get_encoder_input(explored_map, collision_map , goal_pos, start_pos)
    hashTable = get_hash_table()
    predVal = model(encoder_input[None,:].float().cuda(),rgb[None,:].float().cuda(), depth[None,:].float().cuda())
    predClass = predVal[0, :, :].max(1)[1]

    predProb = F.softmax(predVal[0, :, :], dim=1)
    possAnchor = [hashTable[i] for i, label in enumerate(predClass) if label==1 and i<144]
    receptive_field = 32
    # Generate Patch Maps
    patch_map = np.zeros_like(explored_map)
    map_size = explored_map.shape
    for pos in possAnchor:
        goal_start_x = max(0, pos[0]- receptive_field//2)
        goal_start_y = max(0, pos[1]- receptive_field//2)
        goal_end_x = min(map_size[1], pos[0]+ receptive_field//2)
        goal_end_y = min(map_size[0], pos[1]+ receptive_field//2)
        patch_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
    pred_map = predProb[0:144, 1].cpu().detach().numpy()      
    pred_map = pred_map.reshape((12, 12))
    return patch_map, pred_map           

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
