import copy
from random import random
from cv2 import cuda_Event
from env import habitat
from env.habitat.utils.supervision import HabitatMaps 
from env.habitat import exploration_env

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")
from utils.utils import geom2pix, ValidityChecker
import os
import torch

import numpy as np
from torch.nn import functional as F
from habitat import Env
from habitat.config.default import get_config
from utils.model import get_grid
from arguments import get_args
from env.rearrange_dataset import RearrangementDatasetV0
import quaternion
import numpy as np
import sys
import skimage.morphology as skim
from skimage import io
from skimage import color
import pickle
import re
from os import path as osp
import glob
import cv2 as cv
from shutil import copyfile
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy import ndimage
# All measurements are mentioned in meters
# Define global parameters
length = 30 # Size of the map
robot_radius = 0.1
dist_resl = 0.05


def create_prm_planner(ValidityCheckerObj):
    mapSize = ValidityCheckerObj.size
    # Define the space
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(0, mapSize[1]*dist_resl) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl) # Set height bounds (y)
    space.setBounds(bounds)
    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityCheckerObj)
    # Create a simple setup
    ss = og.SimpleSetup(si)
    # # Use RRT*
    #planner = og.RRTstar(si)
    planner = og.PRMstar(si)
    planner.clear()
    return ss, planner



def get_path(start, goal, ss, planner, env = 0, plan_time=50):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    :param ValidityCheckerObj: An object of class ompl.base.StateValidityChecker
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    success = False

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 0.1)
    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time = 4
    solved = ss.solve(time)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(2.0)
        time +=3
        if time>plan_time:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        # Get the number of interpolation points
        num_points = int(4*ss.getSolutionPath().length())#//(dist_resl*32))
        ss.getSolutionPath().interpolate(num_points)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []
    print(path, env)
    return np.array(path), np.array(path_interpolated), success
    
def check_validity(CurMap, state):
    mapSize = CurMap.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    # Set bounds away from  boundary to avoid sampling points outside the map
    bounds.setLow(2.0)
    bounds.setHigh(0, mapSize[1]*dist_resl-2) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl-2) # Set height bounds (y)
    space.setBounds(bounds)
    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
    si.setStateValidityChecker(ValidityCheckerObj)
    return ValidityCheckerObj.isValid(state)

def get_nearest_valid(CurMap, state, N=10000):
    min_index = min(state[0] - 2 , state[1] - 2)
    min_ =   min_index if min_index > 0 else 0
    max_index = max(state[0] + 2 , state[1] + 2)
    max_ =   max_index if max_index < 12  else 12
    print(min_, max_)
    random_state = np.float64(np.random.uniform(min_,max_,(N,2)))
    min_dist = 10000
    min_state = [state[0], state[1]]
    curr_state = [state[0], state[1]]
    for i in range(N):
        if(check_validity(CurMap, random_state[i])):
            if(np.linalg.norm(curr_state - random_state[i]) < min_dist):
                min_dist = np.linalg.norm(curr_state - random_state[i])
                min_state = random_state[i]    
    return min_state            

def get_validity_checker(CurMap, robot_radius=0.1):
    mapSize = CurMap.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    # Set bounds away from  boundary to avoid sampling points outside the map
    bounds.setLow(2.0)
    bounds.setHigh(0, mapSize[1]*dist_resl-2) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl-2) # Set height bounds (y)
    space.setBounds(bounds)
    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap,robot_radius=robot_radius)
    si.setStateValidityChecker(ValidityCheckerObj)
    return space, ValidityCheckerObj

def get_random_valid_pos(CurMap,robot_radius=0.1):
    space,ValidityCheckerObj = get_validity_checker(CurMap,robot_radius=robot_radius) 
    # Define the valid location
    valid = ob.State(space)
    valid.random()
    while not ValidityCheckerObj.isValid(valid()):
        valid.random()
    return valid, ValidityCheckerObj

def generate_path_RRTstar(start, samples, CurMap):
    '''
    Run the experiment for random start and goal points on the real world environment
    :param start: The start index of the samples
    :param samples: The number of samples to collect
    '''
    mapSize = CurMap.shape
    # Planning parametersf
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    # Set bounds away from  boundary to avoid sampling points outside the map
    bounds.setLow(2.0)
    bounds.setHigh(0, mapSize[1]*dist_resl-2) # Set width bounds (x)
    bounds.setHigh(1, mapSize[0]*dist_resl-2) # Set height bounds (y)
    space.setBounds(bounds)
    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    # Validity checking
    ValidityCheckerObj = ValidityChecker(si, CurMap=CurMap)
    si.setStateValidityChecker(ValidityCheckerObj)
    paths=[]
    for i in range(start, start+samples):
        path_param = {}
        # Define the start and goal location
        start = ob.State(space)
        start.random()
        while not ValidityCheckerObj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityCheckerObj.isValid(goal()):   
            goal.random()

        path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success
        paths.append(path_param)
    return paths

def get_sim_location(env):
    agent_state = env.sim.get_agent_state(0)
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def get_gt_map(config, full_map_size):
    
    env = Env(config)
    # Get map in habitat simulator coordinates
    map_obj = HabitatMaps(env)
    
    agent_y = env.sim.get_agent_state().position.tolist()[1]*100.
    sim_map = map_obj.get_map(agent_y, -100., 100.0)

    sim_map[sim_map > 0] = 1.

    # Transform the map to align with the agent
    min_x, min_y = map_obj.origin/100.0
    x, y, o = get_sim_location(env)
    x, y = -x - min_x, -y - min_y
    range_x, range_y = map_obj.max/100. - map_obj.origin/100.

    map_size = sim_map.shape
    scale = 2.
    grid_size = int(scale*max(map_size))
    grid_map = np.zeros((grid_size, grid_size))

    grid_map[(grid_size - map_size[0])//2:
                (grid_size - map_size[0])//2 + map_size[0],
                (grid_size - map_size[1])//2:
                (grid_size - map_size[1])//2 + map_size[1]] = sim_map

    if map_size[0] > map_size[1]:
        st = torch.tensor([[
                (x - range_x/2.) * 2. / (range_x * scale) \
                            * map_size[1] * 1. / map_size[0],
                (y - range_y/2.) * 2. / (range_y * scale),
                180.0 + np.rad2deg(o)
            ]])

    else:
        st = torch.tensor([[
                (x - range_x/2.) * 2. / (range_x * scale),
                (y - range_y/2.) * 2. / (range_y * scale) \
                        * map_size[0] * 1. / map_size[1],
                180.0 + np.rad2deg(o)
            ]])

    rot_mat, trans_mat = get_grid(st, (1, 1,
        grid_size, grid_size), torch.device("cpu"))

    grid_map = torch.from_numpy(grid_map).float()
    grid_map = grid_map.unsqueeze(0).unsqueeze(0)
    translated = F.grid_sample(grid_map, trans_mat)
    rotated = F.grid_sample(translated, rot_mat)

    episode_map = torch.zeros((full_map_size, full_map_size)).float()
    if full_map_size > grid_size:
        episode_map[(full_map_size - grid_size)//2:
                    (full_map_size - grid_size)//2 + grid_size,
                    (full_map_size - grid_size)//2:
                    (full_map_size - grid_size)//2 + grid_size] = \
                            rotated[0,0]
    else:
        episode_map = rotated[0,0,
                            (grid_size - full_map_size)//2:
                            (grid_size - full_map_size)//2 + full_map_size,
                            (grid_size - full_map_size)//2:
                            (grid_size - full_map_size)//2 + full_map_size]



    episode_map = episode_map.numpy()
    episode_map[episode_map > 0] = 1.

    return episode_map

def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def save_plot(map, file_name):
    m, n = map.shape
    colored = np.ones((m, n, 3))
    current_palette = [(1,1,1)]
    colored = fill_color(colored, map, current_palette[0])        
    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    plt.imsave(file_name,colored)
    

def create_map_habitat(map_dir):
    args = get_args()
    full_map_size = args.map_size_cm//args.map_resolution
    config_env = get_config("configs/tasks/pointnav.yaml")
    config_env.defrost()
    config_env.DATASET.SPLIT = 'val'
    config_env.DATASET.DATA_PATH = (
    "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
    )
    config_env.DATASET.TYPE = "RearrangementDataset-v0"
    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 50
    config_env.SIMULATOR.TYPE = "RearrangementSim-v0"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
    config_env.SIMULATOR.GRAB_DISTANCE = 2.0
    config_env.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
    config_env.TASK.TYPE = "RearrangementTask-v0"
    config_env.TASK.SUCCESS_DISTANCE = 1.0
    config_env.TASK.SENSORS = [
        "GRIPPED_OBJECT_SENSOR",
        "OBJECT_POSITION",
        "OBJECT_GOAL",
    ]
    config_env.TASK.GOAL_SENSOR_UUID = "object_goal"
    config_env.TASK.MEASUREMENTS = [
        "OBJECT_TO_GOAL_DISTANCE",
        "AGENT_TO_OBJECT_DISTANCE",
    ]
    config_env.TASK.POSSIBLE_ACTIONS = ["STOP","MOVE_FORWARD","TURN_LEFT","TURN_RIGHT","GRAB_RELEASE"]
    dataset = RearrangementDatasetV0(config_env.DATASET)
    config_env.freeze()
    scenes  = RearrangementDatasetV0.get_scenes_to_load(config_env.DATASET)
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    
    count=0
    for i,scene in enumerate(scenes):
        config_env.defrost()
        config_env.DATASET.CONTENT_SCENES = scenes[i:i+1]
        print(config_env.DATASET.CONTENT_SCENES)
        print("Loading {}".format(config_env.SIMULATOR.SCENE))
        config_env.freeze()
        map = get_gt_map(config_env,full_map_size)
        file_name = osp.join(map_dir, f'map_{i}.png')
        save_plot(map, file_name)
        # for j in range(4):
        #     file_name = osp.join(map_dir, f'map_{count}.png')
        #     rotated_img = ndimage.rotate(map, j*90)
        #     save_plot(rotated_img, file_name)
        #     count+=1


                


