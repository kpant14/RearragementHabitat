import os
import torch
from env import habitat
from env.habitat.utils.supervision import HabitatMaps 
from env.habitat import exploration_env
import numpy as np
from torch.nn import functional as F
from habitat import Env
from habitat.config.default import get_config
from utils.model import get_grid
from arguments import get_args
from env.rearrange_dataset import RearrangementDatasetV0
import quaternion
import matplotlib.pyplot as plt

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

def save_plot(dir, map, map_name):
    m, n = map.shape
    colored = np.ones((m, n, 3))
    current_palette = [(1,1,1)]
    colored = fill_color(colored, map, current_palette[0])        
    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    #plt.imshow(colored)
    #plt.savefig(dir+map_name+".png")
    plt.imsave(dir+map_name+".png",colored)
    


if __name__ == "__main__":
    args = get_args()
    full_map_size = args.map_size_cm//args.map_resolution
    config_env = get_config("configs/tasks/pointnav.yaml")
    config_env.defrost()
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
    # rearrange_config_env = get_config("configs/tasks/pointnav.yaml")
    # rearrange_config_env.defrost()
    # rearrange_config_env.DATASET.DATA_PATH = (
    # "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
    # )
    # rearrange_config_env.freeze()
    scenes  = RearrangementDatasetV0.get_scenes_to_load(config_env.DATASET)
    map_dir = "data/maps/3000cm_100_/"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    for i,scene in enumerate(scenes):
        config_env.defrost()
        config_env.DATASET.CONTENT_SCENES = scenes[i:i+1]
        print(config_env.DATASET.CONTENT_SCENES)
        print("Loading {}".format(config_env.SIMULATOR.SCENE))
        config_env.freeze()
        map = get_gt_map(config_env,full_map_size)
        save_plot(map_dir,map,scene)

