from mapUtils import *

if __name__ == "__main__":
    # Source directory to get the maps of gibson dataset
    map_dir = "data/maps/habitat_val_2400"
    args = get_args()
    full_map_size = args.map_size_cm//args.map_resolution
    #For creating maps ffrom the gibson dataset with different orientation
    create_map_habitat(map_dir)
    num_start_pos  = 1
    num_paths = 20
    mask_size = 1
    # Target directory to store the dataset 
    file_dir = 'data/maps/val'
    if not osp.isdir(file_dir):
        os.mkdir(file_dir)
    for i in range(len(os.listdir(map_dir))):
        map_file_name = osp.join(map_dir, f'map_{i}.png')
        # Reading the black and white image from the source directory
        map = io.imread(map_file_name, as_gray = True)
        for j in range(num_start_pos):
            start,ValidityCheckerObj = get_random_valid_pos(map)
            for k in range(mask_size): 
                if (k==0):
                    masked_img = map
                    # Generate paths on the full maps using RRT* only on the full map
                    paths = []
                    for p in range(num_paths):
                        path_param = {}
                        goal,ValidityCheckerObj = get_random_valid_pos(map)
                        path, path_interpolated, success = get_path(start, goal, ValidityCheckerObj)
                        path_param['path'] = path
                        path_param['path_interpolated'] = path_interpolated
                        path_param['success'] = success
                        paths.append(path_param)
                else:
                    h, w = map.shape[:2]
                    start_pos = (start[0]/dist_resl, args.map_size_cm//args.map_resolution - start[1]/dist_resl)
                    mask = create_circular_mask(h, w, center = start_pos , radius = (full_map_size/2)*(k)/mask_size)
                    masked_img = map.copy()
                    masked_img[~mask] = 1
                env_file_dir = osp.join(file_dir, f'env{i*num_start_pos*mask_size + j*mask_size + k:06d}')
                if not osp.isdir(env_file_dir):
                    os.mkdir(env_file_dir)
                file_name = osp.join(env_file_dir, f'map_{i*num_start_pos*mask_size + j*mask_size + k}.png')
                save_plot(masked_img,file_name)    
                
                for p in range(num_paths):
                    pickle.dump(paths[p], open(osp.join(env_file_dir,f'path_{p}.p'), 'wb'))
                    # fig = plt.figure()
                    # ax = fig.gca()
                    # implot = plt.imshow(masked_img)
                    # for path_ in paths[p]['path_interpolated']:
                    #     ax.plot(path_[0]/dist_resl, args.map_size_cm//args.map_resolution - path_[1]/dist_resl,'.-r', linewidth = 5)
                    # plt.axis('off')    
                    # fig.savefig(env_file_dir+'/'+f'_path_{p}.png')
                    # plt.close(fig)
                    
    # for i in range(len(os.listdir(map_dir))):
    #     map_file_name = osp.join(map_dir, f'map_{i}.png')
    #     # Reading the black and white image from the source directory
    #     map = cv.imread(map_file_name)
    #     # Detecting edge from the occupancy map created from gibson datasets 
    #     map = cv.Canny(map,100,200)
    #     # Generate paths on the full maps using RRT*
    #     paths = generate_path_RRTstar(0, num_paths, map)
    #     for j in range(num_paths):
    #         path_interpolated = paths[j]['path_interpolated']
    #         for k in range(mask_size + 1): 
    #             if (k==0):
    #                 masked_img = map
    #             else:
    #                 h, w = map.shape[:2]
    #                 #If there is not a valid path between start and goal position.
    #                 if path_interpolated.size==0:
    #                     start_pos = None
    #                 else:    
    #                     start_pos = (path_interpolated[0][0]/dist_resl, args.map_size_cm//args.map_resolution - path_interpolated[0][1]/dist_resl)
    #                 mask = create_circular_mask(h, w, center = start_pos , radius = (full_map_size/2)*(k+1)/mask_size)
    #                 masked_img = map.copy()
    #                 masked_img[~mask] = 1
    #             env_file_dir = osp.join(file_dir, f'env{i*num_paths*mask_size + j*mask_size + + k:06d}')
    #             if not osp.isdir(env_file_dir):
    #                 os.mkdir(env_file_dir)
    #             file_name = osp.join(env_file_dir, f'map_{i*num_paths*mask_size + j*mask_size + + k}.png')
    #             cv.imwrite(file_name, masked_img)
    #             pickle.dump(paths[j], open(osp.join(env_file_dir,f'path_0.p'), 'wb'))
                # fig = plt.figure()
                # ax = fig.gca()
                # #Map = io.imread(file_name)
                # implot = plt.imshow(map)
                # for path_ in path_interpolated:
                #     ax.plot(path_[0]/dist_resl, args.map_size_cm//args.map_resolution - path_[1]/dist_resl,'.-r', linewidth = 10)
                # plt.axis('off')    
                # fig.savefig(env_file_dir+'/'+f'_path_{0}.png')
                

        
    

                


