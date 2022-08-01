from mapUtils import *
import skimage
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


if __name__ == "__main__":

    # Target directory to store the dataset 
    args = get_args()
    input_dir = 'data/maps/train'
    output_dir = 'data/maps/train_'   
    if not osp.isdir(output_dir):
        os.mkdir(output_dir) 
    for i in range(2800):
        env_file_dir = osp.join(output_dir, f'env{i:06d}')
        if not osp.isdir(env_file_dir):
            os.mkdir(env_file_dir)
        file_name = osp.join(env_file_dir, f'map_{i}.png')
        mapEnvg = skimage.io.imread(osp.join(input_dir, f'env{i:06d}', f'map_{i}.png'), as_gray=True)
        skimage.io.imsave(file_name,mapEnvg )
        for j in range(len(os.listdir(osp.join(input_dir, f'env{i:06d}')))-1):
            with open(osp.join(input_dir, f'env{i:06d}', f'path_{j}.p'), 'rb') as f:
                data = pickle.load(f)
            valid_path = []    
            if data['success']:
                paths = data['path_interpolated']
            for path in paths:
                if check_validity(mapEnvg,path):
                    valid_path.append(path)
                else:
                    break
            data['path_interpolated'] = np.array(valid_path)
            pickle.dump(data, open(osp.join(env_file_dir,f'path_{j}.p'), 'wb'))             