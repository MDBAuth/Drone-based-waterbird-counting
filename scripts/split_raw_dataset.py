import os
import cv2
import json
import yaml

from tqdm import tqdm
from pathlib import Path
from random import seed, shuffle
from shutil import copyfile, move

from utils.Slicer import Slicer
from utils.utils import ensure_path

def _save_slices_old(slice_dict, input_dict, input_dir, output_dir, datatype):
    """
    This function takes a list of inputs, and for each input does the following:
        - Reads the file as an image
        - For each slice in slice_set, crops the image and saves the output
    """    
    # Generate and save slices for each file in input dictionary
    output_dir = ensure_path(output_dir)
    for filename, slice_set in tqdm(input_dict.items(), desc=f"Saving {datatype} slices"):
        img = cv2.imread(str(input_dir/filename))
        for slice_id in slice_set:
            l,t,r,b = slice_dict[slice_id]
            slice = img[t:b, l:r]
            slicename = (os.path.splitext(filename)[0] + f"_t{t:04d}_b{b:04d}_l{l:04d}_r{r:04d}.jpg")
            slicepath = str(Path(output_dir) / slicename)
            cv2.imwrite(slicepath, slice)

def _save_slices(params, slice_dict, input_dict, input_dir, output_dir, datatype):
    """
    This function takes a list of inputs, and for each input does the following:
        - Reads the file as an image
        - For each slice in slice_set, crops the image and saves the output
    """    
    # Generate and save slices for each file in input dictionary
    
    output_dir = ensure_path(output_dir)
    for filename, slice_set in tqdm(input_dict.items(), desc=f"Saving {datatype} slices"):
        img = cv2.imread(str(input_dir/filename))
        for slice_id in slice_set:
            l,t,r,b = slice_dict[slice_id]
            slice = img[t:b, l:r]
            slicename = (os.path.splitext(filename)[0] + f"_t{t:04d}_b{b:04d}_l{l:04d}_r{r:04d}.jpg")
            slicepath = str(Path(output_dir) / slicename)
            cv2.imwrite(slicepath, slice)
            if datatype == 'trainval':
                cv2.imwrite(str(Path(params['training']['train_data_dir'])/params['data']['project_name']/'sliced_images'/slicename), slice)
                
def split_dataset_old(params, project_dir, points_dict, trainval_dir, holdouts_dir, nolabels_dir):
    """
    This function is the main workhorse in reading the raw project folder
    and splitting it up into training, holdout, and nolabels sub-folders.
    """
    # Use the holdouts ratio defined in params.yaml to split the dataset
    holdouts_ratio = params['data']['holdouts_ratio']

    # Create the folders, if they do not already exist
    for path in [trainval_dir, holdouts_dir, nolabels_dir]:
        ensure_path(path)

    # Separate all files into empty and nonempty files, saving the former into the nolabels folder
    nonempty_filenames = []
    for filename, _ in tqdm(points_dict.items(), desc="Filtering out empty images"):
        points = points_dict[filename]
        if len(points) == 0:
            copyfile(project_dir/filename, nolabels_dir/filename)
        else:
            copyfile(project_dir/filename, trainval_dir/filename)
            nonempty_filenames.append(filename)

    # For all remaining (nonempty) files, split them up into training set and holdout set
    seed(1)
    shuffle(nonempty_filenames)
    nonempty_size = len(nonempty_filenames)
    holdouts_size = int(holdouts_ratio*nonempty_size)
    holdouts_set = nonempty_filenames[:holdouts_size]

    # This is where files are actually moved out of the trainval folder into the holdout folder
    for filename in tqdm(holdouts_set, desc="Creating holdouts set"):
        move(trainval_dir/filename, holdouts_dir/filename)

    return holdouts_set

def split_dataset(params, project_dir, points_dict, trainval_dir, holdouts_dir, nolabels_dir, *args):
    """
    This function is the main workhorse in reading the raw project folder
    and splitting it up into training, holdout, and nolabels sub-folders.
    """
    # Use the holdouts ratio defined in params.yaml to split the dataset
    holdouts_ratio = params['data']['holdouts_ratio']

    # Create the folders, if they do not already exist
    for path in [trainval_dir, holdouts_dir, nolabels_dir]:
        ensure_path(path)

    # Separate all images without annotation data
    noannotation_filenames = args[0]
    withannotation_filenames = {}
    for filename, _ in tqdm(points_dict.items(), desc="Filtering out images with missing annotation data"):
        if filename not in noannotation_filenames:
            withannotation_filenames[filename] = points_dict[filename]
        
    # Separate all files into empty and nonempty files, saving the former into the nolabels folder
    nonempty_filenames = []
    for filename, _ in tqdm(withannotation_filenames.items(), desc="Filtering out empty images"):
        points = withannotation_filenames[filename]
        if len(points) == 0:
            copyfile(project_dir/filename, nolabels_dir/filename)
        else:
            copyfile(project_dir/filename, trainval_dir/filename)
            nonempty_filenames.append(filename)

    # For all remaining (nonempty) files, split them up into training set and holdout set
    seed(1)
    shuffle(nonempty_filenames)
    nonempty_size = len(nonempty_filenames)
    holdouts_size = int(holdouts_ratio*nonempty_size)
    holdouts_set = nonempty_filenames[:holdouts_size]

    # This is where files are actually moved out of the trainval folder into the holdout folder
    for filename in tqdm(holdouts_set, desc="Creating holdouts set"):
        move(trainval_dir/filename, holdouts_dir/filename)

    return holdouts_set, withannotation_filenames

def generate_training_set_old(params, points_dict, holdouts_set, 
                          trainval_input_dir, holdouts_input_dir,
                          trainval_output_dir, holdouts_output_dir):
    """
    Once the dataset has been separated into trainval, holdout and nolabels directories,
    this function splits each image inside the trainval and holdout directories into pieces (slices).
    Saving those slices into respective folders is done in preparation for data labelling.
    """
    # Initialise image and slice dimensions
    image_height = params['data']['image_height']
    image_width = params['data']['image_width']
    slice_height = params['slices']['height']
    slice_width = params['slices']['width']

    sl = Slicer(h=image_height, w=image_width, dh=slice_height, dw=slice_width)
    slice_dict = sl.get_slice_dict()

    trainval_sliced_dict = {}
    holdouts_sliced_dict = {}
    for img_id, coords_dict in tqdm(points_dict.items(), desc="Creating slices dictionary"):
        if img_id not in holdouts_set:
            slice_list = []
            for _, pts in coords_dict.items():
                for pt in pts:
                    slice_id = sl.get_slice_id_from_point(pt)
                    slice_list.append(slice_id)
            slice_set = set(slice_list)
            if slice_set:
                trainval_sliced_dict[img_id] = slice_set
        else:
            slice_list = []
            slice_set = set(slice_dict.keys())
            holdouts_sliced_dict[img_id] = slice_set

    _save_slices(slice_dict, trainval_sliced_dict, trainval_input_dir, trainval_output_dir, datatype='trainval')
    _save_slices(slice_dict, holdouts_sliced_dict, holdouts_input_dir, holdouts_output_dir, datatype='holdouts')

def generate_training_set(params, points_dict, holdouts_set, 
                          trainval_input_dir, holdouts_input_dir,
                          trainval_output_dir, holdouts_output_dir):
    """
    Once the dataset has been separated into trainval, holdout and nolabels directories,
    this function splits each image inside the trainval and holdout directories into pieces (slices).
    Saving those slices into respective folders is done in preparation for data labelling.
    """
    
    # Initialise image and slice dimensions
    image_height = params['data']['image_height']
    image_width = params['data']['image_width']
    slice_height = params['slices']['height']
    slice_width = params['slices']['width']

    sl = Slicer(h=image_height, w=image_width, dh=slice_height, dw=slice_width)
    slice_dict = sl.get_slice_dict()
    
    trainval_sliced_dict = {}
    holdouts_sliced_dict = {}
    for img_id, coords_dict in tqdm(points_dict.items(), desc="Creating slices dictionary"):
        if img_id not in holdouts_set:
            slice_list = []
            for _, pts in coords_dict.items():
                for pt in pts:
                    slice_id = sl.get_slice_id_from_point(pt)
                    slice_list.append(slice_id)
            slice_set = set(slice_list)
            if slice_set:
                trainval_sliced_dict[img_id] = slice_set
        else:
            slice_list = []
            slice_set = set(slice_dict.keys())
            holdouts_sliced_dict[img_id] = slice_set

    _save_slices(params, slice_dict, trainval_sliced_dict, trainval_input_dir, trainval_output_dir, datatype='trainval')
    _save_slices(params, slice_dict, holdouts_sliced_dict, holdouts_input_dir, holdouts_output_dir, datatype='holdouts')

if __name__ == "__main__":
    with open('./params.yaml', 'r') as params_file:
        params = yaml.safe_load(params_file)

    raw_dir = Path(params['data']['raw_dir'])
    project_name = params['data']['project_name']
    points_path = raw_dir / project_name / params['data']['points_file']
    trainval_dir = Path(params['data']['trainval_dir'])/project_name
    holdouts_dir = Path(params['data']['holdouts_dir'])/project_name
    nolabels_dir = Path(params['data']['nolabels_dir'])/project_name
    trainval_output_dir = Path(params['slices']['trainval_dir'])/project_name/'sliced_images'
    holdouts_output_dir = Path(params['slices']['holdouts_dir'])/project_name/'sliced_images'

    with open(points_path) as file:
        data = file.read()
        data = json.loads(data)
        points_dict = data.get('points')
        
    holdouts_set = split_dataset(params, raw_dir / project_name,
                                 points_dict, trainval_dir,
                                 holdouts_dir, nolabels_dir)
    generate_training_set(params, points_dict, holdouts_set, 
                          trainval_dir, holdouts_dir,
                          trainval_output_dir, holdouts_output_dir)