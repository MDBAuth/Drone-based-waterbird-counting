import json
import yaml
import argparse
from pathlib import Path
from utils.utils import ensure_path, str2bool
from utils.Slicer import Slicer
from utils.count import count_folder

def parse_args(params):
    raw_dir = Path(params['data']['raw_dir'])
    project_name = params['data']['project_name']
    model_path = Path(params['models']['best_model'])
    conf_thresh = params['inference']['conf_thresh']
    results_dir = ensure_path(Path(params['inference']['results_dir']))
    with open(params['training']['train_coco'], 'r') as labels_file:
        categories = json.load(labels_file).get('categories')

    parser = argparse.ArgumentParser(description='Compare predicted vs true counts')
    parser.add_argument('--data-root', type=str,
                        default=raw_dir, help="Root of the data directory (default: ./data/raw/)")
    parser.add_argument('--project-folder', type=str,
                        default=project_name, help='Project (image) directory')
    parser.add_argument('--output-folder', type=str,
                        default=str(results_dir), help='Folder to output results')
    parser.add_argument('--conf-thresh', type=float,
                        default=conf_thresh, help='Binary (cut-off) confidence threshold (default: 0.95)')
    parser.add_argument('--overlap', type=float,
                        default=0.10, help='Overlap ratio from 0 to 1 (default: 0.1)')
    parser.add_argument('--sidelap', type=float,
                        default=0.10, help='Sidelap ratio from 0 to 1 (default: 0.1)')
    parser.add_argument('--stride', type=float,
                        default=0.5, help='Stride (or overlap) ratio between slices (top and bottom)')
    parser.add_argument('--model-path', type=str,
                        default=model_path, help='Path to model (default: ./models/detection/weights/best.pth)')
    parser.add_argument('--categories', type=list,
                        default=categories, help='Categories from labels.json')
    parser.add_argument('--save-slices', type=str2bool,
                        default=False, help='Save slices (default: False). Note: Only set to true when debugging')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    with open('params.yaml', 'r') as params_file:
        params = yaml.safe_load(params_file)
    
    # User inputs
    args = parse_args(params)

    # Initialise image and slice dimensions
    image_height = params['data']['image_height']
    image_width = params['data']['image_width']
    slice_height = params['slices']['height']
    slice_width = params['slices']['width']

    sl = Slicer(h=image_height, w=image_width, dh=slice_height, dw=slice_width, stride=args.stride)
    slice_dict = sl.get_slice_dict()
    
    count_folder(slice_dict, args)