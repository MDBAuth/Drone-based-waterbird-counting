import os
import cv2
import json
from pathlib import Path
from yaml import safe_load
from tqdm import tqdm
from random import sample, seed
from utils.Slicer import Slicer
from utils.utils import ensure_path
from utils.data_utils import CocoDataset

def save_slices(filepath, slice_ids, slice_dict, output_dir):
    img = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    slicepaths = []
    for slice_id in slice_ids:
        l,t,r,b = slice_dict[slice_id]
        slice = img[t:b, l:r]
        slicename = (os.path.splitext(filename)[0] + f"_t{t:04d}_b{b:04d}_l{l:04d}_r{r:04d}.jpg")
        slicepath = str(Path(output_dir) / slicename)
        cv2.imwrite(slicepath, slice)
        slicepaths.append(slicepath)
    return slicepaths

def cocofy_annotations(data_dir, labels):
    annotations = {}
    for label in labels:
        lbl_annotations = label['annotations'][0]
        if not lbl_annotations['was_cancelled']:
            if len(lbl_annotations['result'])==0 and lbl_annotations.get('prediction'):
                old_result = lbl_annotations['prediction']['result']
                lbl_annotations['result'] = old_result if len(old_result) > 3 else lbl_annotations['result']
            old_bbox_data = lbl_annotations['result']
            coco_bbox_data = []
            for bbox in old_bbox_data:
                old_bbox = bbox['value']
                coco_bbox = [old_bbox['x']/100, old_bbox['y']/100, old_bbox['width']/100, old_bbox['height']/100]
                coco_label = old_bbox['rectanglelabels'][0]
                coco_bbox_datum = {'bbox': coco_bbox, 'label': coco_label}
                coco_bbox_data.append(coco_bbox_datum)
            filename = os.path.basename(label['data']['image'])
            annotations[str(Path(data_dir) / filename)] = coco_bbox_data
    return annotations

def prepare_training_set(params):
    train_data_dir = ensure_path(params['training']['train_data_dir'])
    train_coco = params['training']['train_coco']
    project_name = params['data']['project_name']
    nolabels_dir = Path(params['data']['nolabels_dir'])/project_name
    slices_trainval_dir = Path(params['slices']['trainval_dir'])/project_name/'sliced_images'
    slices_nolabels_dir = Path(params['slices']['nolabels_dir'])/project_name/'sliced_images'

    image_height = params['data']['image_height']
    image_width = params['data']['image_width']
    slice_height = params['slices']['height']
    slice_width = params['slices']['width']

    labels = params['slices']['labels']

    with open(labels, 'r') as fr:
        annotations = cocofy_annotations(slices_trainval_dir, json.loads(fr.read()))

    coco_dataset = CocoDataset(train_data_dir)
    coco_dataset.add_existing_annotations(annotations)

    # Add negative samples
    sl = Slicer(h=image_height, w=image_width, dh=slice_height, dw=slice_width)
    slice_dict = sl.get_slice_dict()
    output_dir = ensure_path(slices_nolabels_dir)
    i = 0
    for filename in tqdm(os.listdir(nolabels_dir), desc="Adding negative samples"):
        i+=1
        seed(i)
        filepath = str(Path(nolabels_dir) / filename)
        slice_ids = sample(list(slice_dict), 3)
        slicepaths = save_slices(filepath, slice_ids, slice_dict, output_dir)
        coco_dataset.add_negative_samples(slicepaths)
    
    coco_dataset.export_dataset(train_coco)

if __name__ == "__main__":
    with open("./params.yaml", "r") as params_file:
        params = safe_load(params_file)

    prepare_training_set(params)