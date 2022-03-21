import os
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils.train_utils import get_transform
from utils.utils import ensure_path

def within_limits(bbox, limits, slice_bbox):
    slice_l, slice_t,_,_ = slice_bbox
    # if centre of object is outside limits, discount object
    l,t,r,b = bbox
    xc, yc = ((r+l)/2, (t+b)/2)

    # Check slice limits
    if not (limits['slice_left'] < xc < limits['slice_right'] and \
            limits['slice_top'] < yc < limits['slice_bottom']):
        return False

    # Check image limits
    xc += slice_l
    yc += slice_t
    lim_l = limits['image_left']
    lim_t = limits['image_top']
    lim_r = limits['image_right']
    lim_b = limits['image_bottom']

    if not (lim_l < xc < lim_r and lim_t < yc < lim_b):
        return False
    else:
        return True

def whole_image_count(filepath, model, slice_dict, args):
    filename = os.path.basename(filepath)

    if args.save_slices:
        CLASSES = {}
        for category in args.categories:
            CLASSES[category['id']] = category['name']
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    transform = get_transform()
    image = Image.open(filepath)
    width, height = image.size
    sidelap = width*args.sidelap/2
    overlap = height*args.overlap/2
    limits = {
        'image_left': sidelap,
        'image_right': width-sidelap,
        'image_top': overlap,
        'image_bottom': height-overlap
    }
    count = 0
    for _, slice_bbox in tqdm(slice_dict.items(), desc=f"Scanning slices in {filename}", leave=False):
        slice = image.crop(slice_bbox)
        orig_slice = np.array(slice)[:,:,::-1].copy()
        slice_width, slice_height = slice.size
        slice_sidelap = slice_width*args.stride/2
        slice_overlap = slice_height*args.stride/2
        limits.update({
            'slice_left': slice_sidelap,
            'slice_right': slice_width-slice_sidelap,
            'slice_top': slice_overlap,
            'slice_bottom': slice_height-slice_overlap
        })

        slice = transform(slice).unsqueeze(0).cuda() if torch.cuda.is_available() else transform(slice).unsqueeze(0)
        
        detections = model(slice)[0]
        
        scores = detections['scores'].cpu().detach().numpy()
        if args.save_slices:
            for i in range(len(scores)):
                if scores[i] > args.conf_thresh:
                    idx = int(detections["labels"][i])-1 # Subtract b/c background is class 0 in training but not in labels
                    box = detections["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")

                    label = f"{CLASSES[idx]}: {scores[i]*100:.2f}%"

                    cv2.rectangle(orig_slice, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(orig_slice, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if len(scores) > 0: # Avoid saving slices with 0 predictions
                l,t,_,_ = slice_bbox
                output_folder = ensure_path(Path(args.output_folder)/(args.project_folder + '_detections'))
                output_path = output_folder/f'{filename}_l-{l:04d}_t-{t:04d}.jpg'
                cv2.imwrite(str(output_path), orig_slice)

        n = len(scores[scores>args.conf_thresh])
        
        # Exclude low-confidence boxes
        boxes = detections['boxes'].cpu().detach().numpy()
        boxes = boxes[:n]
        # Exclude boxes in overlap/sidelap regions
        boxes = [box for box in boxes if within_limits(box, limits, slice_bbox)]
        count+=len(boxes)
    # print(f"{filename}: {count} detections.")
    return count

def count_folder(slice_dict, args):
    if not torch.cuda.is_available():
        model = torch.load(args.model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(args.model_path, map_location=torch.device('cuda'))
        model.cuda()
    model.eval()

    input_folder = Path(args.data_root) / args.project_folder
    files = os.listdir(input_folder)
    files = sorted([file for file in files if file.endswith('.JPG')])
    
    csv_file = Path(args.output_folder) / f"{args.project_folder}_detections.csv"
    with open(csv_file, 'w') as csvfile:
        csv_columns = ['FILENAME','COUNT']
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for file in tqdm(files, desc=f"Counting birds in {args.project_folder}"):
            count = {}
            whole_image_path = f'{input_folder}/{file}'
            count["FILENAME"] = file
            count["COUNT"] = whole_image_count(whole_image_path, model, slice_dict, args)
            writer.writerow(count)