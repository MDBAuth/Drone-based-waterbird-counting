import sys, os
import csv
import yaml
import torch
import pylab
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

import codecs, json
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops.boxes import box_iou

from utils.Slicer import Slicer
from utils.utils import ensure_path
from utils.train_utils import get_transform

def eval_mdba(params):
    project_name = params['data']['project_name']
    slice_holdout_dir = Path(params['slices']['holdouts_dir'])/project_name/'sliced_images'
    results_dir = ensure_path(Path(params['evaluation']['results_dir']))
    conf_thresh = params['evaluation']['conf_thresh']

    model = torch.load(params['models']['best_model'])
    model.cuda()
    model.eval()

    transform = get_transform()
    predictions = {}
    filenames = sorted(os.listdir(slice_holdout_dir))
    img_ids = []
    first=True
    for filename in tqdm(filenames, desc="Counting birds in holdout set"):
        img_id = '_'.join(filename.split('_')[:3]) + '.JPG'
        if predictions.get(img_id) is None:
            predictions[img_id] = 0
            img_ids.append(img_id)
            if not first:
                print(f"Count for {prev_img_id} == {predictions[prev_img_id]}")
            first=False
        filepath = Path(slice_holdout_dir)/filename
        image = Image.open(filepath)
        image = transform(image).unsqueeze(0).cuda()
        out = model(image)
        scores = out[0]['scores'].cpu().detach().numpy()
        predictions[img_id] += len(scores[scores>conf_thresh])
        prev_img_id = img_id
    
    print(f"Count for {prev_img_id} == {predictions[prev_img_id]}")
    ground_truth = {}
    ground_truth_data = params['data']['ground_truth_counts']
    with open(ground_truth_data, "r") as csv_file:
        gt_data = csv.reader(csv_file)
        for row in gt_data:
            if row[0] in img_ids:
                ground_truth[row[0]] = int(row[1])

    xs = []
    ys = []
    for key, value in predictions.items():
        xs.append(ground_truth[key])
        ys.append(value)

    xs = np.array(xs)
    ys = np.array(ys)

    xs_fit = xs[:,np.newaxis]
    a, res, _, _ = np.linalg.lstsq(xs_fit, ys, rcond=None)
    r2 = 1 - res / (ys.size * ys.var())

    # Output CSV and PNG
    output_basename = "true_vs_predicted_counts"
    dataset = pd.DataFrame({'MANUAL': xs, 'AUTOMATIC': ys}, columns=['MANUAL', 'AUTOMATIC'])
    dataset.to_csv(f"{results_dir}/{output_basename}.csv")

    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 25})
    pylab.plot(xs,ys,'o')
    pylab.plot(xs,a*xs,"r--")
    plt.xlim(0.9,2200)
    plt.ylim(0.9,2200) 
    plt.ylabel('Predicted Count')
    plt.xlabel('True Count')
    plt.title('Comparing Ground Truth and Predicted Counts (Holdout Set)')
    plt.text(650, 750, f"y={a[0]:.3f}x," + " $\mathregular{R^{2}}$" + f"= {r2[0]:0.3f}", rotation = 27.5)
    plt.savefig(f"{results_dir}/{output_basename}.png")

def save_holdout_slices_with_inferences(parameters, slicename, boxes, Irec):
    if len(boxes) > 0:
        for i in range(len(boxes)):
            box = boxes[i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            Irec = cv2.rectangle(Irec, (startX, startY), (endX, endY), (0,0,255), 2)
        output_folder = ensure_path(Path(parameters['inference']['results_dir'])/(parameters['data']['project_name'] + '_detections'))
        output_path = output_folder/f'{slicename}.jpg'
        cv2.imwrite(str(output_path), Irec)
        # plt.imshow(Irec)
        
def save_inference_results(out_folder, predictions):
    pred_filepath = out_folder/'inference_results.json'
    preds = []
    for l in range(len(predictions)):
        preds.append(dict(filepath=predictions[l]['filepath'],
                          boxes=predictions[l]['boxes'].numpy().tolist(),
                          scores=predictions[l]['scores'].numpy().tolist(),
                          labels=predictions[l]['labels'].numpy().tolist()
                         )
                    )
    json.dump(preds, codecs.open(pred_filepath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
def groundTruth_annotations(params, data_dir, gt_annotation_pts, *args):
    """
    This function is used to reformat the annotations from label-studio into that required for training.

    Inputs:
        - data_dir: Data directory where original slice is located
        - labels: Label-studio annotations
        - args[0]: A list of excluded images

    Outpus:
        - annotations: Dictionary containing ground truth annotation
    """
    slice_height = params['slices']['height']
    slice_width = params['slices']['width']
    included_images = args[0]
    annotations = []
    for pts in tqdm(gt_annotation_pts, desc="Obtaining ground truth annotation for holdout set"):
        lbl_annotations = pts['annotations'][0]
        if not lbl_annotations['was_cancelled']:
            # This if statement is meant to deal with issues that may arise with label studio if saving over a project
            # if len(lbl_annotations['result'])==0 and lbl_annotations.get('prediction'):
            #     old_result = lbl_annotations['prediction']['result']
            #     lbl_annotations['result'] = old_result if len(old_result) > 3 else lbl_annotations['result']
            
            # Convert label studio annotations
            bbox_data = lbl_annotations['result']
            # coco_bbox_data = []
            boxes = []
            labels = []
            for bbox in bbox_data:
                orig_bbox = bbox['value']
                box = [orig_bbox['x']*slice_width/100, orig_bbox['y']*slice_height/100, (orig_bbox['x']+orig_bbox['width'])*slice_width/100,(orig_bbox['y']+orig_bbox['height'])*slice_height/100]
                label = 1 if orig_bbox['rectanglelabels'][0]=='bird' else 0
                # coco_bbox_datum = {'bbox': coco_bbox, 'label': coco_label}
                # coco_bbox_data.append(coco_bbox_datum)
                boxes.append(box)
                labels.append(label)
            filename = os.path.basename(pts['data']['image'])
            flag = [file for file in included_images if file.split('.JPG')[0] in filename]
            flag2 = filename in [file for file in included_images]
            # print(filename, flag, flag2)
            if flag2:
                annotations.append(dict(filepath=str(data_dir/filename),
                                        boxes=torch.tensor(boxes), 
                                        labels=torch.tensor(labels)
                                       )
                                  )
    return annotations

def draw_bboxes_save_results(parameters, filenames, groundTruth, holdout_dir):
    # sys.path.append(os.getcwd())
    model = torch.load(parameters['models']['best_model'])
    model.cuda()
    model.eval()    
    transform = get_transform()
    inferences = []
    groundtruths = []
    final_slices = []
    for filename in tqdm(filenames, desc="Drawing ground-truth and predicted boxes around birds in holdout set with available annotation data"):
        if not filename.startswith('.'):
            img_id = '_'.join(filename.split('_')[:3]) + '.JPG'
            raw_path = Path(parameters['data']['raw_dir'])/parameters['data']['project_name']/img_id
            regex = re.search("t(\d{4})..(\d{4})..(\d{4})..(\d{4})", filename)
            s_ids = regex.groups()
            t,b,l,r = tuple(int(s_id) for s_id in s_ids)
            raw_image = Image.open(raw_path);
            raw_slice = raw_image.crop((l,t,r,b)); 
            raw_slice_nparray = cv2.cvtColor(np.array(raw_slice)[:,:,::-1].copy(), cv2.COLOR_BGR2RGB)
            filepath = Path(holdout_dir)/filename
            image = Image.open(filepath)
            image = transform(image).unsqueeze(0).cuda() if torch.cuda.is_available() else transform(image).unsqueeze(0)
            out = model(image)[0]
            boxes = out['boxes'].cpu().detach().numpy()
            scores = out['scores'].cpu().detach().numpy()
            labels = out['labels'].cpu().detach().numpy()
            boxes = boxes[scores > parameters['inference']['conf_thresh']]
            labels = labels[scores > parameters['inference']['conf_thresh']]
            scores = scores[scores > parameters['inference']['conf_thresh']]

            inferences.append(dict(filepath=str(filepath),
                                    boxes=torch.tensor(boxes),
                                    scores=torch.tensor(scores),
                                    labels=torch.tensor(labels)
                               ))
            
            pred_boxes = out['boxes']; pred_scores = out['scores']
            gt_boxes = [groundTruth[i]['boxes'] for i in range(len(groundTruth)) if groundTruth[i]['filepath'] == str(filepath)][0]
            groundtruths.append(dict(filepath=str(filepath),
                                    boxes=gt_boxes,
                                    labels=torch.ones(len(gt_boxes))
                                ))
            # print(filepath, len(gt_boxes))
            Irec = raw_slice_nparray
            if len(gt_boxes) > 0:
                for j in range(len(gt_boxes)):
                    gt_box = gt_boxes[j].detach().cpu().numpy()
                    (startXg, startYg, endXg, endYg) = gt_box.astype("int")
                    Irec = cv2.rectangle(Irec, (startXg, startYg), (endXg, endYg), (0,255,0), 2)
            # plt.imshow(Irec)
            if len(pred_boxes) > 0:
                for i in range(len(pred_boxes)):
                    if pred_scores[i] > parameters['inference']['conf_thresh']:
                        box = pred_boxes[i].detach().cpu().numpy()
                        (startXp, startYp, endXp, endYp) = box.astype("int")
                        Irec = cv2.rectangle(Irec, (startXp, startYp), (endXp, endYp), (0,0,255), 2)
            # plt.imshow(Irec)
            slicename = filename
            output_folder = ensure_path(Path(parameters['inference']['results_dir'])/(parameters['data']['project_name'] + '_detections_available'))
            output_path = output_folder/f'{slicename}.jpg'
            cv2.imwrite(str(output_path), Irec)
            # plt.imshow(Irec)
            final_slices.append(Irec)
    results_save_folder = ensure_path(Path(parameters['inference']['results_dir'])/(parameters['data']['project_name'] + '_detections_available'))
    save_inference_results(results_save_folder, inferences)
    return groundtruths, inferences, final_slices

def chunker(image_list, chunk):
    for pos in range(0, len(image_list), chunk):
        yield image_list[pos : pos+chunk]

def plot_chunks(listofimages):
    nrows = 1
    ncols = len(listofimages)
    if ncols >= 5:
        fig_w, fig_h = 15, 15
    else:
        fig_w, fig_h = 5.7, 5.7
    f, axes = plt.subplots(nrows, ncols, figsize=(fig_w,fig_h))
    # print(len(axes))
    for i in range(1):
        for j in range(len(listofimages)):
            image = listofimages[j]
            axes[j].imshow(image)
            axes[j].set_xticks([])
            axes[j].set_yticks([])

def plot_holdout_images_with_bboxes(images, chunksize=5):
    plt.rcParams.update({'figure.max_open_warning': 0})
    for chunked_list in chunker(images, chunksize):
        plot_chunks(chunked_list)
        
if __name__ == "__main__":
    with open('./params.yaml', 'r') as params_file:
        params = yaml.safe_load(params_file)
    
    eval_mdba(params)
