import os
import csv
import yaml
import torch
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from PIL import Image

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

if __name__ == "__main__":
    with open('./params.yaml', 'r') as params_file:
        params = yaml.safe_load(params_file)
    
    eval_mdba(params)
