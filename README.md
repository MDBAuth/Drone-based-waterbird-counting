# Drone based waterbird counting 

This project is a proof-of-concept tool to automatically count colonially nesting waterbirds from drone imagery, developed initialy on Straw-necked Ibis at Narran Lakes.

The purpose of developing a tool to automatically count colonially nesting waterbirds is to allow accurate,
repeatable and timely monitoring of large colonies. It is envisioned that a user would be able to input drone
imagery and with minimal technical skill derive an accurate count of waterbirds.

## Installation  

This project contains 3 requirements files: 

|File|Type  |Contents  |
|--|--|--|
|requirements.txt|pip| minimal requirment to run the model and label datasets|
|Env.txt| Conda | the python 3.9 enviroment we are runing this project on
|Envrequirements.txt| pip | the packages we have installed on our enviroment|

We have been running this project on a NC4AS_T4_V3 in AzureML, to download this project and set up this enviroment run the following in a terminal

NOTE: Git lfs is used for the larger files https://git-lfs.github.com/

```bash
git clone https://github.com/MDBAuth/Drone-based-waterbird-counting.git

cd Drone-based-waterbird-counting.
create environment:

conda create -n waterbirds python=3.9 

conda activate waterbirds 

conda install --file Env.txt

pip3 install -r Envrequirements.txt

pip3 install -r requirements.txt

export PYTHONPATH=$PWD
```

with the enviroment setup we can run the quickstart notebook in the notebook folder to see the tool in action. Further details about the tool and its development can be see in the report folder.


# Instructions for annotating and training an object detector for automated counting of species

## Recommended (minimum) specs

* (For Training and Inferencing) GPU-enabled server or VM with CUDA 10.2+ and CUDNN 7.5.1+ enabled
* Quad-core CPU
* 12GB RAM
* Python version 3.9

## In terminal 1, clone repo and install requirements

```bash
git clone https://github.com/MDBAuth/bird-counting.git
cd bird-counting
pip3 install -r requirements.txt
export PYTHONPATH=$PWD
```

## Edit params.yaml to adjust the raw image and slice parameters (include path to points file). For example...

```yaml
data:
  project_name: project_name
  raw_dir: ./data/raw
  image_height: 5460
  image_width: 8192
  holdout_ratio: 0.15
  point_file: '*.pnt' 
  # NOTE: It is necessary to include the *.pnt file from DotDotGoose or other dot labelling software to filter negative samples and produce a reasonably sized training set for annotation.
  ...
slices:
  height: 546
  width: 512
  ...
```

## Save raw images into ```./data/raw/project_name/``` and run the following command to prepare the dataset for annotation

```
python3 ./scripts/split_raw_dataset.py
python3 ./start_label_studio.py
```

### After logging in, load files.txt on the label studio GUI

1. Click Create (Project)
2. Name your project
3. Click Data Import tab, then Upload Files button
4. Find and select files.txt (in root folder)
5. Once prompted to "Treat CSV/TST as:", Select "List of tasks"
6. Click on Labeling Setup and Remove Airplane and Car labels and add Bird (and/or other labels as required)
7. Click Save... Happy labeling!

## When finished labelling, instructions for processing and training on labelled data

1. Click Export and select JSON format
2. Click Export and save file into the filepath identified by 'slices:labels' in ```./repro/train/params.yaml``` (this is where the following workflow will look for the data). If the filename is different from labels.json, make sure to update the labels field in the ```./repro/train/params.yaml``` file.
3. Run the following command for training (if file is saved as described in step 2, ignore the part in parentheses). Note this may take many hours!:

```bash
dvc repro
```

## Counting birds on a brand new dataset

Simply run the following command (arguments in square brackets are optional):

<code>
python3 scripts/bird_count.py
--project-folder <i>project_folder</i>
[
  --data-root <i>data_root</i>
  --output-folder <i>path_to_output</i>
  --conf-thresh <i>confidence_threshold</i>
  --overlap <i>overlap_ratio</i>
  --sidelap <i>sidelap_ratio</i>
  --stride <i>stride_ratio</i>
  --model-path <i>model_path</i>
  --save-slices <i>save_slices</i> # 0 for False, 1 for True
]
</code>

##

* <b>--project-folder:</b> This is the parent folder where image dataset is stored.
* <b>--data-root:</b> This is root directory inside which the project-folder lies (default: ./data/raw/).
* <b>--output-folder:</b> This is the path to store the counts CSV file (default: ./models/results/).
* <b>--conf-thresh:</b> The confidence threshold (default: 0.95) allows for calibration of over- or under-counting.
* <b>--overlap:</b> This ratio determines the amount of overlap (default: 0.1) between adjacent whole images to avoid double counting.
* <b>--sidelap:</b> This ratio determines the amount of sidelap (default: 0.1) between adjacent whole images to avoid double counting.
* <b>--stride:</b> This factor determines the stride (default: 0.5) between adjacent slices during inferencing.
* <b>--model-path:</b> This is the path where the inferencing model is located (default: ./models/detection/weights/best.pth).
* <b>--save-slices:</b> This optional parameter enables saving visual predictions of non-empty slices (default: 0). Set to '1' to enable. 
* WARNING: The <b>--save-slices</b> parameter if set to True can easily consume a large amount of storage so it is advised to create a small test folder containing 1 or only very few whole images for testing and calibration purposes. 
