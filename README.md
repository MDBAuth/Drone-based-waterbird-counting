# Drone based waterbird counting 

This project is a proof-of-concept tool to automatically count colonially nesting waterbirds from drone imagery, developed initialy on Straw-necked Ibis at Narran Lakes.

The purpose of developing a tool to automatically count colonially nesting waterbirds is to allow accurate,
repeatable and timely monitoring of large colonies. It is envisioned that a user would be able to input drone
imagery and with minimal technical skill derive an accurate count of waterbirds.

## Quickstart instructions  

This project contains 3 requirements files: 

|File|Type  |Contents  |
|--|--|--|
|requirements.txt|pip| minimal requirment to run the model and label datasets|
|Env.txt| Conda | the python 3.9 enviroment we are runing this project on
|Envrequirements.txt| pip | the packages we have installed on our enviroment|

We have been running this project on a NC4AS_T4_V3 in AzureML, to download this project and set up this enviroment run the following in a terminal

NOTE: Git lfs is used for the larger files https://git-lfs.github.com/

```bash
sudo apt-get install git-lfs

git lfs install

git clone https://github.com/MDBAuth/Drone-based-waterbird-counting.git

cd Drone-based-waterbird-counting
```
create environment:
```bash
conda create -n waterbirds python=3.9 

conda activate waterbirds 

conda install --file Env.txt

pip3 install -r Envrequirements.txt

pip3 install -r requirements.txt

export PYTHONPATH=$PWD
```

with the enviroment setup we can run the quickstart notebook in the notebook folder to see the tool in action. Further details about the tool and its development can be see in the report folder.


# <center>Instruction Guide<center>

## <center>Annotating and training an object detector for automated counting of species<center>

##### <center>IMPORTANT NOTE: Parameters in ***params.yaml*** are used throughout the repository.<center>

## Table of Contents

1. Definitions
1. Preparing and Labelling a New Training Set
1. Training a Model on a Training Set
1. Counting Species using a Trained Model

## 1. Definitions

The definitions below will be referenced throughout this document.

### 1.1. _Raw Dataset_

- **Project**: Also called a *mission*, whereby a drone is sent to collect images over a specified terrestrial area, using a push broom or mosaicking pattern. Images are saved in a folder (e.g. ```project_name```).
- **Dataset**: A set of PNG or JPG images typically located in a single folder. These images must all have the same dimensions.
- **Overlap**: When using a push broom or mosaic pattern of overhead image collection, the overlap is the fractional overlap between vertically adjacent images.
- **Sidelap**: When using a push broom or mosaic pattern of overhead image collection, the overlap is the fractional overlap between horizontally adjacent images.

### 1.2. _Machine Learning & Image Processing_

- **Slices**: Due a combination of the limited memory capacity of processors as well as the relatively small objects of interest (e.g. birds), high-resolution images must be processed in smaller pieces. In this guide we refer to these pieces as slices. [Click here](https://tinyurl.com/4yv44zve) for more information.
- **Training Set**: A set of slices that have been labelled with bounding boxes. This will typically consist of a data folder containing the sliced images themselves, and a labels.json file containing the label metadata.
- **Stride**: Object detection algorithms are less performant near the edges of images, therefore allowing a fractional overlap (a.k.a. stride) between slices allows a ML model to improve detection performance. **Note**: trimming the prediction area by the same stride value avoids double-counting.

## 2. Preparing and Labelling a New Training Set

### 2.1. _Loading a Dataset_

The first step to preparing a **Dataset** is to copy a **Project** into the root directory, which is set by default to ```./data/raw```. With this pattern, raw images should be located ```./data/raw/project_name/``` directory.

### 2.2. _Pre-Annotations Using DotDotGoose_

Use DotDotGoose software to pre-annotate labels. The output of a DotDotGoose labelling job will be a PNT file with a ```.pnt``` extension, which must be placed inside the ```./data/raw/project_name/``` directory along with the raw images. **NOTE**: There must only be one PNT file per project.

### 2.3. _Update ```params.yaml```_

To process the correct **Project**, the *image_height*, *image_width* and *project_name* fields must be updated to reflect the **Project** attributes. The *project_name* is simply the name of the folder inside of which the raw images are stored, while the *image_height* and *image_width* fields are the dimensions of the original high resolution (raw) images.

### 2.4. _Prepare Data for Labelling_

Once a **Dataset** is prepared, including a single PNT file, and the ```params.yaml``` is updated as described above, the following command should be executed to prepare the dataset for labelling.

<center>
```python3 ./scripts/split_raw_dataset.py```
</center>

### 2.5. _Labelling Data using Label Studio_

Following the preparation of a dataset for labelling, follow the instructions below to begin labelling data using Label Studio [(a commercially available third party product)](https://labelstud.io/guide/).

**NOTES**:

- this repository is only compatible with Label Studio annotation formatting, use of other annotation software will require work to ensure compatibility
- the following steps should be run on a local computerl they are not guaranteed to work on cloud servers

1. Run the following command

    <center>```python3 ./scripts/start_label-studio.py```</center>

2. Login to Label Studio
3. Load files.txt (located in the root directory) on the label studio GUI
4. Click Create (Project) and name your project
5. Click the Data Import tab*, then Upload Files button
6. Find and select files.txt (in root folder)
7. Once prompted to "Treat CSV/TST as:", Select "List of tasks"
8. Click on Labeling Setup and Remove Airplane and Car labels and add Bird (and/or other labels as required)
9. Click Save and begin labelling ([click here](https://blog.superannotate.com/introduction-to-bounding-box-annotation-best-practices/) for labelling best practices)
10. Once finished labelling, Click Export and select JSON format
11. Click Export and save the output file into ```./data/labels/``` directory. Update the relevant filename or filepath in ```params.yaml``` (located in ```slices:labels```)

### 2.6. _Prepare Training Set from Labelled Data_

Once the steps in Section 2.5 have been followed to completion, first update the following fields in the __training__ section of ```params.yaml``` as follows:

- ```train_data_dir```: ```./data/training-sets/project_name/data```
- ```train_coco```: ```./data/training-sets/project_name/labels.json```

Finally, run the following command to prepare the **Training Set**:

<center>```python3 scripts/prepare_training_set.py```</center>

### 2.7. _Adding to an Existing Training Set_

If all of the previous steps were followed for two projects (let's call them ```project_1``` and ```project_2```), then there should be two different ```labels.json``` files each in their respective folders, as well as two corresponding ```training-sets/project_name/data/`` directories containing the image (slices) data inputs. The following methodology can be used to combine these two **Training Set** (this process can be used to combine any number of **Training Sets**):

1. Create a new folder called ```./data/training-sets/combined_project_name/data/```
2. Copy contents of ```./data/training-sets/project_1/data/``` and ```./data/training-sets/project_2/data/``` into folder created in Step 1
3. Create an empty JSON file ```./data/training-sets/combined_project_name/labels.json```
4. Concatenate the two JSON files from ```project_1``` and ```project_2``` and copy the output into the JSON file created in Step 3
5. Update the relevant fields in ```params.yaml``` (see Section 2.6)

## 3. Training a Model on a Training Set

Once a **Training Set** has been prepared according to the steps described in Section 2, first update the _project_name_ of the following field in the __models__ section of ```params.yaml```:

- ```best_model```: ```./models/detection/weights/project_name.pth```

as well as the __training__ section of ```params.yaml```:

- ```train_data_dir```: ```./data/training-sets/project_name/data```
- ```train_coco```: ```./data/training-sets/project_name/labels.json```

Finally, run the following command to start a training job:

<center>```python3 scripts/train_mdba.py```</center>

While this should "just work", hyperparameters may be tuned, and data augmentation techniques can be added (see lines 96-107 in ```./utils/train_utils.py``` regarding the use of ```torchvision.transforms```).

## 4. Counting Species using a Trained Model

### 4.1 Save Raw Images

Simply follow the instructions in Section 2.1. Keep the project name in mind for the next step.

### 4.2. Run Bird Counting Script

Update the relevant parameters in ```params.yaml``` then run the following command (arguments in square brackets are optional):

<code>
python3 scripts/bird_count.py
--project-name <i>```project_name```</i></br></br>
[</br>
</br>```*These optional arguments can be set here or in params.yaml```</br></br>
  --data-root <i>```data_root```</i></br>
  --output-folder <i>```path_to_output```</i></br>
  --conf-thresh <i>```confidence_threshold```</i></br>
  --overlap <i>```overlap_ratio```</i></br>
  --sidelap <i>```sidelap_ratio```</i></br>
  --stride <i>```stride_ratio```</i></br>
  --model-path <i>```model_path```</i></br>
  --save-slices <i>```save_slices```</i> # 0 for False, 1 for True</br></br>
]</code>

where the parameters are defined as follows:

* <b>--project-name:</b> This is the parent folder where image dataset is stored.
* <b>--data-root:</b> This is root directory inside which the project-folder lies (default: ./data/raw/).
* <b>--output-folder:</b> This is the path to store the counts CSV file (default: ./models/results/).
* <b>--conf-thresh:</b> The confidence threshold (default: 0.95) allows for calibration of over- or under-counting.
* <b>--overlap:</b> This ratio determines the amount of overlap (default: 0.1) between adjacent whole images to avoid double counting.
* <b>--sidelap:</b> This ratio determines the amount of sidelap (default: 0.1) between adjacent whole images to avoid double counting.
* <b>--stride:</b> This factor determines the stride (default: 0.5) between adjacent slices during inferencing.
* <b>--model-path:</b> This is the path where the inferencing model is located (default: ./models/detection/weights/best.pth).
* <b>--save-slices:</b> This optional parameter enables saving visual predictions of non-empty slices (default: 0). Set to '1' to enable. 
* WARNING: The <b>--save-slices</b> parameter if set to True can easily consume a large amount of storage so it is advised to create a small test folder containing 1 or only very few whole images for testing and calibration purposes.

See an example of a [detection output](https://ibb.co/C1M8rZT) (when save-slices is set to 1 or True).

### Supported through funding from the Australian Government Murray–Darling Water and Environment Research Program
