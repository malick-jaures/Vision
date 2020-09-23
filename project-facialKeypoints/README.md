
# Computer Vision Capstone Project: 
## Facial Keypoint Detection and Real-time Filtering

---
## Project Overview

This is a project from the Computer Vision Nanodegree program on [Udacity](https://www.udacity.com/course/computer-vision-nanodegree--nd891). The project's instructions and data can be found [here](https://github.com/udacity/AIND-CV-FacialKeypoints) 

---

To complete the project, we’ll combine knowledge of computer vision techniques and deep learning to build and end-to-end facial keypoint recognition system!

There are three main parts to this project:

**Part 1** : Investigating [OpenCV](https://en.wikipedia.org/wiki/OpenCV), pre-processing, and face detection

* Step 0 : Detect Faces Using a Haar Cascade Classifier
* Step 1 : Add Eye Detection
* Step 2 : De-noise an Image for Better Face Detection
* Step 3 : Blur an Image and Perform Edge Detection
* Step 4 : Automatically Hide the Identity of an Individual

**Part 2** : Training a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) in [Pytorch](https://en.wikipedia.org/wiki/PyTorch) to detect facial keypoints

* Step 5 : Build, Train and Test a CNN to Recognize Facial Keypoints

**Part 3** : Putting parts 1 and 2 together to identify facial keypoints on any image!

* Step 6 : Build a Robust Facial Keypoints Detector


**Bonus** : Add Snapchat filter (e.g. sunglass) on any image identify using detected facial keypoints

* Step 7 : Add a sunglass filter on any image


### Set up the Environment 

1. Clone this repository, and navigate to the downloaded folder.

2. Create (and activate) a new environment with Python 3.7

	- __Linux__ or __Mac__: 
	```
	conda create --name dp-cv python=3.7 numpy
	source activate dp-cv
	```
	- __Windows__: 
	```
	conda create --name dp-cv python=3.7 numpy scipy
	activate dp-cv
	```

3. Install/Update Pytorch (for this project, you may use CPU only).
__To install Pytorch__, follow [the guide](https://pytorch.org/)
    - Option 1: __To install Pytorch with CPU support only__:
	```
	conda install pytorch torchvision cpuonly -c pytorch
    ```
    Or
    ```
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
	```
	- Option 2: __To install Pytorch with GPU support__:
	```
	conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    ```
    Or
    ```
    pip install torch torchvision
	```

4. Install a few required pip packages (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in this repo, in the subdirectory `data`. In this folder are a zipped training and test set of data.

1. Navigate to the data directory
```
cd data
```

2. Load the data in python
    - Option 1: __Unzip the training and test data__:
Unzip the training and test data (in that same location). If you are in Windows, you can download this data and unzip it by double-clicking the zipped files. In Mac, you can use the terminal commands below.
```
unzip training.zip
unzip test.zip
```
You should be left with two `.csv` files of the same name. You may delete the zipped files.
    - Option 2: __Read the zipped file using pandas__:
If a zipped file data.zip contains only one csv file data.csv, it can be read as follow
```
import pandas as pd
df = pd.read_csv("data.zip")
```

*Troubleshooting*: If you are having trouble unzipping this data, you can download that same training and test data on [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data).


## Run the Notebook

1. Navigate back to the repo. (Also your source environment should still be activated at this point)
```shell
cd project-facialKeypoints
```

2. Open the notebook and follow the instructions.
```shell
jupyter notebook cv_project.ipynb
```
