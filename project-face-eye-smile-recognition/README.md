
# Face, Eye and Smile Recognition

---
## Project Overview

This project is about using the implementation in [OpenCV](https://en.wikipedia.org/wiki/OpenCV) of the Haar cascade classifiers described in [this paper](https://www.researchgate.net/publication/3940582_Rapid_Object_Detection_using_a_Boosted_Cascade_of_Simple_Features). We'll be detecting faces, eyes, and smiles within images. To do that you are going to use the following pre-trained classifier files:

* haarcascade_frontalface_default.xml for faces,
* haarcascade_eye.xml for eyes,
* haarcascade_smile.xml for smiles.

**Note**: Find more of these files [here](https://github.com/opencv/opencv/tree/master/data/haarcascades)

The detection is performed either on your PC's webcam or on a video file named ***young_sheldon_s02_p02.mp4*** stored in [videos folder](./videos/) and you can decide to store the video with the detection in a file. That video file will be in the [videos folder](./videos/) and named ***face_detector_video.avi***. You can also choose to split the RGB channels into separate R, G, and B channel.


## Set up the Environment 

1. Clone this repository, and navigate to the downloaded folder.

2. Install a few required packages 
```
pip install -U numpy 
pip install -U opencv-python
```
Or
```
conda install -c anaconda numpy
conda install -c conda-forge opencv
```

## Data

The only data require to run the project are the pre-trained Haar cascade classifier files in [haarcascades folder](./haarcascades/). You'll find the ones used here and three more files.

If you decide to perform the detection on a movie files, use ***young_sheldon_s02_p02.mp4*** stored in [videos folder](./videos/)

**Note**: You can detect faces on any movie file of your choice. The simplest way is to put the video in [videos folder](./videos/) and then you just need to replace ***young_sheldon_s02_p02.mp4*** by that file in the code.

## Run the Notebook

1. Navigate back to the repo
```shell
cd project-face-eye-smile-recognition
```

2. Open the notebook and follow the instructions.
```shell
jupyter notebook face_recognition.ipynb
```
