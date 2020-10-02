
# Style Transfer with Deep Neural Networks

---
## Overview

I would like to thank [Udacity](www.udacity.com) for the great content course their made available for free. [Check it out here]() if you would like to take it. Here is the original paper of the [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

The main idea is that the model uses the features extracted by a pre-trained [VGG19 network](https://arxiv.org/abs/1409.1556). Indeed, the style transfer technique uses the following layers from the VGG19 network:

* conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 for the style representation. 
* conv4_2 for the content representation.

In this notebook, we are going to transfer the style of the image on the left to the image of me on the right.

<table>
  <tr>
    <td> <img src="images/paint3.jpg" alt="Drawing" style="width: 250px; height: 300px;"/> </td>
    <td> <img src="images/Me.jpg" alt="Drawing" style="width: 250px; height: 300px;"/> </td>
  </tr>
</table>

**Note**: I know this style image (image above on the left) is not the most beautiful one but I am a kind of liking it :). So, feel free to choose your own style and content images. Use the cell below to set their paths




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
jupyter notebook style_transfer.ipynb
```
