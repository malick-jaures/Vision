
# Style Transfer with Deep Neural Networks

---
## Overview

I would like to thank [Udacity](www.udacity.com) for the great content course their made available for free. [Check it out here]() if you would like to take it. Here is the original paper of the [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

The main idea is that the model uses the features extracted by a pre-trained [VGG19 network](https://arxiv.org/abs/1409.1556). Indeed, the style transfer technique uses the following layers from the VGG19 network:

* conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1 for the style representation. 
* conv4_2 for the content representation.

In this notebook, we are going to transfer the [style of this image](./images/paint3.jpg) to [this image](./images/Me.jpg).


**Note**: I know this style image (image above on the left) is not the most beautiful one but I am a kind of liking it :). So, feel free to choose your own style and content images. Use the cell below to set their paths


## Set up the Environment 

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

## Data

There no data require to run the project. You only need an image to used as the style image and another image as the content image. Desppite, I put some images in the  [images folder](./images/) that you can use.


## Run the Notebook

1. Navigate back to the repo
```shell
cd style-transfer
```

2. Open the notebook and follow the instructions.
```shell
jupyter notebook style_transfer.ipynb
```
