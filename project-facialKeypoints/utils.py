import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

def plot_data(img, landmarks, axis):
    """
    Plot image (img), along with normalized facial keypoints (landmarks)
    """
    axis.imshow(np.squeeze(img), cmap='gray') # plot the image
    landmarks = landmarks * 48 + 48 # undo the normalization
    # Plot the keypoints
    axis.scatter(landmarks[0::2], landmarks[1::2], marker='o', c='c', s=40)
    

def data_loader(csv_file, batch_size=32, train=True, is_validation=False):
    """
    Create your own data loader
    """  
    df = pd.read_csv(csv_file) 

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96) # return each images as 96 x 96 x 1

    if train:  # only FTRAIN has target columns
        y = df[df.columns[:-1]].values
        
        split = int(len(X)*0.2)  # split the train data into 80% training and 20% validation
        if is_validation:
            X, y = X[:split], y[:split]
        else:
            X, y = X[split:], y[split:]
        
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        y = y.astype(np.float32)
        
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
          
    indices = [i for i in range(len(X))]

    num = len(indices)/batch_size
    n_batches = int(num) if num.is_integer() else int(num)+1
    for i in range(n_batches):
        if train:
            yield (
                torch.from_numpy(X[i*batch_size:(i+1)*batch_size]), 
                torch.from_numpy(y[i*batch_size:(i+1)*batch_size])
            )
        else:
            yield torch.from_numpy(X[i*batch_size:(i+1)*batch_size])
            
            
def image_to_tensor(img):
    """
    Transforms and reshapes a grayscale image into an appropriate tensor shape to feed into the model
    """  

    X = img / img.max()  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96) # return each images as 1 x 96 x 96
    
    return torch.from_numpy(X)

def undo_landmarks_normalization(landmarks):
    landmarks = landmarks.detach().numpy()[0]
    return landmarks * 48 + 48 # undo the normalization