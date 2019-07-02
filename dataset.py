import pandas as pd
import os, glob
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


def load_position(input_path):
    # load with pandas
    cols = ['x', 'y']
    df = pd.read_csv(input_path, sep=",", header=None, names=cols)
    return df


def load_images(input_path):
    images = []
    imgw, imgh = 224, 224
    imagelist = os.listdir(input_path)
    imagelist = [x for x in imagelist if x[-4:] == '.jpg']
    for house_path in imagelist:
        try:
            image = cv2.imread(input_path + house_path)
            image = cv2.resize(image, (imgw, imgh))
            # image = image/255.0
        except:
            print("failed_for: ", house_path)

        images.append(image)

    return np.array(images)
