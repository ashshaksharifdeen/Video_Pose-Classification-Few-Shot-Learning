import argparse
import csv
import glob
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from tqdm._tqdm import trange
from PIL import Image
import tensorflow as tf
import cv2
import time

META_TRAIN_DIR = 'CAUCAFall/train'
metatrain_folders = [
                                    os.path.join(META_TRAIN_DIR, label, sublabel)\
                                            for label in os.listdir(META_TRAIN_DIR)\
                                            for sublabel in os.listdir((os.path.join(META_TRAIN_DIR, label)))\
                                                if os.path.isdir(os.path.join(META_TRAIN_DIR, label, sublabel))]
for metatrain_folder in  metatrain_folders:
            #each folder have several images
            image_file = [os.path.join(metatrain_folder, image) \
                         for image in os.listdir(metatrain_folder)\
                                    if image.lower().endswith(('.png', '.jpg', '.jpeg'))]

        
            
            for image_file in image_file:
                img = cv2.imread(image_file)
                cv2.imshow("image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()   