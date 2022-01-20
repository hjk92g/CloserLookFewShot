#It is modified from https://github.com/oscarknagg/few-shot/blob/master/scripts/prepare_mini_imagenet.py and https://github.com/oscarknagg/few-shot/blob/master/few_shot/utils.py
#This code follows a license in filelists/miniImagenet/prepare_mini_imagenet.LICENSE
"""
Run this script to prepare the miniImageNet dataset.

This script uses the 100 classes of 600 images each used in the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from a link (https://goo.gl/e3orz6).

1. Download files from https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view and place in
    filelists/miniImagenet/images
2. Run this script
"""
from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

try:
    os.mkdir('images_background')
except:
    pass

try:
    os.mkdir('images_evaluation')
except:
    pass

print(os.getcwd())

DATA_PATH = '../' #None

def mkdir(dir):
    """Create a directory, ignoring exceptions
    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions
   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass

# Clean up folders
rmdir(DATA_PATH + '/miniImagenet/images_background')
rmdir(DATA_PATH + '/miniImagenet/images_evaluation')
mkdir(DATA_PATH + '/miniImagenet/images_background')
mkdir(DATA_PATH + '/miniImagenet/images_evaluation')

# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/miniImagenet/images/'):
    for f in files:
        if f.endswith('.jpg'):
            classes.append(f[:-12])

classes = list(set(classes))

# Train/test split
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:80], classes[80:]

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + f'/miniImagenet/images_background/{c}/')

for c in evaluation_classes:
    mkdir(DATA_PATH + f'/miniImagenet/images_evaluation/{c}/')

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/miniImagenet/images'):
    for f in tqdm(files, total=600*100):
        if f.endswith('.jpg'):
            class_name = f[:-12]
            image_name = f[-12:]
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = f'{root}/{f}'
            dst = DATA_PATH + f'/miniImagenet/{subset_folder}/{class_name}/{image_name}'
            shutil.copy(src, dst)
