import os
import io
import pickle
import requests
import zipfile
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.transform import resize
from imageio import imread, imsave
from skimage import img_as_float, color, exposure
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')

path_filbert = './ocr/training/Miko/'
path_elsa = './ocr/training/Elsa/'
path_eric = './ocr/training/Eric/'


#get all the images in the above folders
#note thta we are looking for specific extensions (png)
filbert_filenames = sorted([filename for filename in os.listdir(path_filbert) if (filename.endswith('.png')) ])
elsa_filenames = sorted([filename for filename in os.listdir(path_elsa) if (filename.endswith('.png')) ])
eric_filenames = sorted([filename for filename in os.listdir(path_eric) if (filename.endswith('.png')) ])

#add the full path to all the filenames
filbert_filenames = [path_filbert+filename for filename in filbert_filenames]
elsa_filenames = [path_elsa+filename for filename in elsa_filenames]
eric_filenames = [path_eric+filename for filename in eric_filenames]

print('Number of training images -> Filbert: ' + str(len(filbert_filenames)))
print('Number of training images -> Elsa: ' + str(len(elsa_filenames)))
print('Number of training images -> Eric: ' + str(len(eric_filenames)))

#create the list that will hold ALL the data and the labels
#the labels are needed for the classification task
#0 -> Filbert
#1 - > Elsa
#2 -> Eric
data = []
labels = []

#fill the training dataset

for filename in filbert_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    #hog_features = hog(image, orientations=12, pixels_per_cell=(16,16), cells_per_bock=(1,1))
    data.append(hog_features)
    labels.append(0)
print('Finished adding Filbert samples to dataset')

for filename in elsa_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    #hog_features = hog(image, orientations=12, pixels_per_cell=(16,16), cells_per_bock=(1,1))
    data.append(hog_features)
    labels.append(1)
print('Finished adding Elsa samples to dataset')

for filename in eric_filenames:
    #read the images
    image = imread(filename)
    #flatten it
    image = resize(image, (200,200))
    hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    #hog_features = hog(image, orientations=12, pixels_per_cell=(16,16), cells_per_bock=(1,1))
    data.append(hog_features)
    labels.append(2)
print('Finished adding Eric samples to dataset')


print('Training the SVM')
#create the SVC
clf = OneVsRestClassifier(SVC(kernel="linear", probability=True))
#traing the svm
clf.fit(data, labels)
#pickle it - save it to a file
clf = pickle.dump(clf, open("signature.detector","wb"))
