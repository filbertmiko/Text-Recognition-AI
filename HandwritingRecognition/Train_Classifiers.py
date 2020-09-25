import os
import io
import time
import pickle
import requests
import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from imageio import imread, imsave
from skimage import img_as_float, color, exposure
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import seaborn as sn
import keras
import warnings
from letters_extractor import Extract_Letters
from sklearn import metrics

start_time = time.time()

path_char=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9']
training_path = "./training_type/"
filenames = []
print("Loading Files...")
#get all the images in the above folders
#note that we are looking for specific extensions which is .png
for i in range(len(path_char)):
    filenames.append(sorted([filename for filename in os.listdir(training_path+path_char[i]) if filename.endswith('.png')]))

for i in range(len(filenames)):
    filenames[i] = [training_path+path_char[i] + '/' + filename for filename in filenames[i]]

#add the full path to all the filenames
for i in range(len(filenames)):
    print ('Number of training images -> '+str(path_char[i])+': ' + str(len(filenames[i])))

#create the list that will hold all data and the labels
data = []
labels = []

#fill the training dataset
for i in range(len(filenames)):
    for filename in filenames[i]:
        #reead the images
        image = imread(filename)
        #flatten it
        image = resize(image, (200,200))
        #get hog features
        hog_features = hog(image, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1))
        #print hog features
        data.append(hog_features)
        labels.append(path_char[i])

    print("Finished adding " + str(path_char[i]))


print("Training the SVM")
start_time = time.time()
#create the SVC
clf = SVC(kernel="linear", probability=True)
#train the SVM
clf.fit(data,labels)
print("Time taken: " + str(time.time() - start_time))
pickle.dump(clf, open("svm.detector", "wb"))

#KNN
print("Training KNN")
start_time = time.time()
knnC = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5)
#finding the optimal value of n_neighbors
#train the model using the training sets
knnC.fit(data, labels)
print("Time taken: " + str(time.time() - start_time))
#pickle it - save it to a file
pickle.dump(knnC, open("knn.detector", "wb"))
#pickle it - save it to a file

#Multilayer Perceptron NN
print("Training MLP")
start_time = time.time()
mlp = MLPClassifier()
#train
mlp.fit(np.array(data), labels)
print("Time taken: " + str(time.time() - start_time))
pickle.dump(mlp, open("mlp.detector", "wb"))

#Getting Mean Accuracy from Cross Val Score
models = []
models.append(('SVM', clf))
models.append(('KNN', knnC))
models.append(('MLP', mlp)))

results = []
names = []
mean = []
for name, model in models:
    kfold = KFold(n_splits=5, random_state=12, shuffle=True)
    cv_results = cross_val_score(model, data, labels, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    output_message = "%s| Mean=%f STD=%f" % (name, cv_results.mean(), cv_results.std())
    mean.append(cv_results.mean()*100)
    print(output_message)
