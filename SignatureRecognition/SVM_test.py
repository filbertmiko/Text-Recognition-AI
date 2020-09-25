import os
import io
import pickle
import time
import numpy as np
from scipy import ndimage
from skimage.feature import hog
from skimage.transform import resize
from imageio import imread, imsave
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib as mpl
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from itertools import chain
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier

#from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix, roc_curve # roc curve tools

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
#load the detector
clf = pickle.load(open("signature.detector","rb"))

test_files = ['./ocr/testing/miko.png', './ocr/testing/elsa.png', './ocr/testing/eric.png',
              './ocr/testing/miko1.png', './ocr/testing/elsa1.png', './ocr/testing/miko2.png']
recognised = []
y_score = []
#load the test images and get the hog features
#test_image = imread('./ocr/testing/miko.png')
for filename in test_files:
    test_image = imread(filename)
    test_image = resize(test_image, (200,200))

    hog_features = hog(test_image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    result_type = clf.predict(hog_features.reshape(1,-1))
    score = clf.predict_proba(hog_features.reshape(1,-1))
    y_score.append(score)
    recognised.append(result_type)

y_true = [0, 1, 2, 0, 1, 0]
y_test = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [1,0,0]]

for i in range(len(recognised)):
    print(test_files[i])
    if recognised[i] == 0:
        print("This is Filbert's Signature")
    elif recognised[i] == 1:
        print("This is Elsa's Signature")
    elif recognised[i] == 2:
        print("This is Eric's Signature")
    print("")
print("")

#print("Time taken: " + str(time.time() - start_time) + " seconds")

print("confusion matrix: ")
print(metrics.confusion_matrix(y_true, recognised))

precision = metrics.precision_score(y_true, recognised, average='weighted')
recall = metrics.recall_score(y_true, recognised, average='weighted')
accuracy = metrics.accuracy_score(y_true, recognised)
f1 = metrics.f1_score(y_true, recognised, average='weighted')

print("Precision Score: " + str(precision)) #TP / TP+FP
print("Recall Score: " + str(recall)) # TP/TP+FN
print("Accuracy Score: " + str(accuracy)) #total correct
print("f1 Score Score: " + str(f1))

plt.clf()

y_score = list(chain.from_iterable(y_score))
n_classes = np.array([0,1,2])
y_test = np.array(y_test)
y_true = np.array(y_true)
y_score = np.array(y_score)

fpr = dict()
tpr = dict()
roc_auc = dict

#Plot ROC Curve for each Class because if we plot it altogether
#it would overlap one another and so we won't be able to see the other 2
for i in range(len(n_classes)):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
    #roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    if i == 0:
        plt.plot(fpr[i], tpr[i],lw=2,label='ROC curve for Filbert')
    elif i == 1:
        plt.plot(fpr[i], tpr[i],lw=2,label='ROC curve for Elsa')
    elif i == 2:
        plt.plot(fpr[i], tpr[i],lw=2,label='ROC curve for Eric')


    plt.plot([0, 1], [0, 1], 'k-', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Signature')
    plt.legend(loc="lower right")
    plt.show()

#Plot Histogram
plt.figure(dpi=150)
plt.hist(y_score, bins=20)
plt.title('Classification Probabilites')
plt.xlabel('Probability')
plt.ylabel('# Of instances')
plt.xlim([0.5,1.0])
plt.show()
