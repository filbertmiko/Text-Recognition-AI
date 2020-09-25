#Tesing SVM, KNN, Multilayer Perceptron NN
import os
import io
import pickle
import requests
import zipfile
import numpy as np
import tensorflow.compat.v1 as tf
from scipy import ndimage
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage.transform import resize
from imageio import imread, imsave
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sn

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

#SVM
clf = LinearSVC(dual=False, verbose=1)
#traing the svm
clf.fit(data, labels)

#KNN
print("Training KNN")
knnC = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
#finding the optimal value of n_neighbors
hyper_parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10]}]
knn = GridSearchCV(KNeighborsClassifier(), hyper_parameters, scoring='accuracy', cv=5)
knn.fit(data, labels)
print(knn.best_params_)
#train the model using the training sets
knnC.fit(data, labels)

#Multilayer Perceptron NN
mlp = MLPClassifier()
#train
mlp.fit(data, labels)

#comparing machine learning algorithms
models = []
models.append(('SVM', LinearSVC(dual=False, verbose=1)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('MLP', MLPClassifier()))

results = []
names = []
mean = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=12, shuffle=True)
    cv_results = cross_val_score(model, data, labels, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    output_message = "%s| Mean=%f STD=%f" % (name, cv_results.mean(), cv_results.std())
    mean.append(cv_results.mean()*100)
    print(output_message)

#predict output

test_files = ['./ocr/testing/miko.png', './ocr/testing/elsa.png', './ocr/testing/eric.png',
              './ocr/testing/miko1.png', './ocr/testing/elsa1.png', './ocr/testing/miko2.png']

recognised_SVM = []
recognised_KNN = []
recognised_MLP = []
#y_score = []
#load the test images and get the hog features
#test_image = imread('./ocr/testing/miko.png')
for filename in test_files:
    test_image = imread(filename)
    test_image = resize(test_image, (200,200))

    hog_features = hog(test_image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    result_knn = knnC.predict(hog_features.reshape(1,-1))
    result_svm = clf.predict(hog_features.reshape(1,-1))
    result_mlp = mlp.predict(hog_features.reshape(1,-1))

    #score = model.predict_proba(hog_features.reshape(1,-1))
    #y_score.append(score)
    recognised_KNN.append(result_knn)
    recognised_SVM.append(result_svm)
    recognised_MLP.append(result_mlp)

n_classes = [0,1,2]
y_true = [0, 1, 2, 0, 1, 0]
y_test = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [1,0,0]]
#y_test = []

print(y_true)

SVM_accuracy = metrics.accuracy_score(y_true, recognised_SVM)*100
KNN_accuracy = metrics.accuracy_score(y_true, recognised_KNN)*100
MLP_accuracy = metrics.accuracy_score(y_true, recognised_MLP)*100

model = ['SVM', 'KNN', 'MLP']
pos = np.arange(len(model))
pred_result = [SVM_accuracy, KNN_accuracy, MLP_accuracy]
print(pred_result)

#printing accuracy score
print("SVM Accuracy: {}".format(SVM_accuracy))
print("KNN Accuracy: {}".format(KNN_accuracy))
print("Multilayer Perceptron Accuracy: {}".format(MLP_accuracy))

fig, ax = plt.subplots()

pred_bar = plt.bar(pos-0.2, pred_result, width=0.4, label="Prediction Result")
plt.xticks(pos, model)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy %")

for pred in pred_bar:
    height = pred.get_height()
    ax.text(pred.get_x() + pred.get_width()/2., 0.99*height,'%d' % int(height) + "%", ha='center', va='bottom')

plt.legend()
plt.show()

target_names = ['Filbert', 'Elsa', 'Eric']


print("SVM Classification Report")
print("Accuracy Score: {}".format(SVM_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_SVM, average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_SVM,average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_SVM, average='micro') * 100))
print('')

print(metrics.classification_report(y_true, recognised_SVM))

print("KNN Classification Report")
print("Accuracy Score: {}".format(KNN_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_KNN,average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_KNN,average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_KNN,average='micro') * 100))
print('')

print(metrics.classification_report(y_true, recognised_KNN))

print("MLP Classification Report")
print("Accuracy Score: {}".format(MLP_accuracy))
print("F1 Score: {}".format(metrics.f1_score(y_true, recognised_MLP, average='micro') * 100))
print("Precision Score: {}".format(metrics.precision_score(y_true, recognised_MLP, average='micro') * 100))
print("Recall Score: {}".format(metrics.recall_score(y_true, recognised_MLP,average='micro') * 100))
print('')

print("SVM Classification Report")
print(metrics.classification_report(y_true, recognised_SVM, target_names=target_names))
print("KNN Classification Report")
print(metrics.classification_report(y_true, recognised_KNN, target_names=target_names))
print("MLP Classification Report")
print(metrics.classification_report(y_true, recognised_MLP, target_names=target_names))

cm_SVM = metrics.confusion_matrix(y_true, recognised_SVM)
cm_KNN = metrics.confusion_matrix(y_true, recognised_KNN)
cm_MLP = metrics.confusion_matrix(y_true, recognised_MLP)

#Confusion Matrix
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_SVM, annot=True)
plt.title("SVM")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_KNN, annot=True)
plt.title("KNN")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(cm_MLP, annot=True)
plt.title("MLP")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

