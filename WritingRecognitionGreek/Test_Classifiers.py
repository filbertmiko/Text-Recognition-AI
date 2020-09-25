import os
import io
import time
import pickle
import requests
import cv2
import numpy as np
from scipy import ndimage
from sys import stdout
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

path_char=['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']

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
clf = SVC(kernel='linear', probability=True)
#train the SVM
clf.fit(data,labels)
print("Time taken: " + str(time.time() - start_time))

#KNN
print("Training KNN")
start_time = time.time()
knnC = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=5)
#train the model using the training sets
knnC.fit(np.array(data), labels)
print("Time taken: " + str(time.time() - start_time))

#Multilayer Perceptron NN
print("Training MLP")
start_time = time.time()
mlp = MLPClassifier()
#train
mlp.fit(data, labels)
print("Time taken: " + str(time.time() - start_time))

#Getting Mean Accuracy from Cross Val Score
models = []
models.append(('SVM', clf))
models.append(('KNN', knnC))
models.append(('MLP', mlp))

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

extract = Extract_Letters()
start_time = time.time()

test_file_path = './ocr/testing/greekgod_1.png'

recognised_SVM = []
recognised_KNN = []
recognised_MLP = []
y_score = []

print("Extracting File...")
#extract test image so we can get the letters
letters = extract.extractFile(test_file_path)

print("Recognised text: ")
for letter in letters:
    letter = resize(letter, (200,200))
    hog_features = hog(letter, orientations = 8, pixels_per_cell=(16,16), cells_per_block=(1,1))
    result_svm = clf.predict(hog_features.reshape(1,-1))
    result_knn = knnC.predict(hog_features.reshape(1,-1))
    result_mlp = mlp.predict(hog_features.reshape(1,-1))

    recognised_SVM.append(result_svm)
    recognised_KNN.append(result_knn)
    recognised_MLP.append(result_mlp)

#truth = 'ζευσιστηεβροτηεροφποσειδονανδηαδεσηειστηεγοδοφτηυνδερανδτηεστρονγεσταφτερξηρονοσ'
#truth = 'ατηεναποσειδονδιονψσυσηερααπηροδιτεαπολλοαρεσζευσαρτεμισηαδεσηεπηαεστυσηερμεσδεμετερηεστιανψχκρατοσνικειρισπανγαεα'
truth = 'ζευσηεραποσειδονδεμετεραρεσατηενααπολλοαρτεμισαπηροδιτεηερμεσδιονψσυσηαδεσηψπνοσνικενεμεσισιρισηεξατεψξηε'
#truth = 'ηελλοτηισισατεχττοτεσττηερεξογνιτιονσψστεμφορφορειγνλανγυαγε'
truth = truth.lower()

y_true = []
for i in range(len(truth)):
    y_true.append(truth[i])

#print("Extracted: " + str(len(recognised)))
print("# of letter in the test image: " + str(len(truth)))

SVM_accuracy = metrics.accuracy_score(y_true, recognised_SVM)*100
KNN_accuracy = metrics.accuracy_score(y_true, recognised_KNN)*100
MLP_accuracy = metrics.accuracy_score(y_true, recognised_MLP)*100

model = ['SVM', 'KNN', 'MLP']
pos = np.arange(len(model))
pred_result = [SVM_accuracy, KNN_accuracy, MLP_accuracy]
print(pred_result)

fig, ax = plt.subplots()

pred_bar = plt.bar(pos, pred_result, width=0.4, label="Prediction Result")
#mean_bar = plt.bar(pos+0.2, mean, width=0.4, label="Mean")
plt.xticks(pos, model)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy %")

for pred in pred_bar:
    height = pred.get_height()
    ax.text(pred.get_x() + pred.get_width()/2., 0.99*height,'%d' % int(height) + "%", ha='center', va='bottom')

#for mean in mean_bar:
#    height = mean.get_height()
#    ax.text(mean.get_x() + mean.get_width()/2., 0.99*height,'%d' % int(height) + "%", ha='center', va='bottom')

plt.legend()
plt.show()

print("SVM Classification Report")
print("SVM Precision: {}".format(metrics.precision_score(y_true, recognised_SVM, average='micro')*100))
print("SVM Recall: {}".format(metrics.recall_score(y_true, recognised_SVM, average='micro')*100))
print("SVM Accuracy: {}".format(SVM_accuracy))
print("SVM F1 Score: {}".format(metrics.f1_score(y_true, recognised_SVM, average='micro')*100))

print("KNN Classification Report")
print("KNN Precision: {}".format(metrics.precision_score(y_true, recognised_KNN, average='micro')*100))
print("KNN Recall: {}".format(metrics.recall_score(y_true, recognised_KNN, average='micro')*100))
print("KNN Accuracy: {}".format(KNN_accuracy))
print("KNN F1 Score: {}".format(metrics.f1_score(y_true, recognised_KNN, average='micro')*100))

print("MLP Classification Report")
print("Multilayer Perceptron Precision: {}".format(metrics.precision_score(y_true, recognised_MLP, average='micro')*100))
print("Multilayer Perceptron Recall: {}".format(metrics.recall_score(y_true, recognised_MLP, average='micro')*100))
print("Multilayer Perceptron Accuracy: {}".format(MLP_accuracy))
print("Multilayer Perceptron F1 Score: {}".format(metrics.f1_score(y_true, recognised_MLP, average='micro')*100))

print("Confusion matrix for SVM: ")
cm_SVM = metrics.confusion_matrix(y_true, recognised_SVM)
print("Confusion matrix for KNN: ")
cm_KNN = metrics.confusion_matrix(y_true, recognised_KNN)
print("Confusion matrix for MLP: ")
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
